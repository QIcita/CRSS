"""Finite acquisition/preparation/acceptance/training workflow with resource gates."""
import argparse
import fcntl
import os
import shutil
import subprocess
import time
from pathlib import Path
from .common import BASE, ROOT, PYTHON, read, save, runtime, sha, verify


def live(pid, root):
    try:
        return os.readlink('/proc/' + str(pid) + '/cwd') == str(root)
    except (OSError, PermissionError):
        return False


def resources():
    gpu = {}
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.total,memory.free',
                                   '--format=csv,noheader,nounits'], text=True)
    for line in raw.splitlines():
        i, total, free = map(float, line.split(','))
        gpu[int(i)] = dict(total=total, free=free, effective_free=free)
    raw = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_memory',
                                   '--format=csv,noheader,nounits'], text=True)
    used = {}
    for line in raw.splitlines():
        try:
            p, u = map(float, line.split(','))
            used[int(p)] = u
        except ValueError:
            pass
    reservations = {}

    def visit(value, root):
        if isinstance(value, dict):
            if isinstance(value.get('pid'), int) and isinstance(value.get('gpu'), int):
                pid, device = value['pid'], value['gpu']
                if device in gpu and live(pid, root):
                    reserve = value.get('reservation_mib')
                    if reserve is None and value.get('id'):
                        path = root / 'acceptance' / (value['id'] + '.json')
                        if path.exists() and 'peak_gpu_bytes' in read(path):
                            reserve = max(2200, int(read(path)['peak_gpu_bytes'] / 2**20 * 1.25) + 1200)
                    if reserve is not None:
                        reservations[pid] = (device, max(float(reserve), reservations.get(pid, (device, 0))[1]))
            for item in value.values():
                if isinstance(item, (list, dict)):
                    visit(item, root)
        elif isinstance(value, list):
            for item in value:
                visit(item, root)
    for root in BASE.glob('CPSF*'):
        for filename in ['campaign_status.json', 'queue_status.json']:
            p = root / filename
            if p.exists():
                try:
                    visit(read(p), root)
                except (ValueError, KeyError):
                    pass
    for pid, (device, reserve) in reservations.items():
        gpu[device]['effective_free'] -= max(0, reserve - used.get(pid, 0))
    ram = next(int(x.split()[1]) * 1024 for x in open('/proc/meminfo') if x.startswith('MemAvailable:'))
    return dict(gpus=gpu, ram_available=ram, disk_free=shutil.disk_usage(ROOT).free,
                outstanding_reservations={str(p): {'gpu': d, 'reserve_mib': r, 'used_mib': used.get(p, 0)}
                                          for p, (d, r) in reservations.items()})


def run_stage(stage, args, expected, need=None, dataenv=False):
    frozen = ROOT / 'training_protocol.json'
    if frozen.exists():
        for path, expected_sha in read(frozen)['files'].items():
            assert sha(path) == expected_sha, ('campaign source drift', path)
    if expected.exists():
        value = read(expected)
        assert value.get('passed', value.get('complete', value.get('state') == 'complete'))
        return
    failure = ROOT / 'failures' / ('queue_' + stage + '.json')
    assert not failure.exists(), 'Inspect failure before any retry'
    while True:
        resource = resources()
        choices = [(v['effective_free'], device) for device, v in resource['gpus'].items()
                   if need is None or v['effective_free'] >= need + 512]
        if choices and resource['ram_available'] > 12 * 2**30 and resource['disk_free'] > 12 * 2**30:
            device = max(choices)[1]
            break
        save(ROOT / 'queue_status.json', dict(phase='waiting_' + stage, required_mib=need,
             resources=resource, pid=os.getpid(), time=time.time()))
        time.sleep(20)
    env = runtime(device)
    if need:
        env['MICRONS_GPU_LIMIT_MIB'] = str(need)
    python = ROOT / 'dataenv/bin/python' if dataenv else PYTHON
    log = ROOT / 'logs' / (stage + '.log')
    log.parent.mkdir(exist_ok=True)
    with log.open('ab', buffering=0) as stream:
        process = subprocess.Popen([str(python), '-u', *args], cwd=ROOT, env=env,
                    stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    receipt = dict(stage=stage, pid=process.pid, gpu=device, reservation_mib=need or 0,
                   time=time.time(), protocol_sha256=sha(ROOT / 'protocol.json'))
    save(ROOT / 'launches' / (stage + '.json'), receipt)
    while process.poll() is None:
        save(ROOT / 'queue_status.json', dict(phase=stage, active=receipt, pid=os.getpid(), time=time.time()))
        time.sleep(5)
    if process.returncode != 0 or not expected.exists():
        save(failure, dict(stage=stage, returncode=process.returncode, expected=str(expected), log=str(log)))
        raise RuntimeError('Stage failed: ' + stage)
    value = read(expected)
    assert value.get('passed', value.get('complete', value.get('state') == 'complete')), 'Stage did not pass'


def main(pilot):
    lock = (ROOT / ('pilot_queue.lock' if pilot else 'queue.lock')).open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    verify()
    names = ['sgformer', 'r38_dense', 'utonia_scratch_r2']
    if not pilot:
        assert read(ROOT / 'pilot_models_complete.json')['complete']
        assert read(ROOT / 'acceptance/geometry_contracts.json')['passed']
        assert (ROOT / 'training_protocol.json').exists()
        run_stage('acquisition', ['-m', 'tools.cpsf_microns.acquire'], ROOT / 'status/acquisition_complete.json', dataenv=True)
        run_stage('preparation', ['-m', 'tools.cpsf_microns.prepare'], ROOT / 'manifest.json', need=1500)
    for name in names:
        need = {'sgformer': 2000, 'r38_dense': 4800, 'utonia_scratch_r2': 7200}[name]
        args = ['-m', 'tools.cpsf_microns.train', '--model', name, '--accept']
        if pilot:
            args += ['--pilot']
        stage = name + ('_pilot' if pilot else '_acceptance')
        run_stage(stage, args, ROOT / 'acceptance' / (name + ('_pilot' if pilot else '') + '.json'), need=need)
    if pilot:
        save(ROOT / 'pilot_models_complete.json', dict(complete=True, models=names))
    else:
        for name in names:
            peak = read(ROOT / 'acceptance' / (name + '.json'))['peak_gpu_bytes'] / 2**20
            need = max(2000, peak * 1.2 + 768)
            run_stage(name + '_training', ['-m', 'tools.cpsf_microns.train', '--model', name],
                      ROOT / 'results' / name / 'metrics.json', need=need)
        run_stage('geometry_baselines', ['-m', 'tools.cpsf_microns.geometry_baselines'], ROOT / 'geometry_baselines.json')
        result = {name: read(ROOT / 'results' / name / 'metrics.json') for name in names}
        save(ROOT / 'comparison.json', dict(complete=True, results=result, seed=42,
             manifest_sha256=sha(ROOT / 'manifest.json')))
    save(ROOT / 'queue_status.json', dict(phase='pilot_complete' if pilot else 'complete', models=names, pid=os.getpid()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pilot', action='store_true')
    a = parser.parse_args()
    try:
        main(a.pilot)
    except Exception as exc:
        save(ROOT / ('pilot_queue_failure.json' if a.pilot else 'queue_failure.json'),
             dict(error_type=type(exc).__name__, error=str(exc), pid=os.getpid()))
        raise
