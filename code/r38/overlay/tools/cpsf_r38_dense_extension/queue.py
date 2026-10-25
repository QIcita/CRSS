"""One finite task; reserve space for live jobs before launching each stage."""
import fcntl
import os
import shutil
import subprocess
import time
import traceback
from pathlib import Path
from .common import ROOT, BASE, DENSE, R38, R34, INPUT, FINE, PYTHON, read, save, sha, verify


def env(gpu, fraction):
    value = dict(os.environ)
    value.update(read(BASE / 'CPSF_EXTERNAL_BASELINES_PHASE3_20261012/runtime_env.json'))
    value.update(CUDA_VISIBLE_DEVICES=str(gpu), CPSF_R38_DENSE_EXTENSION_ROOT=str(ROOT),
                 CPSF_R38_DENSE_ROOT=str(DENSE), CPSF_CMM_PAIR_ROOT=str(R38),
                 CPSF_R34_FPS_ROOT=str(R34), CPSF_REBENCH_ROOT=str(INPUT),
                 PYTHONPATH=str(INPUT / 'deps'), CPSF_GPU_FRACTION=str(fraction),
                 OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', CUBLAS_WORKSPACE_CONFIG=':4096:8')
    return value


def live_at(pid, root):
    try:
        return os.readlink('/proc/' + str(pid) + '/cwd') == str(root)
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return False


def active_entries(state):
    """Normalize single, list, and job-keyed active records from other queues."""
    active = state.get('active')
    if isinstance(active, dict):
        if 'pid' in active:
            return [active]
        return [dict(value, id=value.get('id', key)) for key, value in active.items()
                if isinstance(value, dict) and 'pid' in value]
    if isinstance(active, list):
        return [value for value in active if isinstance(value, dict) and 'pid' in value]
    return []


def resources():
    output = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.total,memory.free',
                                      '--format=csv,noheader,nounits'], text=True)
    gpu = {int(i): dict(total=float(total), free=float(free), effective_free=float(free))
           for i, total, free in (line.split(',') for line in output.splitlines())}
    output = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_memory',
                                      '--format=csv,noheader,nounits'], text=True)
    used = {int(pid): float(value) for pid, value in (line.split(',') for line in output.splitlines())}
    reservations = {}
    # Include later TED/MICrONS queues as well as the original parent campaigns.
    for root in BASE.glob('CPSF_*'):
        for filename in ('campaign_status.json', 'queue_status.json'):
            path = root / filename
            if not path.exists():
                continue
            for entry in active_entries(read(path)):
                pid, device = entry.get('pid'), entry.get('gpu')
                if not isinstance(pid, int) or device not in gpu or not live_at(pid, root):
                    continue
                reserve = entry.get('reservation_mib')
                if reserve is None:
                    identifier = entry.get('id')
                    gate = root / 'acceptance' / (str(identifier) + '.json') if identifier else None
                    # Unknown active jobs retain a conservative reservation.
                    reserve = (max(2200, int(read(gate)['peak_gpu_bytes']/2**20*1.25)+1200)
                               if gate and gate.exists() and 'peak_gpu_bytes' in read(gate) else 20000)
                previous = reservations.get(pid)
                if previous is None or reserve > previous['reserve_mib']:
                    reservations[pid] = dict(pid=pid, gpu=device, reserve_mib=reserve)
    reserves = []
    for pid, entry in reservations.items():
        extra = max(0, entry['reserve_mib']-used.get(pid, 0))
        gpu[entry['gpu']]['effective_free'] -= extra
        reserves.append(dict(entry, additional_mib=extra))
    ram = next(int(line.split()[1])*1024 for line in open('/proc/meminfo') if line.startswith('MemAvailable:'))
    return dict(gpus=gpu, parent_reservations=reserves, ram_available=ram, disk_free=shutil.disk_usage(ROOT).free)


def main():
    lock = open(ROOT / 'queue.lock', 'a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    verify()
    # Refuse to duplicate an orphaned child after controller interruption.
    for proc in Path('/proc').iterdir():
        if proc.name.isdigit() and int(proc.name) != os.getpid() and live_at(int(proc.name), ROOT):
            try:
                cmd = (proc / 'cmdline').read_bytes()
            except FileNotFoundError:
                continue
            assert b'tools.cpsf_r38_dense_extension.worker' not in cmd and b'tools.cpsf_r38_dense_extension.local_equivalence' not in cmd, ('worker already running', proc.name)
    stages = [
        ('local_equivalence', ['-m', 'tools.cpsf_r38_dense_extension.local_equivalence'], ROOT / 'h01_local_equivalence.json', 3500),
        ('smoke', ['-m', 'tools.cpsf_r38_dense_extension.worker', '--task', 'flywire_fine', '--smoke'], ROOT / 'acceptance/flywire_fine_smoke.json', 5000),
        ('acceptance', ['-m', 'tools.cpsf_r38_dense_extension.worker', '--task', 'flywire_fine', '--accept'], ROOT / 'acceptance/r38_flywire_fine.json', 10500),
        ('training', ['-m', 'tools.cpsf_r38_dense_extension.worker', '--task', 'flywire_fine'], ROOT / 'results/r38_flywire_fine/metrics.json', None),
    ]
    for stage, args, expected, need in stages:
        if expected.exists():
            result = read(expected)
            assert result.get('passed', stage == 'training')
            continue
        failure = ROOT / ('failure_' + stage + '.json')
        assert not failure.exists(), 'Failed stages require review; no automatic retry'
        if need is None:
            peak = read(ROOT / 'acceptance/r38_flywire_fine.json')['peak_gpu_bytes'] / 2**20
            need = max(2400, peak*1.12+768)
        while True:
            resource = resources()
            choices = [(v['effective_free'], k) for k, v in resource['gpus'].items() if v['effective_free'] >= need+512]
            if choices and resource['ram_available'] > 10*2**30 and resource['disk_free'] > 16*2**30:
                _, device = max(choices)
                break
            save(ROOT / 'queue_status.json', dict(phase='waiting_' + stage + '_resources', task='flywire_fine',
                 required_mib=need+512, resources=resource, pid=os.getpid(), time=time.time()))
            time.sleep(20)
        (ROOT / 'logs').mkdir(exist_ok=True)
        with (ROOT / 'logs' / (stage + '.log')).open('ab', buffering=0) as log:
            process = subprocess.Popen([str(PYTHON), '-u', *args], cwd=ROOT,
                env=env(device, min(.90, need/resource['gpus'][device]['total'])),
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        receipt = dict(stage=stage, pid=process.pid, gpu=device, reservation_mib=need,
                       protocol_sha256=sha(ROOT / 'protocol.json'), time=time.time())
        save(ROOT / 'launches' / (stage + '.json'), receipt)
        while process.poll() is None:
            save(ROOT / 'queue_status.json', dict(phase=stage, active=receipt, pid=os.getpid(), time=time.time()))
            time.sleep(5)
        if process.returncode or not expected.exists():
            report = dict(phase='failed_' + stage, returncode=process.returncode,
                          expected=str(expected), exists=expected.exists(), pid=os.getpid())
            save(failure, report)
            save(ROOT / 'queue_status.json', report)
            return
    save(ROOT / 'queue_status.json', dict(phase='complete', completed=['flywire_fine'], pid=os.getpid()))


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        save(ROOT / 'queue_failure.json', dict(error=repr(error), traceback=traceback.format_exc()))
        raise
