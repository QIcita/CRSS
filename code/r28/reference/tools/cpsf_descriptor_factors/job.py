import argparse,subprocess,sys,json
from .common import *

def main():
    p=argparse.ArgumentParser();p.add_argument('--task',required=True);p.add_argument('--arm',required=True);p.add_argument('--descriptor',required=True);a=p.parse_args()
    c=json.loads((ROOT/'contracts/tlp.json').read_text())
    assert c['passed'] and c['protocol_sha256']==sha(ROOT/'protocol.json')
    args=[sys.executable,'-m','tools.cpsf_descriptor_factors.train','--task',a.task,'--arm',a.arm,'--descriptor',a.descriptor]
    subprocess.run(args+['--smoke'],check=True,cwd=ROOT)
    subprocess.run(args,check=True,cwd=ROOT)

if __name__=='__main__':main()
