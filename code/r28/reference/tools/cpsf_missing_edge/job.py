import argparse,subprocess,sys,json
from .common import *
def main():
    p=argparse.ArgumentParser();p.add_argument('--task',required=True);p.add_argument('--arm',required=True);a=p.parse_args()
    assert json.loads((ROOT/'contracts'/(a.task+'.json')).read_text())['passed']
    args=[sys.executable,'-m','tools.cpsf_missing_edge.train','--task',a.task,'--arm',a.arm]
    subprocess.run(args+['--smoke'],check=True,cwd=ROOT)
    subprocess.run(args,check=True,cwd=ROOT)
if __name__=='__main__':main()
