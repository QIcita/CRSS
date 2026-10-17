import argparse,subprocess,sys
from .common import *
def main(task,arm):
    key=task+'_'+arm
    commands=[('contracts',['--task',task,'--arm',arm]),
        ('train',['--task',task,'--arm',arm,'--out',str(ROOT/'smoke'/key),'--smoke']),
        ('train',['--task',task,'--arm',arm,'--out',str(ROOT/'results'/key)])]
    for stage,args in commands:
        save(ROOT/'job_status'/(key+'.json'),dict(state='running',stage=stage,args=args))
        subprocess.run([sys.executable,'-m','tools.cpsf_v26_complete.'+stage,*args],cwd=ROOT,check=True)
    save(ROOT/'job_status'/(key+'.json'),dict(state='complete'))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--task',required=True);ap.add_argument('--arm',required=True);a=ap.parse_args()
    try:main(a.task,a.arm)
    except BaseException as e:save(ROOT/'job_status'/(a.task+'_'+a.arm+'.json'),dict(state='failed',error=repr(e)));raise
