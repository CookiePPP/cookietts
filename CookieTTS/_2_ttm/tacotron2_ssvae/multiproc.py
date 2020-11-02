import time
import torch
import sys
import os
import subprocess

argslist = list(sys.argv)[1:]
num_gpus = torch.cuda.device_count()
argslist.append(f'--n_gpus={num_gpus}')
workers = []
job_id = time.strftime("%Y_%m_%d-%H%M%S")
argslist.append(f"--group_name=group_{job_id}")
os.makedirs('logs', exist_ok=True)

for i in range(num_gpus):
    argslist.append('--rank={}'.format(i))
    stdout = None if i == 0 else open(f"logs/{job_id}_GPU_{i}.log", "w")
    print(argslist)
    p = subprocess.Popen([str(sys.executable)]+argslist, stdout=stdout)
    workers.append(p)
    argslist = argslist[:-1]

for p in workers:
    p.wait()