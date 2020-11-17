import time
import torch
import sys
import os
import subprocess
import signal

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
    p = subprocess.Popen([str(sys.executable)]+argslist, stdout=None, stderr=subprocess.STDOUT)#, stdout=stdout)
    workers.append(p)
    argslist = argslist[:-1]

try:
    all_running = True
    while all_running:
        if any(p.poll() is not None for p in workers):# if any one of the graphics cards is not running...
            all_running = False#                        set all_running to False
        time.sleep(0.5)# and time between polls
    
    time.sleep(10.0)# grace period (in case the processes are ending due to finishing the training run)
    for p in workers:# if the some of the subprocesses are still alive, kill the remaining processes.
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        except Exception:
            pass
except KeyboardInterrupt:
    # if the user does a KeyboardInterrupt (Ctrl+C), forward the KeyboardInterrupt to every GPU.
    print("Got KeyboardInterrupt. Killing Subprocesses!\n")    
    if True:
        for p in workers:# kill the remaining processes if they don't stop via KeyboardInterrupt in the time_limit.
            if p.poll() is None:
                os.killpg(os.getpgid(p.pid), signal.SIGTERM)
    else:
        num_interrupt_attemps = 3
        for i in range(num_interrupt_attemps):
            for p in workers:
                if p.poll() is None:
                    try:
                        os.killpg(os.getpgid(p.pid), signal.SIGINT)
                    except Exception:
                        pass
            if not all(p.poll() is not None for p in workers):
                time.sleep(0.2)
        
        all_stopped = False
        start_time = time.time()
        time_limit = 5.0
        while (not all_stopped) and (time.time()-start_time) < time_limit:
            if all(p.poll() is not None for p in workers):
                all_stopped = True
            time.sleep(0.2)# and time between polls
        
        if not all_stopped:
            for p in workers:# kill the remaining processes if they don't stop via KeyboardInterrupt in the time_limit.
                if p.poll() is None:
                    try:
                        os.killpg(os.getpgid(p.pid), signal.SIGTERM)
                    except Exception:
                        pass
