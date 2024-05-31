import subprocess
import numpy as onp

splitting = onp.arange(0, 4956005, 10000)

for i in range(len(splitting)-1):
    subprocess.Popen("python PrecomputeEdgesAngles.py "+str(splitting[i]) + " " + str(splitting[i+1]), shell=True)
# subprocess.Popen("")