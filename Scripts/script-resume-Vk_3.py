import numpy as np
import FMC as fmc
import MyData as md
from time import time

filename = 'datasets/strategeion-resume-skills/resumes_pilot.csv'
n = 1000; m=100 # default values
iterations=10

print('n,m,k,budget,len(output)_avg,len(covered)_avg,epsilon_avg,time_avg,len(output)_std,len(covered)_std,epsilon_std,time_std')
md.readCSVfile(filename,SkipEnd=4,colorIndex=1,nrows=n,ncols=m) # colorIndex = 1 is Female in this dataset
for k in [5,10,15,20,25]:
    sel=np.zeros(iterations); covered =np.zeros(iterations); eps = np.zeros(iterations);t = np.zeros(iterations)
    for iter in range(iterations):
        t0 = time()
        tmp,eps[iter],covered[iter] = fmc.FairMC_RepRand(md.n, md.m, md.H, md.color, k,md.S,budget=n)
        sel[iter] = len(tmp)
        t[iter] = time()-t0
    print(n,m,k,n,np.average(sel),np.average(covered),np.average(eps),np.average(t),np.std(sel),np.std(covered),np.std(eps),np.std(t))
