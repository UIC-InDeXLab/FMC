import numpy as np
import FMC as fmc
import MyData as md
from time import time

filename = 'datasets/strategeion-resume-skills/resumes_pilot.csv'
n = 1000; k=10 # default values

print('n,m,k,len(output),len(covered),b0,b1,epsilon,time')
for iter in range(30):
    for m in [50,100,150,219]:
        md.readCSVfile(filename,SkipEnd=4,colorIndex=1,nrows=n,ncols=m) # colorIndex = 1 is Female in this dataset 
        t0 = time()
        output = fmc.FairMC_Randomized(md.n, md.m, md.H, md.color, k)
        t = time()-t0
        covered =set(); b1=0; b0=0
        for j in output:
            for i in md.S[j]: covered.add(i)
        for i in covered:
            if md.color[i]==0: b0+=1
            else: b1+=1
        epsilon = 1.*b0/b1 if b0>b1 else 1.*b1/b0
        print (n,m,k,len(output),len(covered),b0,b1,epsilon,t)