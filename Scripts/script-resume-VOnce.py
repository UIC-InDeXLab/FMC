import numpy as np
import FMC as fmc
import MyData as md
from time import time

filename = 'datasets/strategeion-resume-skills/resumes_pilot.csv'
n = 1000; m=100; k=10 # default values

print('n,m,k,len(covered),b0,b1,epsilon,time')
for n in [200,500,1000,1500,1986]:
    md.readCSVfile(filename,SkipEnd=4,colorIndex=1,nrows=n,ncols=m) # colorIndex = 1 is Female in this dataset
    t0 = time()
    fmc.FairMC_RepRand(md.n, md.m, md.H, md.color, k,md.S,budget=n)
    t = time()-t0
    print (n,m,k,n,t)

n=1000
for m in [50,100,150,200]:
    md.readCSVfile(filename,SkipEnd=4,colorIndex=1,nrows=n,ncols=m) # colorIndex = 1 is Female in this dataset
    t0 = time()
    fmc.FairMC_RepRand(md.n, md.m, md.H, md.color, k,md.S,budget=n)
    t = time()-t0
    print (n,m,k,n,t)

m=100
md.readCSVfile(filename,SkipEnd=4,colorIndex=1,nrows=n,ncols=m) # colorIndex = 1 is Female in this dataset
for k in [5,10,15,20,25]:
    t0 = time()
    fmc.FairMC_RepRand(md.n, md.m, md.H, md.color, k,md.S,budget=n)
    t = time()-t0
    print (n,m,k,n,t)