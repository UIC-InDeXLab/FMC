import numpy as np
import MC as mc
import MyData as md
from time import time

filename = 'datasets/strategeion-resume-skills/resumes_pilot.csv'
n = 1000; m=100; k=10 # default values

print('n,m,k,len(covered),b0,b1,epsilon,time')
for n in [200,500,1000,1500,1986]:
    md.readCSVfile(filename,SkipEnd=4,colorIndex=1,nrows=n,ncols=m) # colorIndex = 1 is Female in this dataset
    t0 = time()
    output,covered = mc.max_cover(md.S,k)
    t = time()-t0
    b1=0; b0=0
    for i in covered:
        if md.color[i]==0: b0+=1
        else: b1+=1
    epsilon = 1.*b0/b1 if b0>b1 else 1.*b1/b0
    print (n,m,k,len(covered),b0,b1,epsilon,t)

n=1000
for m in [50,100,150,216]:
    md.readCSVfile(filename,SkipEnd=4,colorIndex=1,nrows=n,ncols=m) # colorIndex = 1 is Female in this dataset
    t0 = time()
    output,covered = mc.max_cover(md.S,k)
    t = time()-t0
    b1=0; b0=0
    for i in covered:
        if md.color[i]==0: b0+=1
        else: b1+=1
    epsilon = 1.*b0/b1 if b0>b1 else 1.*b1/b0
    print (n,m,k,len(covered),b0,b1,epsilon,t)
m=100
md.readCSVfile(filename,SkipEnd=4,colorIndex=1,nrows=n,ncols=m) # colorIndex = 1 is Female in this dataset
for k in [5,10,15,20,25]:
    t0 = time()
    output,covered = mc.max_cover(md.S,k)
    t = time()-t0
    b1=0; b0=0
    for i in covered:
        if md.color[i]==0: b0+=1
        else: b1+=1
    epsilon = 1.*b0/b1 if b0>b1 else 1.*b1/b0
    print (n,m,k,len(covered),b0,b1,epsilon,t)