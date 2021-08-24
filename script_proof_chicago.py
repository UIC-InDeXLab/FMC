import numpy as np
import FMC as fmc
import MC as mc
import MyData as md

k=5 # default values
md.readChicago()
output,eps,covered = fmc.FairMC_RepRand(md.n, md.m, md.H, md.color, k,md.S,budget=100)
print(output)
print(eps,covered)
output,covered = mc.max_cover(md.S,k)
print(output)
b1=0; b0=0
for i in covered:
    if md.color[i]==0: b0+=1
    else: b1+=1
print(b0,b1, b0*1./b1,len(covered))
