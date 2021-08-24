import numpy as np
import FMC_NB as fmc
import MC as mc
import MyData as md

''' 
n = 10 # number of items
m=5
S = [[0,2,3,8],[1,3,5,6,7],[2,5,6,9],[0,1,2,3,8],[1,3,4,5,7]] # m = 5
H = [[0,3],[1,3,4],[0,2,3],[0,1,3,4],[4],[1,2,4],[1,2],[1,4],[0,3],[2]]
k = 2 # number of sets to select
C = [0,0,1,1,0,1,0,1,0,1]
print fmc.FairMC_RepRand(n, m, H, C, k, S)
#print(S)
#print eps,covered

#--------------- Max Cover -----------
n = 10 # number of items
S = [set([0,2,3,8]),set([1,3,5,6,7]),set([2,5,6,9]),set([0,1,2,3,8]),set([1,3,4,5,7])]
k = 2 # number of sets to select
C = [0,0,1,1,0,1,0,1,0,1]
output,covered = mc.max_cover(S,k)
print output
print covered
b1=0; b0=0
for i in covered:
    if C[i]==0: b0+=1
    else: b1+=1
print output
print len(covered)
print b0,b1, b0*1./b1
'''

md.readChicago()
print(md.n, md.m)
k = 10 # number of sets to select
print(fmc.FairMC_RepRand(md.n, md.m, md.H, md.color, k,md.S))