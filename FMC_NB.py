# Abolfazl Asudeh Jan. 2020
# a.asudeh@gmail.com
import numpy as np
import math
from MyLPSolver import solve #,solveBinaryIP

def FairMC_Randomized(n, m, H, C, k):
    # n: number of items; I assume the item indices are from 0 ... (n-1)
    # m: number of sets
    # H: the collection of sets each item hit
    # C: 1D array of colors
    # k: the parameter for max coverage by k sets
    [x,y] = LPRelaxed(n,m,H,C,k) # the values of x and y
    output = list()
    rnd = np.random.random(m)
    for i in range(m):
        if rnd[i]<=y[i]: output.append(i)
    return output

def FairMC_RepRand(n, m, H, C, k, S, budget=0):
    # n: number of items; I assume the item indices are from 0 ... (n-1)
    # m: number of sets
    # H: the collection of sets each item hit
    # C: 1D array of colors
    # k: the parameter for max coverage by k sets
    [x,y] = LPRelaxed(n,m,H,C,k) # the values of x and y
    output = list(); cSize = 0; minEps = 10000 # a really large number
    if budget<=0: budget = n
    for iter in range(budget):
        tmp = list()
        rnd = np.random.random(m)
        for i in range(m):
            if rnd[i]<=y[i]:
                tmp.append(i)
        covered =set(); b1=0; b0=0
        for j in tmp:
            for i in S[j]: covered.add(i)
        #epsilon = disparity(covered, C) 
        for i in covered:
            if C[i]==0: b0+=1
            else: b1+=1
        if b0==0 or b1==0: continue
        epsilon = 1.*b0/b1 if b0>b1 else 1.*b1/b0
        if epsilon<minEps:
            output = tmp[:]
            cSize = len(covered)
            minEps = epsilon
    return output,minEps-1,cSize


'''
def FairMC_IP(n, m, H, C, k):
    # n: number of items; I assume the item indices are from 0 ... (n-1)
    # m: number of sets
    # H: the collection of sets each item hit
    # C: 1D array of colors
    # k: the parameter for max coverage by k sets
    x = IP(n,m,H,C,k) # the values of x and y
    print(x)
'''


# ------------------- Private functions ----------------
def LPRelaxed(n,m, H, C, k):
    # n: number of items; I assume the item indices are from 0 ... (n-1)
    # m: number of sets
    # H: the collection of sets each item hit
    # C: 1D array of colors
    # k: the parameter for max coverage by k sets

    # ----- construct the LP ----
    # x_1 to x_n: items; x_{n+1} to x_{m+n} sets:
    # objective: max sum x_i
    objective = np.zeros(n+m)
    for i in range(n): objective[i] = -1 # since the solver minimizes the obj
    # adding constraints
    base = [0 for i in range(n+m)]
    # 1: sum y_j \leq k
    tmp = base[:]
    for i in range(m): tmp[i+n] = 1
    left = np.array([tmp])
    right = np.array(k)
    # 2: forall x_i: x_i - sum H[i] \leq 0
    for i in range(n):
        tmp = base[:]
        tmp[i] = 1 # x_i = 1
        for j in H[i]: 
            tmp[n+j] = -1 # - sum H[i]
        left = np.append(left,[tmp],axis=0)
        right = np.append(right,0) # leq 0
    # 3: color should be equal. Note: this is for binary color
    tmp = base[:]
    for i in range(n):
        tmp[i] = 1 if C[i]==1 else -1
    left_eq = np.array([tmp])
    right_eq = np.array(0)
    # 4: x_i\in[0,1], y_i\in[0,1]
    bnds = [(0,1) for i in range(n+m)]
    [status,msg,result] = solve(objective,left,right,left_eq,right_eq, bnds)
    if status == False: print (msg)
    return [result[:n],result[n:]]

def disparity(Items, C):
    chi  = max(C)+1
    counts = np.zeros(chi)
    for i in Items:
        counts[C[i]]+=1
    max_disp=0
    for i in range(chi-1):
        for j in range(i+1,chi):
            epsilon = 1.*counts[i]/counts[j] if counts[i]>counts[j] else 1.*counts[j]/counts[i]
            if epsilon>max_disp: max_disp = epsilon
    return max_disp

'''
def IP(n,m, H, C, k):
    # n: number of items; I assume the item indices are from 0 ... (n-1)
    # m: number of sets
    # H: the collection of sets each item hit
    # C: 1D array of colors
    # k: the parameter for max coverage by k sets

    # ----- construct the LP ----
    # x_1 to x_n: items; x_{n+1} to x_{m+n} sets:
    # objective: max sum x_i
    objective = np.zeros(n+m)
    for i in range(n): objective[i] = -1 # since the solver minimizes the obj
    # adding constraints
    base = [0 for i in range(n+m)]
    # 1: sum y_j \leq k
    tmp = base[:]
    for i in range(m): tmp[i+n] = 1
    left = np.array([tmp])
    right = np.array(k)
    # 2: forall x_i: x_i - sum H[i] \leq 0
    for i in range(n):
        tmp = base[:]
        tmp[i] = 1 # x_i = 1
        for j in H[i]: 
            tmp[n+j] = -1 # - sum H[i]
        left = np.append(left,[tmp],axis=0)
        right = np.append(right,0) # leq 0
    # 3: color should be equal. Note: this is for binary color
    tmp = base[:]
    for i in range(n):
        tmp[i] = 1 if C[i]==1 else -1
    left_eq = np.array([tmp])
    right_eq = np.array(0)
    # 4: x_i\in[0,1], y_i\in[0,1]
    bnds = [(0,1) for i in range(n+m)]
    return solveBinaryIP(objective,left,right,left_eq,right_eq)
'''

'''
# unit test for LPRelaxed
n = 10 # number of items
S = [[0,2,3,8],[1,3,5,6,7],[2,5,6,9],[0,1,2,3,8],[1,3,4,5,7]] # m = 5
H = [[0,3],[1,3,4],[0,2,3],[0,1,3,4],[4],[1,2,4],[1,2],[1,4],[0,3],[2]]
k = 2 # number of sets to select
C = [0,0,1,1,0,1,0,1,0,1]
[Items,Sets] = fmc.LPRelaxed(n,5,H,C,k)
print Items
print Sets
# Output:
# [ 1.  1.  1.  1.  0.  1.  1.  0.  1.  1.]
# [ 0.  0.  1.  1.  0.]
'''
