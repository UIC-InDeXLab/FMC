#Abolfazl Asudeh Jan 2020
#       http://asudeh.github.io

import numpy as np
import csv

Items = [] # the indices/names of the items
Sets = [] # the indices/names of sets
data = None # the dataset: every row is an item; sets are the columns
color = [] # color of items
n = 0 # No items
m = 0 # No sets
S = [] # the items each set covers
H = [] # the sets each item hits
q = [] # this is only used for proportional fairness

def readCSVfile(filename,delim=',', header=1, SkipBeg=1, SkipEnd=1, colorIndex=[0],nrows=-1,ncols=-1): # the last columns are the sensitive attributes
    # header: 1 if the dataset contains header
    # SkipBeg: number of (ID) columns to skip from the beginning 
    # SkipEnd: number of columns to skip at the beginning 
    # colorIndex: the index(es) of the sensitive attribute to use (if it is right after the last set the index is 0)
    # nrows: number of itmes: -1 = use all
    # ncols: number of sets: -1 = use all
    global Items, Sets, data, color, n, m, S, H
    data_raw = np.genfromtxt(filename, delimiter=delim,skip_header=header)
    (n,m) = data_raw.shape
    n = int(n); m = int(m)-(SkipEnd + SkipBeg)
    Sets = [i for i in range(m)]
    Items = [i for i in range(n)]
    data = data_raw[:, SkipBeg : m + SkipBeg]
    if (type(colorIndex) is int) or len(colorIndex)==1: 
        color = list(data_raw[:,m + SkipBeg + colorIndex].astype(int))
    else: 
        color = _getColors(data_raw[:,[m+SkipBeg+i for i in colorIndex]].astype(int))
    if nrows>-1:
        data = data[:nrows,:]
        color = color[:nrows]
        n = nrows
    if ncols>-1:
        data = data[:,:ncols]
        m = ncols
    H = [[] for i in range(n)]
    S = [set([]) for i in range(m)]
    for i in range(n): 
        for j in range(m):
            if data[i,j] == 1: 
                H[i].append(j)
                S[j].add(i)

def readChicago(nrow=3000,nset=56): # the last columns are the sensitive attributes
    global Items, Sets, data, color, n, m, S, H,q
    reader = csv.reader(open("ChicagoZipFMC.csv","r",newline=''), delimiter=',',quoting=csv.QUOTE_ALL)
    _items = []
    _cnt = 0
    for row in reader: 
        _items.append(row)
        _cnt+=1
        if _cnt==nrow: break
    m = nset; n = len(_items)
    Sets = [i for i in range(m)]
    Items = [i for i in range(n)]
    H = [[] for i in range(n)]
    S = [set([]) for i in range(m)]
    for i in range(n):
        for j in range(len(_items[i])-1): # the last index is color
            k = int(_items[i][j])
            if k>=m: continue #skip the sets that are not in this setting
            H[i].append(k)
            S[k].add(i)
        color.append(int(_items[i][len(_items[i])-1]))
    q=[.58,.42]

def _getColors(A):
    A = A.astype(int)
    maxvals = np.amax(A, axis=0)
    n,m = A.shape
    card = 1
    for i in maxvals: card=card*(i+1)
    colorIndex = {}
    temp = [0 for i in range(m)]
    for i in range(int(card)):
        colorIndex[tuple(temp)]=i
        temp = _next(temp,maxvals)
    c = [colorIndex[tuple(A[i,:])] for i in range(n)]
    return c

def _next(current,maxvals):
    next = current[:]
    for i in range(len(maxvals)):
        if next[i]+1<= maxvals[i]:
            next[i]= next[i] + 1
            break
        else: next[i]=0
    return next