import numpy as np
from scipy.optimize import linprog
# from mip import Model, xsum, maximize, BINARY # https://python-mip.readthedocs.io/en/latest/install.html
import math

bounds = None

def solve(c, A_ub=None,b_ub=None,A_eq=None, b_eq=None,bnds=None):
    global bounds
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds if bnds is None else bnds, options={"disp": False})
    return [res.success,res.message,res.x]

'''
def solveBinaryIP(c, A_ub=None,b_ub=None,A_eq=None, b_eq=None):
    n = len(c)
    m = Model()
    x = [m.add_var(var_type=BINARY) for i in range(n)]
    m.objective = maximize(xsum(c[i] * x[i] for i in range(n)))
    for j in range(len(b_ub)):
        m += xsum(A_ub[j,i] * x[i] for i in range(n)) <= b_ub[j]
    for j in range(len(b_eq)):
        m += xsum(A_eq[j,i] * x[i] for i in range(n)) == b_eq[j]
    m.optimize()
    if m.num_solutions:
        return x
    else:
        return None
'''