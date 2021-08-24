# Adopted from http://www.martinbroadhurst.com/greedy-set-cover-in-python.html
# Abolfazl Asudeh Feb. 2020
# a.asudeh@gmail.com
import numpy as np
import math


def max_cover(subsets, k):
    covered = set()
    cover = []
    # Greedily add the subsets with the most uncovered points
    for i in range(k): # select k sets
        # subset = max(subsets, key=lambda s: len(s - covered))
        best=0
        for j in range(len(subsets)):
            tmp = len(subsets[j] - covered)
            if tmp>best:
                best = tmp
                subset = j
        cover.append(subset)
        covered |= subsets[subset]
    return cover,covered