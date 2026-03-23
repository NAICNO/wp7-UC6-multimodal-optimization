#!/usr/bin/env python

# std libs
from collections import defaultdict

###############################################################################
# funtions
###############################################################################
def remove_identical(individuals):

    seen = {}
    unique = []
    for ind in individuals:
        if ind[0] not in seen:
            seen[ind[0]] = True
            unique.append(ind)

    return(unique)

def dublicates(individuals):
    seen = {}
    dub = []
    for ind in individuals:
        if ind[0] not in seen:
            seen[ind[0]] = True
        else:
            dub.append(ind)

    return(dub)

def assert_distinct(individuals):
    seen = {}
    for ind in individuals:
        if ind[0] in seen:
            assert(0)
        else:
            seen[ind[0]] = True

def clearing_1d_2f(individuals, delta_0=-1, delta_1=-1):
    assert(delta_0>=0)
    assert(delta_1>=0)
    assert(delta_0+delta_1>0)

    f0_max = max([ind.fitness.values[0] for ind in individuals])
    f1_max = max([ind.fitness.values[1] for ind in individuals])

    individuals.sort(key=lambda ind: ind.fitness.values[0])

    idx_min = 0
    for k in range(1,len(individuals)):
        if k==idx_min:
            next
        d0 = individuals[k].fitness.values[0] - individuals[idx_min].fitness.values[0]
        d1 = abs(individuals[k].fitness.values[1] - individuals[idx_min].fitness.values[1])
        if d0<delta_0 and d1<delta_1:
            individuals[k].fitness.values = (2*f0_max, 2*f1_max)
        else:
            idx_min = k+1



