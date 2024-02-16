import numpy as np
import torch
import copy
import operator
import pickle
import os
import time
import random


def tournament_selection(pop):
    p1 = np.random.randint(len(pop))
    p2 = np.random.randint(len(pop))
    while p1 == p2:
        p2 = np.random.randint(len(pop))

    if pop[p1].sym_fitness > pop[p2].sym_fitness:
        return p1
    else:
        return p2


def survivor_selection(pop, popsize):
    # Tournament selection based on age and fitness

    # TODO: Check to see if the pareto front size is larger than the population size
    # and increase population size if it is.
    # This shouldn't happen if the population size is large enough but if it does
    # then it would make this function enter an infinite loop -- bad.

    # Remove dominated individuals until the target population size is reached
    while len(pop) > popsize:

        # Choose two different individuals from the population
        ind1 = np.random.randint(len(pop))
        ind2 = np.random.randint(len(pop))
        while ind1 == ind2:
            ind2 = np.random.randint(len(pop))

        if dominates(ind1, ind2, pop):  # ind1 dominates

            # remove ind2 from population and shift following individuals up in list
            for i in range(ind2, len(pop) - 1):
                pop[i] = pop[i + 1]
            pop.pop()  # remove last element from list (because it was shifted up)

        elif dominates(ind2, ind1, pop):  # ind2 dominates

            # remove ind1 from population and shift following individuals up in list
            for i in range(ind1, len(pop) - 1):
                pop[i] = pop[i + 1]
            pop.pop()  # remove last element from list (because it was shifted up)

    assert len(pop) == popsize
    return pop


def dominates(ind1, ind2, pop):
    # Returns true if ind1 dominates ind2, otherwise false
    if pop[ind1].sym_fitness == pop[ind2].sym_fitness and pop[ind1].num_fitness == pop[ind2].num_fitness:
        return pop[ind1].uniqueID > pop[ind2].uniqueID  # if equal, return the newer individual

    elif pop[ind1].sym_fitness <= pop[ind2].sym_fitness and pop[ind1].num_fitness <= pop[ind2].num_fitness:
        return True
    else:
        return False
