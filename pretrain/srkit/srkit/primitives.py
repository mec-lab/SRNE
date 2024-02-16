##!/usr/bin/env python
# -*- coding: utf-8 -*-

# primitives.py
# Jim Bagrow
# Last Modified: 2023-10-27

"""Primitives are the building blocks of expressions. A primitive set or "pset"
is a collection (unordered set) of primitives used to build expressions. An
expression is an ordered sequence of primitives.

TODO: we probably need _plists_ as well as psets as a neural network will emit
primitive IDs (ints)...
"""

import numpy as np
import torch
from fractions import Fraction # wow


class Primitive():
    """A primitive, building block, or token of an SR expression.
    An SR expression is an ordered sequence or tree of primitives

    function: callable

    name : str

    arity : int

    ...
    """

    def __init__(self, function, name, arity):
        self.function = function
        self.name = name
        self.arity = arity

        # TODO: confirm exog-var/terminal has arity 0, no function.

    def __call__(self, *args):
        try:
            return self.function(*args)
        except TypeError as e:
            if not callable(self.function):
                print(f"Primitive {self.name} is not executable.")
                return None # or raise?
            else:
                raise e

    def __repr__(self):
        return self.name
        #return f"{self.name} (arity: {self.arity})"

    def is_binary(self):
        return self.arity == 2

    def is_unary(self):
        return self.arity == 1

    def is_exog(self):
        return self.arity == 0 and self.function is None

    def is_const(self):
        return self.arity == 0 and not self.function is not None

def protected_div(): # TODO
    pass


def protected_log():  # TODO
    pass


def protected_sqrt():  # TODO
    pass


def protected_pow():  # TODO
    pass


binary_pset = [
    Primitive(torch.add,      "add", arity=2),
    Primitive(torch.subtract, "sub", arity=2),
    Primitive(torch.mul, "mul", arity=2),
    Primitive(torch.div,   "div", arity=2), # WARN
    Primitive(torch.pow,    "pow", arity=2), # WARN
]


unary_pset = [
    Primitive(torch.sin,        "sin",  arity=1),
    Primitive(torch.cos,        "cos",  arity=1),
    Primitive(torch.tan,        "tan",  arity=1),
    Primitive(torch.exp,        "exp",  arity=1), # WARN
    Primitive(torch.log,        "log",  arity=1), # WARN
    Primitive(torch.sqrt,       "sqrt", arity=1), # WARN
    Primitive(torch.abs,        "abs",  arity=1),
    Primitive(torch.minimum,    "min",  arity=1),
    Primitive(torch.maximum,    "max",  arity=1),
    Primitive(torch.tanh,       "tanh", arity=1),
    Primitive(torch.reciprocal, "inv",  arity=1), # WARN
]

exog_pset = [
    Primitive(None, f"x{i}", arity=0) for i in range(1,3+1)
]

power_pset = [
    Primitive(lambda: 2, "2", arity=0),
    Primitive(lambda: 3, "3", arity=0),
    Primitive(lambda: 4, "4", arity=0),
]

primitives = binary_pset + unary_pset + exog_pset + power_pset
name2primitive = { f.name : f for f in primitives }
