#!/usr/bin/env python
# -*- coding: utf-8 -*-

# expressions.py
# Jim Bagrow
# Last Modified: 2023-10-28

"""Expressions are ordered sequences of primitives."""

import sys, os
import numpy as np
from srkit.srkit.primitives import primitives, Primitive


class Expression():
    """An expression is an ordered sequence of primitives, corresponds to a
    pre-order traversal of the e-tree.
    """
    def __init__(self, expression):
        self.expression = expression

    def execute(self, x):
        """Execute the expression, given an input x. Expression should be in 
        prefix notation.
        """
        # TODO: can handle scalar and vector exog, but not multiple exogs

        stack = []
        for p in self.expression:
            stack.append([p])

            while len(stack[-1]) == stack[-1][0].arity + 1:
                cp = stack[-1][0]  # current prim
                tp = stack[-1][1:] # "terminal" prims

                if cp.is_exog():
                    result = x # TODO multivar
                else:
                    result = cp(*tp)

                if len(stack) != 1:
                    stack.pop()
                    stack[-1].append(result)
                else:
                    return result

        sys.exit("MAJOR FAIL")

    def execute_postfix(self,x):
        """"Postfix" execution of expression."""
        stack = []
        for p in self.expression:
            if p.arity == 0:  # terminal (exog or const)
                if p.is_exog():
                    stack.append(x)
                else:
                    stack.append( p() )
            else:  # operator
                # pop p's arity number of operands:
                operands = [stack.pop() for _ in range(p.arity)]
                result = p(*reversed(operands))  # reversed since popping right to left
                stack.append(result)
        return stack[0]


    def __call__(self, *args):
        return self.execute()


class ETree():
    """An expression tree
    """
    # JPB: do we need both Expressions and ETrees?q
    pass


# MEMOIZE?


if __name__ == '__main__':
    print(primitives)

    # assemble some test cases: #TODO move to test suite
    prim_sin = Primitive(np.sin, "sin", arity=1)
    prim_add = Primitive(np.add, "add", arity=2)
    prim_mul = Primitive(np.multiply, "mul", arity=2)
    prim_x1 = Primitive(None, 'x1', arity=0)
    prim_2_0 = Primitive(lambda :2.0, '2.0', arity=0)  # constant 2.0
    prim_1_0 = Primitive(lambda :1.0, '1.0', arity=0)  # constant 1.0

    test_expression = [prim_sin, prim_x1]
    expression = Expression(test_expression)
    assert expression.execute(2.0) == np.sin(2.0)

    test_expression = [prim_add, prim_x1, prim_x1]
    expression = Expression(test_expression)
    assert expression.execute(1.2) == 1.2 + 1.2
    assert expression.execute(-0.9) == -0.9 + -0.9

    test_expression = [prim_add, prim_1_0, prim_x1]
    expression = Expression(test_expression)
    assert expression.execute(1.2) == 1.2 + 1.0

    test_expression_postfix = [prim_2_0, prim_x1, prim_1_0, prim_add, prim_sin, prim_mul]
    expression = Expression(test_expression_postfix)
    assert expression.execute_postfix(3.14) == 2 * np.sin(3.14 + 1)
    print(expression.execute_postfix(np.array([2.0, 3.14, 3.14])))


    test_expression = [prim_mul, prim_2_0, prim_sin, prim_add, prim_x1, prim_1_0]
    expression = Expression(test_expression)
    y = expression.execute(3.14)
    yt = 2 * np.sin(3.14 + 1)
    assert y == yt

    test_expression = [prim_add, prim_sin, prim_x1, prim_1_0]
    expression = Expression(test_expression)
    assert expression.execute(2.0) == np.sin(2.0) + 1

    print(expression.execute(np.array([2.0, 2.0])))
