"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$

def mul(x: float, y: float) -> float:
    return x * y

def id(x: float) -> float:
    return x

def add(x: float, y: float) -> float:
    return x + y

def neg(x: float) -> float:
    return -x

def lt(x: float, y: float) -> bool:
    return x < y

def eq(x: float, y: float) -> bool:
    return x == y

def max(x: float, y: float) -> float:
    if lt(x, y):
        return y
    return x

def is_close(x: float, y: float) -> bool:
    if abs(x - y) < 1e-2:
        return True
    return False

def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        exp_x = math.exp(x)
        return exp_x / (1.0 + exp_x)

def relu(x: float) -> float:
    if x > 0:
        return x
    return 0.0

def log(x: float) -> float:
    return math.log(x)

def exp(x: float) -> float:
    return math.exp(x)

def log_back(x: float, d: float) -> float:
    return d / x

def inv(x: float) -> float:
    return 1.0 / x

def inv_back(x: float, d: float) -> float:
    return -d / (x**2)

def relu_back(x: float, d: float) -> float:
    if x > 0:
        return d
    return 0.0

# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists
def map(fn):
    def apply(ls):
        result = []
        for x in ls:
            result.append(fn(x))
        return result
    return apply


def zipWith(fn):
    def apply(ls1, ls2):
        result = []
        for x, y in zip(ls1, ls2):
            result.append(fn(x, y))
        return result
    return apply


def reduce(fn, start):
    def apply(ls):
        result = start
        for x in ls:
            result = fn(result, x)
        return result
    return apply


negList = map(neg)
addLists = zipWith(add)
sum = reduce(add, 0.0)
prod = reduce(mul, 1.0)
