import numpy as np


def student_id():
    return 123456789, r'reference@mail.tau.ac.il'


# Part I
class QuadraticFunction:
    def __init__(self, Q, b):
        pass

    def __call__(self, x):
        pass

    def grad(self, x):
        pass

    def hessian(self, x):
        pass


class NewtonOptimizer:
    def __init__(self, func, alpha, init):
        pass

    def step(self):
        pass

    def optimize(self, threshold, max_iters):
        pass


class ConjGradOptimizer:
    def __init__(self, func, init):
        pass

    def step(self):
        pass

    def optimize(self):
        pass

    def update_grad(self):
        pass

    def update_dir(self, prev_grad):
        pass

    def update_alpha(self):
        pass

    def update_x(self):
        pass


# Part II
class FastRoute:
    def __init__(self, start_x, start_y, finish_x, finish_y, velocities):
        pass

    def __call__(self, x):
        pass

    def grad(self, x):
        pass

    def hessian(self, x):
        pass


def find_fast_route(objective, init, alpha=1, threshold=1e-3, max_iters=1e3):
    pass

def find_alpha(start_x, start_y, finish_x, finish_y, num_layers):
    # Bonus
    pass