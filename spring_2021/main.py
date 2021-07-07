from typing import Callable, Tuple

import numpy as np


def student_id():
    """
    Returns a tuple of student's ID and TAU email address.
    """

    return '123456789', r'template@mail.tau.ac.il'


class QuadraticFunction:
    """
    Implementation of a quadratice function f(x) = 1/2 * x^T*Q*x + b^T*x
    with its gradient and Hessian matrix.
    """

    def __init__(
        self, 
        Q: np.ndarray, 
        b: np.ndarray
    ) -> None:
        """
        Initializes a quadratic function object with Q, b

        Arguments:
            Q : (n,n) matrix

            b : (n,) vector
        """

        self.Q = Q
        self.b = b

    def __call__(
        self,
        x: np.ndarray
    ) -> np.ndarray:
        """
        Evaluates f(x)
        
        Arguments:
            x : (n,) vector

        Return:
            fx : scalar
        """

        qx = np.dot(self.Q, x)
        fx = 0.5 * np.dot(x, qx) + np.dot(self.b,x)

        return fx

    def grad(
        self,
        x: np.ndarray
    ) -> np.ndarray:
        """
        Evaluates g(x), the gradient of f, at x
        
        Arguments:
            x : (n,) vector

        Return:
            gx : (n,) vector
        """ 

        gx = 0.5 * np.dot(self.Q.T + self.Q, x) + self.b

        return gx

    def hessian(
        self,
        x: np.ndarray
    ) -> np.ndarray:
        """
        Evaluates h(x), the Hessian matrix of f, at x
        
        Arguments:
            x : (n,) vector
        
        Return:
            hx : (n,n) matrix
        """ 

        hx = 0.5 * (self.Q.T + self.Q)

        return hx


class NewtonOptimizer:
    """
    Implements Newton's method optimizer with constant alpha.
    """

    def __init__(
        self,
        objective: Callable,
        x_0: np.ndarray,
        alpha: float,
        threshold: float = 1e-10,
        max_iters: int = 100
    ) -> None:
        """
        Arguments:
            
            objective : callable to an objective function
            
            x_0 : initial point, (n,) numpy array

            alpha : a scalar step size

            threshold : (float) stopping criteria |x_k-▁x_(k-1)|<threshold

            max_iters : (int) maximal number of iterations (stopping criteria)            

        """

        self.objective = objective
        self.curr_x = x_0
        self.alpha = alpha
        self.threshold = threshold
        self.max_iters = max_iters
        
    def step(self):
        """
        Executes a single step of Newton's method.

        Return:

            next_x : an N-dimensional numpy vector, next x

            gx : vector, the gradient of f(x) evaluated at the current x 
                 (the one used for the step calculation)

            hx : matrix, the Hessian of f(x) evaluated current x 
                 (the one used for the step  calculation)
        """

        gx = self.objective.grad(self.curr_x)
        hx = self.objective.hessian(self.curr_x)
        next_x = self.curr_x - self.alpha * np.dot(np.linalg.inv(hx), gx)

        return next_x, gx, hx

    def optimize(self):
        """
        Execution of optimization flow

        Return:
            fmin : objective function evaluated at x_opt

            minimizer : x_opt

            num_iters : number of iterations until convergence
        """

        for iter_idx in range(self.max_iters):
            prev_x = self.curr_x
            self.curr_x, _, _ = self.step()  # update self.curr_x internally
            if np.linalg.norm(self.curr_x - prev_x) < self.threshold:
                break

        return self.objective(self.curr_x), self.curr_x, iter_idx+1


class BFGSOptimizer:
    """
    Implements the Quasi-Newton algorithm, using BFGS algorithm 
    for approximating the inverse Hessian.
    """

    def __init__(
        self, 
        objective: Callable,
        x_0: np.ndarray,
        B_0: np.ndarray,
        alpha_0: float,
        beta: float = 0.2,
        sigma: float = 0.3,
        threshold: float = 1e-10,
        max_iters: int = 100
    ) -> None:
        """
        Arguments:
            objective : callable to an objective function

            x_0 : initial point, (n,) numpy array

            B_0 : initial guess of the inverse Hessian

            alpha_0 : initial step size of Armijo line search (scalar)

            beta : (float) beta parameter of Armijo line search, a float in range (0,1)

            sigma : (float) sigma parameter of Armijo line search, a float in range (0,1)

            threshold : (float) stopping criteria |x_k-▁x_(k-1)|<threshold

            max_iters : (int) maximal number of iterations (stopping criteria)

        """

        self.objective = objective
        self.init_step_size = alpha_0
        self.beta = beta
        self.sigma = sigma
        self.threshold = threshold
        self.max_iters = max_iters

        self.curr_x = x_0
        self.curr_inv_hessian = B_0
        self.curr_step = 0
        self.curr_dir = np.zeros_like(x_0)

    def update_dir(self) -> np.ndarray:
        """
        Compute step direction

        Return:
            next_d : updated direction, (n,) numpy array
        """

        grad = self.objective.grad(self.curr_x)

        return -np.dot(self.curr_inv_hessian, grad)

    def update_step_size(self) -> np.ndarray:
        """
        Compute the new step size using Backtracking Line Search algorithm (Armijo rule).

        Return:
            step_size : scalar
        """ 

        # Initialize
        curr_step = self.init_step_size
        objective_1d = lambda alpha: self.objective(self.curr_x + alpha*self.curr_dir)  # restrict f to a line
        h = lambda alpha: objective_1d(alpha) - objective_1d(0)
        g = np.dot(self.objective.grad(self.curr_x), self.curr_dir)  # Note: d_phi(alpha)/dalpha = g(x+alpha*d)^T*d, alpha=0

        if h(curr_step) > self.sigma * g * curr_step:
            while h(curr_step) > self.sigma * g * curr_step:
                prev_step = curr_step
                curr_step = self.beta * curr_step
            step_size = curr_step
        else:
            while h(curr_step) <= self.sigma * g * curr_step:
                prev_step = curr_step
                curr_step = curr_step / self.beta
            step_size = prev_step

        return step_size

    def update_x(self) -> np.ndarray:
        """
        Take a step of size curr_step in direction dk.

        Return:
            next_x : updated point, (n,) numpy array
        """
        
        return self.curr_x + self.curr_step * self.curr_dir

    def update_inv_hessian(
        self,
        prev_x: np.ndarray
    ) -> np.ndarray:
        """
        Take a rank 2 BFGS update step to update the inverse Hessian.

        Arguments:
            prev_x : previous point, (n,) numpy array

        Return:
            next_inv_hessian : updated inverse Hessian, (n,n) numpy array
        """

        p = self.curr_x - prev_x
        q = self.objective.grad(self.curr_x) - self.objective.grad(prev_x)
        s = np.dot(self.curr_inv_hessian, q)
        tau = np.dot(s, q)
        mu = np.dot(p, q)
        v = p / mu - s / tau

        next_inv_hessian = self.curr_inv_hessian + np.outer(p, p) / mu - np.outer(s, s) / tau + tau * np.outer(v, v)

        return next_inv_hessian

    def step(self):
        """
        Executes a single step of Quasi-Newton algorithm.
        
        Return:
            next_x : previous point, (n,) numpy array

            next_d : vector, updated direction
            
            next_step_size : scalar
            
            next_inv_hessian : (n,n) numpy array, approximator of the inverse Hessian matrix
        """

        self.curr_dir = self.update_dir()
        self.curr_step = self.update_step_size()
        prev_x = self.curr_x
        self.curr_x = self.update_x()
        self.curr_inv_hessian = self.update_inv_hessian(prev_x)

        return self.curr_x, self.curr_dir, self.curr_step, self.curr_inv_hessian

    def optimize(self) -> Tuple:
        """
        Execution of optimization flow
        
        Return:
            fmin : objective function evaluated at x_opt

            minimizer : x_opt

            num_iters : number of iterations until convergence
        """

        for iter_idx in range(self.max_iters):
            prev_x = self.curr_x
            _ = self.step()  # update self.curr_x internally
            if np.linalg.norm(self.curr_x - prev_x) < self.threshold:
                break

        return self.objective(self.curr_x), self.curr_x, iter_idx+1


class TotalVariationObjective:
    """
    Implementation of a Total Variation objective function (MSE + TV) for image denoising.
    """

    def __init__(
        self,
        src_img: np.ndarray,
        mu: float = 1e-3,
        eps: float = 1e-8
    ) -> None:
        """
        Initializes a FastRoute problem

        Arguments:
            src_img : (n,m) matrix, input noisy image

            mu : regularization parameter, determines the weight of total variation term

            eps : small number for numerical stability
        """

        self.src_img = src_img
        self.mu = mu
        self.eps = eps

    def __call__(
        self,
        img: np.ndarray
    ) -> np.ndarray:
        """
        Compute TV objective.
        
        Arguments:
            img : (nxm,) vector, denoised image
        
        Return:
            total_variation : scalar, objective's value
        """

        # Convert vector to matrix
        img = self._vec_to_mat(img)

        mse = np.square(self.src_img - img).mean()
        
        row_diff, col_diff = self._diff(img)
        row_diff_pad = np.pad(row_diff, ((0,1), (0,0)), 'constant', constant_values=0)
        col_diff_pad = np.pad(col_diff, ((0,0), (0,1)), 'constant', constant_values=0)
        total_variation = np.sqrt(row_diff_pad**2  + col_diff_pad**2 + self.eps).sum()

        return mse + self.mu * total_variation

    def grad(
        self,
        img: np.ndarray
    ) -> np.ndarray:
        """
        Evaluate the gradient of the objective.
        
        Arguments:
            img : (nxm,) row-major vector, denoised image
        
        Return:
            grad : (nxm,) row-major vector, the objective's gradient
        """

        # Convert vector to matrix
        img = self._vec_to_mat(img)

        # MSE term
        mse_grad = -2*(self.src_img - img) / img.size

        # TV terms
        eps = self.eps
        row_diff, col_diff = self._diff(img, pad=True)
        term_1 = row_diff[:-1,1:-1] / np.sqrt(row_diff[:-1,1:-1]**2 + col_diff[:-2,1:]**2 + eps)
        term_2 = (row_diff[1:,1:-1] + col_diff[1:-1,1:]) / np.sqrt(row_diff[1:,1:-1]**2 + col_diff[1:-1,1:]**2 + eps)
        term_3 = col_diff[1:-1,:-1] / np.sqrt(row_diff[1:,:-2]**2 + col_diff[1:-1,:-1]**2 + eps)

        grad = mse_grad + self.mu * (term_1 - term_2 + term_3)

        return self._mat_to_vec(grad)

    def _mat_to_vec(
        self,
        img: np.ndarray
    ) -> np.ndarray:
        """
        Converts a vector to an (n,m) matrix.
        """

        return img.flatten()

    def _vec_to_mat(
        self,
        img: np.ndarray
    ) -> np.ndarray:
        """
        Converts an image to a row major (nxm,) vector
        """

        return img.reshape(self.src_img.shape)

    def _diff(
        self,
        x: np.ndarray,
        pad: bool = False
    ) -> Tuple:
        """
        Calculate the row and collumn difference arrays

        Arguments:
            x : (n,m) matrix

            pad : False, zero padding on all boundaries

        Return:
            row_diff : row_diff[i,:] = x[i+1,:] - x[i,:], 
                        output dimensions:
                            pad is False: (n-1,m)
                            pad is True: (n+1,m+2)

            col_diff : col_diff[:,i] = x[:,i+1] - x[:,i]
                        output dimensions:
                            pad is False: (n,m-1)
                            pad is True: (n+2,m+1)
        """

        row_diff = np.diff(x, axis=0)
        col_diff = np.diff(x, axis=1)
    
        if pad:
            row_diff = np.pad(row_diff, ((1,1),(1,1)), 'constant', constant_values=0)
            col_diff = np.pad(col_diff, ((1,1),(1,1)), 'constant', constant_values=0)

        return row_diff, col_diff


def denoise_img(
    noisy_img: np.ndarray,
    B_0: np.ndarray,
    alpha_0: float,
    beta: float = 0.2,
    sigma: float = 0.3,
    threshold: float = 1e-10,
    max_iters: int = 1000,
    mu: float = 1e-4,
    eps: float = 1e-8
) -> Tuple:
    """
    Optimizes a Total Variantion objective using BFGS optimizer to 
    denoise a noisy image.
    
    Arguments:
    	noisy_img : (n,m) matrix, input noisy image

        For the rest: see BFGSOptimizer and TotalVariationObjective docstrings.
    
    Return:
        total_variation : loss at minimum

        img : matrix, denoised image

        num_iters : number of iterations until convergence

    """

    # Initialize objective and optimizer
    denoising_objective = TotalVariationObjective(src_img=noisy_img, mu=mu, eps=eps)

    opt = BFGSOptimizer(
        objective=denoising_objective,
        x_0=noisy_img.flatten(),
        B_0=B_0,
        alpha_0=alpha_0,
        beta=beta,
        sigma=sigma,
        threshold=threshold,
        max_iters=max_iters
    )

    # Optimize
    total_variation, denoised_img, num_iters = opt.optimize()

    # Post process
    denoised_img = np.clip(denoised_img, a_min=0, a_max=1)
    denoised_img = denoised_img.reshape(32,32)

    return total_variation, denoised_img, num_iters