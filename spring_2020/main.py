import numpy as np


def student_id():
    """
    Returns a tuple of student's ID and TAU email address.
    """
    return 123456789, r'reference@mail.tau.ac.il'


class QuadraticFunction:

    """
    Implementation of a quadratice function 
    f(x) = 1/2 * x^T*Q*x + b^T*x
    with its gradient and Hessian matrix.
    """

    def __init__(self, Q, b):

        """
        Initializes a quadratic function.
        
        Arguments:
            Q : (N,N) numpy array

            b : (N,) numpy array
        """

        self.Q = Q
        self.b = b

    def __call__(self, x):

        """
        Evaluate f(x).
        
        Arguments:
            x : (N,) numpy array

        Return:
            fx : evaluation of f(x)
        """

        fx = 0.5 * np.dot(x, np.dot(self.Q, x)) + np.dot(self.b,x)

        return fx

    def grad(self, x):

        """
        Evaluates g(x), the gradient of f(x).

        Arguments:
            x : (N,) numpy array

        Output:
            gx : (N,) numpy array
                 evaluation of the function's gradient at x
        """ 

        gx = 0.5 * np.dot(self.Q.T + self.Q, x) + self.b

        return gx

    def hessian(self, x):

        """
        Evaluate H(x), the Hessian matrix of f(x).

        Arguments:
            x : (N,) numpy array

        Return:
            hx : (N,N) numpy array,
                 evaluation of the function's Hessian at point x
        """ 

        hx = 0.5 * (self.Q.T + self.Q)

        return hx


class NewtonOptimizer:

    """
    Newton's method optimizer with constant step size.
    """

    def __init__(self, func, alpha, init):

        """
        Arguments:
            func : objective function handle/reference

            alpha : a scalar step size
            
            init : (N,) numpy array, initial guess
        """

        self.func = func
        self.alpha = alpha
        self.curr_x = init

    def step(self):

        """
        Executes a single step of Newton's method.
        
        Return:
            next_x : (N,) numpy array, next x
        """

        gx = self.func.grad(self.curr_x)
        hx = self.func.hessian(self.curr_x)
        self.curr_x = self.curr_x - self.alpha * np.dot(np.linalg.inv(hx), gx)

        return self.curr_x, gx, hx

    def optimize(self, threshold, max_iters):
        
        """
        Execution of optimization flow

        Arguments:
            threshold : stopping criteria |x_k-▁x_(k-1) |<threshold

            max_iters : maximal number of iterations (stopping criteria)

        Return:
            fmin : objective function evaluated at the minimum

            minimizer : the optimal ▁x 

            num_iters : number of iterations
        """

        for iter_idx in range(max_iters):
            prev_x = self.curr_x
            _, _, _ = self.step()  # update self.curr_x internally
            if np.linalg.norm(self.curr_x - prev_x) < threshold:
                break

        return self.func(self.curr_x), self.curr_x, iter_idx+1


class ConjGradOptimizer:

    """
    Conjugate Gradients optimizer.
    """

    def __init__(self, func, init):

        """
        Arguments:

            func : quadratice function handle/reference

            init : (N,) numpy array, initial guess
        """

        self.func = func
        self.Q = self.func.Q
        self.curr_x = init
        self.curr_grad = func.grad(self.curr_x)
        self.curr_dir = -self.curr_grad
        self.curr_alpha = self.update_alpha()

    def optimize(self):

        """
        Executation of optimization flow
        
        Return:
            fmin : objective function evaluated at the minimum

            minimizer : the optimal ▁x 

            num_iters : number of iterations
        """

        num_iters = self.curr_x.shape[0]  # number of iteration is constant
        for iter_idx in range(num_iters):
            _, _, _, _ = self.step()

        return self.func(self.curr_x), self.curr_x, num_iters

    def step(self):

        """
        Executes a single step of Newton's method.

        Return:
            next_x : (N,) numpy array

            next_grad_x : (N,) numpy array

            next_dir : (N,) numpy array

            next_alpha : scalar
        """

        self.curr_x = self.update_x()
        self.curr_grad, prev_grad = self.update_grad()
        self.curr_dir = self.update_dir(prev_grad)
        self.curr_alpha = self.update_alpha()

        return self.curr_x, self.curr_grad, self.curr_dir, self.curr_alpha

    def update_grad(self):

        """
        Calculates and returns (g_k, g_(k-1)) the new and previous gradients respctively.
        """

        return self.func.grad(self.curr_x), self.curr_grad
    
    def update_dir(self, prev_grad):

        """
        Calculates and returns the new direction d_k
        """

        beta = np.dot(self.curr_grad, self.curr_grad - prev_grad) / np.dot(prev_grad, prev_grad)

        return -self.curr_grad + beta * self.curr_dir
        
    def update_alpha(self):

        """
        Calculates and returns the new step size alpha_k
        """

        numer = np.dot(self.curr_dir, self.curr_grad)
        denom = np.dot(np.dot(self.curr_dir, self.Q), self.curr_dir)

        return - numer / denom

    def update_x(self):

        """
        Calculates and returns x_k
        """

        return self.curr_x + self.curr_alpha * self.curr_dir


class FastRoute:

    """
    Implementation of fast route problem.
    """

    def __init__(self, start_x, start_y, finish_x, finish_y, velocities):

        """
        Initializes a FastRoute board.

        Arguments:
            start_x : x coordinate of starting point

            start_y : y coordinate of starting point

            finish_x : x coordinate of finish point

            finish_y : y coordinate of finish point

            velocities : an N-dimensional numpy vector,
                         the i-th element is the velocity at the i-th layer
        """

        self.start_x, self.start_y = start_x, start_y
        self.finish_x, self.finish_y = finish_x, finish_y
        self.velocities = velocities
        self.y = np.linspace(self.start_y, self.finish_y, len(velocities)+1)

    def __call__(self, x):

        """
        Calculates the time of the route given by a vector of x's

        Arguments:
            x : (N-1,) numpy array of the x coordinates of crossing points

        Return:
            total_time : the total travel time of the route
        """

        # Expand x with start and finish points
        x = self.expand_x(x)

        # distances vectors
        distances = self.find_distances(x)
        times = distances / self.velocities
        total_time = np.sum(times)

        return total_time

    def grad(self, x):
        """
        Evaluates the gradient of the objective function at (x[1],...,x[n-1])
        for the route given by a vector of x's

        Argument:
            x : (N-1,) numpy array of the x coordinates of cross points

        Return:
            grad : (N-1,) numpy array, the gradient of the objective w.r.t to (x[1],...,x[n-1])

            dt/dx_k = (x_k - x_(k-1)) / (v_k * d_k) - (x_(k+1) - x_k / (v_(k+1) * d_(k+1)) for k={1,2,...,n-1}
        """

        # Expand x with start and finish points
        x = self.expand_x(x)

        # distances vectors
        distances = self.find_distances(x)
        grad = (x[1:-1] - x[:-2]) / (self.velocities[:-1] * distances[:-1]) - (x[2:] - x[1:-1]) / (self.velocities[1:] * distances[1:])

        return grad

    def hessian(self, x):

        """
        Evaluates the hessian of the objective function at x=(x[1],...,x[n-1])
        for the route given by a vector of x's.
        See detailed calculation in solution pdf file.

        Arguemtns:
            x : (N-1,) numpy array of the x coordinates of cross points

        Return:
            hx : (N-1,N-1) numpy array, the hessian of the objective evaluated at (x[1],...,x[n-1])
        """

        # Expand x with start and finish points
        x = self.expand_x(x)

        # distances vectors
        distances = self.find_distances(x)
        term_1 = np.power(self.y[1:-1] - self.y[:-2], 2) / (self.velocities[:-1] * np.power(distances[:-1], 3))
        term_2 = np.power(self.y[2:] - self.y[1:-1], 2) / (self.velocities[1:] * np.power(distances[1:], 3))
        N = len(self.velocities)
        hx = np.zeros((N-1, N-1))
        for i in range(N-1):
            hx[i,i] = term_1[i] + term_2[i]
            if i > 0:
                hx[i-1,i] = -term_1[i]
            if i < N-2:
                hx[i+1,i] = -term_2[i]

        return hx
        
    def find_distances(self, x):

        """
        Calculates a vector of N distances.

        Arguments:
            x : (N-1,) numpy array of the x coordinates of cross points

        Return:
            (N-1,) numpy array of the distance traveled in each segment
        """

        return np.sqrt(np.power(x[1:] - x[:-1], 2) + np.power(self.y[1:] - self.y[:-1], 2))

    def expand_x(self, x):

        """
        Adds the starting point and finish point to vector x
        
        Arguments:
            x : (N-1,) numpy array of the x coordinates of cross points
            
        Return:
            (N-1,) numpy array with x coordinates of route 
            including start and finsh point
        """

        return np.append(np.insert(x, 0, self.start_x), self.finish_x)


def find_fast_route(objective, init, alpha=1, threshold=1e-3, max_iters=1e3):

    """
    Optimizes FastRoute objective using Newton’s method optimizer to 
    find a fast route between the starting point and finish point.
    
    Arguments:
    	objective : an initialized FastRoute object with preset start and finish points, 
                    velocities and initialization vector.

        init : (N-1,) numpy array, initial guess for the crossing points

	    alpha : step size for the NewtonOptimizer

	    threshold : stopping criteria |x_(k+1)- x_k |<threshold

	    max_iters : maximal number of iterations (stopping criteria)

    Return:
        route_time : scalar

        route : (N-1,) numpy array with x coordinate of the optimal route,
                    i.e., a vector of x-coordinates of crossing points (not including start and finish point)

        num_iters : number of iteration
    """

    opt = NewtonOptimizer(func=objective, alpha=alpha, init=init)
    route_time, route, num_iters = opt.optimize(threshold=threshold, max_iters=max_iters)

    return route_time, route, num_iters