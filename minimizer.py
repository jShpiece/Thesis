import numpy as np
'''
Custom implementation of the Adam optimizer for minimizing a function of multiple variables.
Written to handle minimization of the chi-squared function for NFW lensing profile fitting.
'''

def numerical_gradient(func, x, params, epsilon=1e-8):
    '''
    Compute the numerical gradient of a function at a given point x.
    Parameters:
    func: Function to compute the gradient of.
    x: Point at which to compute the gradient.
    params: Additional parameters to pass to the function.
    epsilon: Small number to use for finite differences.
    Returns:
    grad: Numerical gradient of the function at x.
    '''
    x = np.asarray(x, dtype=float)  # Ensure x is a NumPy array
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_plus = np.copy(x)
        x_minus = np.copy(x)
        x_plus[i] += epsilon
        x_minus[i] -= epsilon
        grad[i] = (func(x_plus, params) - func(x_minus, params)) / (2 * epsilon)
    return grad


def clip_gradients(grads, threshold):
    # Clip gradients to a maximum value
    return np.clip(grads, -threshold, threshold)


def smooth_gradient(grads, beta, grad_avg=None):
    # Smooth the gradients using an exponential moving average
    if grad_avg is None:
        grad_avg = np.zeros_like(grads)
    grad_avg = beta * grad_avg + (1 - beta) * grads
    return grad_avg


def gradient_descent(func, initial_x, learning_rate, num_iterations, params):
    # A simple gradient descent optimizer
    x = initial_x
    x = np.asarray(x, dtype=float)
    path = []
    for i in range(num_iterations):
        grad = numerical_gradient(func, x, params, epsilon=1e-8)
        grad = clip_gradients(grad, 1) # Clip gradients to avoid large steps
        prev_x = x
        x = x - learning_rate * grad
        path.append(x)
        # Check for convergence
        if np.linalg.norm(x - prev_x) < 1e-6:
            break
    return x, path


def adam_optimizer(func, initial_x, learning_rates, max_iterations=1000, beta1=0.9, beta2=0.999, tol=1e-6, params=None):
    '''
    Minimize a function using the Adam optimizer.
    Parameters:
        func: Function to minimize.
        initial_x: Initial guess for the minimizer.
        learning_rates: Learning rates for each parameter.
        max_iterations: Maximum number of iterations to run.
        beta1: Exponential decay rate for the first moment estimates.
        beta2: Exponential decay rate for the second raw moment estimates.
        tol: Tolerance for convergence.
        params: Additional parameters to pass to the function.
    Returns:
        x: Final value of the minimizer.
        points: List of points visited during optimization.
    '''

    x = np.asarray(initial_x, dtype=float)
    learning_rates = np.asarray(learning_rates, dtype=float)
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    t = 0
    prev_func_val = func(x, params)
    grad_avg = np.zeros_like(x)
    
    for i in range(max_iterations):
        t += 1
        grad = numerical_gradient(func, x, params)
        grad = clip_gradients(grad, 1)
        grad_avg = smooth_gradient(grad, beta1, grad_avg)
        m = beta1 * m + (1 - beta1) * grad_avg # Update biased first moment estimate
        v = beta2 * v + (1 - beta2) * (grad_avg ** 2) # Update biased second raw moment estimate
        m_hat = m / (1 - beta1 ** t) # Bias correction
        v_hat = v / (1 - beta2 ** t) # Bias correction
        x -= learning_rates * m_hat / (np.sqrt(v_hat) + 1e-2) # Update parameters
        
        # Check for convergence
        current_func_val = func(x, params)
        if np.abs(current_func_val - prev_func_val) < tol:
            break
        prev_func_val = current_func_val
        

    return x
