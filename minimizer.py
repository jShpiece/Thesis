import numpy as np

def numerical_gradient(func, x, params, epsilon=1e-8):
    x = np.asarray(x, dtype=float)  # Ensure x is a NumPy array
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_plus = np.copy(x)
        x_minus = np.copy(x)
        x_plus[i] += epsilon
        x_minus[i] -= epsilon
        grad[i] = (func(x_plus, params) - func(x_minus, params)) / (2 * epsilon)
        # grad[i] = (func(x_plus) - func(x_minus)) / (2*epsilon)
    return grad


def gradient_descent(func, initial_x, learning_rate, num_iterations, momentum, params=None):
    x = np.asarray(initial_x, dtype=float)
    velocity = np.zeros_like(x)  # Ensure velocity is a NumPy array
    grad_avg = np.zeros_like(x)
    points = []
    prev_func_val = func(x, params)
    tol = 1e-6
    for i in range(num_iterations):
        grad = numerical_gradient(func, x, params)
        grad_avg = momentum * grad_avg + (1 - momentum) * grad
        velocity = momentum * velocity - learning_rate * grad_avg
        x = x + velocity
        points.append(np.copy(x))
        # print(f"Iteration {i+1}: x = {x}, f(x) = {func(x, params)}")

        # Check for convergence
        current_func_val = func(x, params)
        if np.abs(current_func_val - prev_func_val) < tol:
            print(f"Converged at iteration {i+1}")
            break
    
        prev_func_val = current_func_val
    return x, points


def clip_gradients(grads, threshold):
    return np.clip(grads, -threshold, threshold)


def smooth_gradient(grads, beta, grad_avg=None):
    if grad_avg is None:
        grad_avg = np.zeros_like(grads)
    grad_avg = beta * grad_avg + (1 - beta) * grads
    return grad_avg


def adam_optimizer(func, initial_x, learning_rates, max_iterations=1000, beta1=0.9, beta2=0.999, tol=1e-6, params=None):
    x = np.asarray(initial_x, dtype=float)
    learning_rates = np.asarray(learning_rates, dtype=float)
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    t = 0
    points = []  # To store the points for visualization
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
        points.append(np.copy(x))  # Store the current point
        
        # Check for convergence
        current_func_val = func(x, params)
        if np.abs(current_func_val - prev_func_val) < tol:
            # print(f"Converged at iteration {i+1} at x = {x}")
            break
        prev_func_val = current_func_val
        
        # print(f"Iteration {i+1}: x = {x}, f(x) = {current_func_val}")

    return x, points


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# test against a 2d 'pit' function
def plot_optimization_path(points, true_value, func):
    points = np.array(points)
    x1 = np.linspace(-10, 10, 400)
    x2 = np.linspace(-10, 10, 400)
    X1, X2 = np.meshgrid(x1, x2)
    Z = func(np.array([X1, X2]))

    plt.figure(figsize=(10, 8))
    cp = plt.contour(X1, X2, Z, levels=50, cmap='viridis')
    plt.colorbar(cp)
    plt.plot(points[:, 0], points[:, 1], marker='o', label='Optimization Path')
    plt.scatter(*true_value, color='r', label='True Value', s=100)  # Plot true value
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.legend()
    plt.title('Optimization Path and Contour Plot')
    plt.show()


def pit_function(x):
    noise = np.random.normal(0, 0.1)
    return x[0]**2 + x[1]**2 + noise
