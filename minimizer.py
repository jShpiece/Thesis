import numpy as np

def numerical_gradient(func, x, params, epsilon=1e-8):
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_plus = np.copy(x)
        x_minus = np.copy(x)
        x_plus[i] += epsilon
        x_minus[i] -= epsilon
        grad[i] = (func(x_plus, params) - func(x_minus, params)) / (2 * epsilon)
    return grad

def gradient_descent_3d_with_momentum(func, initial_x, learning_rate, num_iterations, momentum, params):
    # Momentum-based gradient descent for 3D functions
    # Momentum is a hyperparameter in the range [0, 1], where 0 is no momentum and 1 is full momentum
    # The idea is to add a fraction of the previous step to the current step, which can help accelerate convergence
    x = np.array(initial_x, dtype=float)
    velocity = np.zeros_like(x)
    for i in range(num_iterations):
        grad = numerical_gradient(func, x, params)
        velocity = momentum * velocity - learning_rate * grad
        x += velocity
        # print(f"Iteration {i+1}: x = {x}, f(x) = {func(x, params)}")
    return x


def adam_optimizer(func, initial_x, params, learning_rate=0.001, num_iterations=1000, beta1=0.9, beta2=0.999, epsilon=1e-8):
    x = np.asarray(initial_x, dtype=float)
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    t = 0
    for i in range(num_iterations):
        t += 1
        grad = numerical_gradient(func, x, params)
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad ** 2)
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        x = x - learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
    return x

