import numpy as np

def numerical_gradient(func, x, params, epsilon=1e-8):
    grad = np.zeros_like(x)
    for i in range(len(x)):
        if i == 2:
            # epsilon = 1e5 # Increase epsilon for the third dimension (mass) to avoid numerical instability
            pass
        x_plus = np.copy(x)
        x_minus = np.copy(x)
        x_plus[i] += epsilon
        x_minus[i] -= epsilon
        grad[i] = (func(x_plus, params) - func(x_minus, params)) / (2 * epsilon)
        if i == 2:
            # print(f"Partial derivative with respect to x[{i}]: {grad[i]}")
            pass
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


def adam_optimizer(func, initial_x, learning_rate=0.001, num_iterations=1000, beta1=0.9, beta2=0.999, epsilon=1e-8, *args):
    x = np.asarray(initial_x, dtype=float)
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    t = 0
    for i in range(num_iterations):
        t += 1
        grad = numerical_gradient(func, x, epsilon, *args)
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad ** 2)
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        x = x - learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
        print(f"Iteration {i+1}: x = {x}, f(x) = {func(x, *args)}")
    return x


def adam_optimizer_with_different_learning_rates(func, initial_x, params, learning_rates, num_iterations=1000, beta1=0.9, beta2=0.999, epsilon=1e-8):
    # Adam optimizer with different learning rates for each dimension
    # Adam is a popular optimization algorithm that combines the advantages of AdaGrad and RMSProp
    # It uses both the average of past gradients and the average of past squared gradients
    # It also includes bias correction terms to account for the fact that the averages are initialized at zero
    # The learning rates are used to scale the updates for each dimension
    # The parameters beta1 and beta2 control the decay rates for the moving averages
    # The parameter epsilon is used to prevent division by zero

    x = np.asarray(initial_x, dtype=float)
    learning_rates = np.asarray(learning_rates, dtype=float)
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
        x = x - learning_rates * m_hat / (np.sqrt(v_hat) + epsilon)
        print(f"Iteration {i+1}: x = {x}, f(x) = {func(x, params)}")
    return x

'''
# Example usage with a specific 3D function

# Define the 3D function to minimize
def function_to_minimize_3d(x):
    return x[0]**2 + x[1]**2 + x[2]**2 + 4*x[0] + 4*x[1] + 4*x[2] + 8

# Parameters
initial_x = [10, 10, 10]
learning_rate = 0.1
num_iterations = 100

# Run gradient descent with momentum
minimized_x = gradient_descent_3d_with_momentum(function_to_minimize_3d, initial_x, learning_rate, num_iterations)
print(f"Minimized value of x: {minimized_x}")
print(f"Function value at minimized x: {function_to_minimize_3d(minimized_x)}")
'''