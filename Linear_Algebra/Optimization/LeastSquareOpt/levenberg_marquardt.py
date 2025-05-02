import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data
true_a = 2.5
true_b = 0.9
x_data = np.linspace(0, 5, 50)
y_true = true_a * np.exp(true_b * x_data)
y_data = y_true + np.random.normal(0, 1.0, size=x_data.shape)  # Add noise

# Initial parameter guess: [a, b]
params = np.array([1.0, 0.5])
lambda_ = 0.01  # Damping factor

def model(x, a, b):
    return a * np.exp(b * x)

def residuals(params):
    a, b = params
    return y_data - model(x_data, a, b)

def jacobian(params):
    a, b = params
    J = np.zeros((x_data.size, 2))
    exp_bx = np.exp(b * x_data)
    J[:, 0] = -exp_bx
    J[:, 1] = -a * x_data * exp_bx
    return J

# Levenberg-Marquardt iterations
for iteration in range(1000):
    r = residuals(params)
    J = jacobian(params)
    H = J.T @ J
    g = J.T @ r

    # Solve for parameter update
    dp = np.linalg.inv(H + lambda_ * np.eye(2)) @ g
    new_params = params - dp

    # Compute new residuals and error
    new_r = y_data - model(x_data, *new_params)
    if np.sum(new_r ** 2) < np.sum(r ** 2):
        params = new_params
        lambda_ /= 10
    else:
        lambda_ *= 10

# Final model prediction
y_pred = model(x_data, *params)

# Plot
plt.figure(figsize=(10, 6))
plt.scatter(x_data, y_data, label='Noisy Data', color='red')
plt.plot(x_data, y_true, label='True Function', linestyle='--', color='green')
plt.plot(x_data, y_pred, label='Fitted Curve (LM)', color='blue')
plt.title('Levenberg-Marquardt Curve Fitting')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()

# Print final parameters
print("Estimated Parameters:")
print(f"a = {params[0]:.4f}, b = {params[1]:.4f}")



import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data for a quadratic model
true_a = 2.0
true_b = 1.5
true_c = 0.5
x_data = np.linspace(-5, 5, 50)
y_true = true_a * x_data**2 + true_b * x_data + true_c
y_data = y_true + np.random.normal(0, 3.0, size=x_data.shape)  # Add noise

# Initial parameter guess: [a, b, c]
params = np.array([1.0, 1.0, 1.0])
lambda_ = 0.01  # Damping factor

def model(x, a, b, c):
    return a * x**2 + b * x + c

def residuals(params):
    a, b, c = params
    return y_data - model(x_data, a, b, c)

def jacobian(params):
    a, b, c = params
    J = np.zeros((x_data.size, 3))
    J[:, 0] = -x_data**2  # Derivative wrt a
    J[:, 1] = -x_data     # Derivative wrt b
    J[:, 2] = -1          # Derivative wrt c
    return J

# Levenberg-Marquardt iterations
for iteration in range(1000):
    r = residuals(params)
    J = jacobian(params)
    H = J.T @ J
    g = J.T @ r

    # Solve for parameter update
    dp = np.linalg.inv(H + lambda_ * np.eye(3)) @ g
    new_params = params - dp

    # Compute new residuals and error
    new_r = y_data - model(x_data, *new_params)
    if np.sum(new_r ** 2) < np.sum(r ** 2):
        params = new_params
        lambda_ /= 10
    else:
        lambda_ *= 10

# Final model prediction
y_pred = model(x_data, *params)

# Plot
plt.figure(figsize=(10, 6))
plt.scatter(x_data, y_data, label='Noisy Data', color='red')
plt.plot(x_data, y_true, label='True Function', linestyle='--', color='green')
plt.plot(x_data, y_pred, label='Fitted Curve (LM)', color='blue')
plt.title('Levenberg-Marquardt Polynomial Fitting')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()

# Print final parameters
print("Estimated Parameters:")
print(f"a = {params[0]:.4f}, b = {params[1]:.4f}, c = {params[2]:.4f}")



import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data for a sinusoidal model
true_amplitude = 3.0
true_frequency = 1.5
true_phase = 0.5
x_data = np.linspace(0, 10, 50)
y_true = true_amplitude * np.sin(true_frequency * x_data + true_phase)
y_data = y_true + np.random.normal(0, 0.3, size=x_data.shape)  # Add noise

# Initial parameter guess: [amplitude, frequency, phase]
params = np.array([1.0, 1.0, 0.0])
lambda_ = 0.01  # Damping factor

def model(x, a, f, p):
    return a * np.sin(f * x + p)

def residuals(params):
    a, f, p = params
    return y_data - model(x_data, a, f, p)

def jacobian(params):
    a, f, p = params
    J = np.zeros((x_data.size, 3))
    J[:, 0] = np.sin(f * x_data + p)  # Derivative wrt a
    J[:, 1] = a * x_data * np.cos(f * x_data + p)  # Derivative wrt f
    J[:, 2] = a * np.cos(f * x_data + p)  # Derivative wrt p
    return J

# Levenberg-Marquardt iterations
for iteration in range(1000):
    r = residuals(params)
    J = jacobian(params)
    H = J.T @ J
    g = J.T @ r

    # Solve for parameter update
    dp = np.linalg.inv(H + lambda_ * np.eye(3)) @ g
    new_params = params - dp

    # Compute new residuals and error
    new_r = y_data - model(x_data, *new_params)
    if np.sum(new_r ** 2) < np.sum(r ** 2):
        params = new_params
        lambda_ /= 10
    else:
        lambda_ *= 10

# Final model prediction
y_pred = model(x_data, *params)

# Plot
plt.figure(figsize=(10, 6))
plt.scatter(x_data, y_data, label='Noisy Data', color='red')
plt.plot(x_data, y_true, label='True Function', linestyle='--', color='green')
plt.plot(x_data, y_pred, label='Fitted Curve (LM)', color='blue')
plt.title('Levenberg-Marquardt Sinusoidal Fitting')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()

# Print final parameters
print("Estimated Parameters:")
print(f"Amplitude = {params[0]:.4f}, Frequency = {params[1]:.4f}, Phase = {params[2]:.4f}")
