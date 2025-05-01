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
