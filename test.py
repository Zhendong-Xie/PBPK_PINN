import numpy as np
from scipy.integrate import ode
import matplotlib.pyplot as plt 

# Define the ODE system
def f(t, y, args):
    dydt = -args * y + 1.0
    return dydt

# Initialize the solver
r = ode(f).set_integrator('zvode', method='bdf', order=15)

# Set initial conditions
y0 = [0.5]
t0 = 0.0
para = 1
r.set_initial_value(y0, t0)
r.set_f_params(para)

# Set up time points where you want the solution
t1 = 5.0
dt = 0.1
time_points = np.arange(t0, t1, dt)
solution = []
# Integrate
for t in time_points:
    if r.successful() and r.t < t:
        r.integrate(t)
    solution.append(r.y[0])

# Display the solution
for t, y in zip(time_points, solution):
    print(f't = {t:.2f}, y = {y:.4f}')

plt.figure()

plt.plot(time_points, solution)
plt.show()