import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

# givens
Va = 2.8 # [m/s]
g = 9.81 # [m/s^2]
phi_max = np.radians(89)

def phi_at_time(t: float, phi_max: float, phis: np.ndarray, pad=False) -> float:
    # return phi_max
    if pad:
        phis = np.concatenate([np.array([0]), phis, np.array([phi_max])])
        phis_all = np.clip(phis_all, -phi_max, phi_max)
        phi = np.interp(t, np.linspace(0, t_end, len(phis_all)), phis_all).item(0)
        return phi
    else:
        phi = np.interp(t, np.linspace(0, t_end, len(phis)), phis).item(0)
        return phi

def x_dot(t: float, x: np.ndarray, phis: np.ndarray, pad: bool) -> np.ndarray:
    p = x[:2]
    v = x[2:]

    phi = phi_at_time(t, phi_max, phis, pad)
    # curvature = (g * np.tan(phi)) / Va**2
    # a = V**2 / R = V**2 * curvature
    phi_sign = 1 if np.sign(phi) == 0 else -1
    a_hat = np.array([[0, phi_sign], [-phi_sign, 0]]) @ v
    a_hat /= np.linalg.norm(a_hat)
    a = g * np.tan(phi) * a_hat
    
    xdot = np.concatenate([v.flatten(), a.flatten()])
    return xdot

def simulate(phis: np.ndarray, pad=False) -> tuple[np.ndarray, np.ndarray, float]:
    sol = solve_ivp(lambda t, x: x_dot(t, x, phis, pad), t_span, x0, method='Radau', t_eval=ts)
    p = sol.y[:2]
    v = sol.y[2:]
    path_length = np.sum(np.linalg.norm(np.diff(p, axis=1), axis=0))
    return p, v, path_length

def objective(phis: np.ndarray) -> float:
    _, _, path_length = simulate(phis, pad=True)
    return path_length

def final_pos_is_good(phis: np.ndarray) -> float:
    p, v, path_length = simulate(phis)
    p_end_actual = p.T[-1]
    diff = np.linalg.norm(p_end_actual - p_end).item(0)
    print(f'p_end_actual: {np.round(p_end_actual,3)}, diff = {round(diff,3)}')
    return diff

# # use scipy.integrate.solve_ivp to integrate x_dot
dir = np.radians(45)
p_start = np.array([[0., 0.]])
t_end = 3. # [s]

v_start = np.array([[np.cos(dir), np.sin(dir)], [-np.sin(dir), np.cos(dir)]]) @ np.array([[0], [Va]])
t_span = (0, t_end)
x0 = np.concatenate((p_start.flatten(), v_start.flatten()))

# initial guess
N = int(10000 * t_end)
ts = np.linspace(0, t_end, N)
phis_start = np.sin(5*ts) * phi_max
phis = phis_start
p, v, path_length = simulate(phis)

# plot the trajectory
plt.axis('equal')
plt.plot(p[0], p[1])
plt.xlabel('x')
plt.ylabel('y')
plt.title('Initial Trajectory')
plt.show()
quit()


p_end = np.array([[2., 4.]])
cons = ( # non negative values in inequality constriants mean that it's good
    {'type': 'ineq', 'fun': lambda x: phi_max - x},
    {'type': 'ineq', 'fun': lambda x: x - phi_max},
    {'type': 'ineq', 'fun': lambda x: 2. - final_pos_is_good(x)},
)
sol = minimize(objective, phis_start, constraints=cons)
print(sol)

# extract the optimized solution

phis = sol.x
p, v, path_length = simulate(phis)

# plot the trajectory
plt.axis('equal')
plt.plot(p[0], p[1])
plt.plot(p_end.item(0), p_end.item(1), 'ro')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Optimized Trajectory')
plt.show()

ts = np.linspace(0, t_end, 100)
phis = np.zeros_like(ts)
for i, t in enumerate(ts):
    phis[i] = phi_at_time(t, phi_max, phis_start)
plt.plot(ts, phis)
plt.xlabel('t')
plt.ylabel('phi')
plt.title('Phi vs Time')
plt.show()