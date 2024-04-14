import aerosandbox as asb
import aerosandbox.numpy as np
import matplotlib.pyplot as plt

opti = asb.Opti()

phi_max = np.radians(30) # [rad]
phi_dot_max = np.radians(1) # [rad/s]
p_init = np.array([0., 0.]) # [m]
course_init = np.radians(30) # [rad]
speed = 10. # [m/s]
g = 9.79768 # [m/s^2]
t_end_init = 10. # [s]
N = 100 # [-]

# set initial values
time_init = np.linspace(0, t_end_init, N)
phis_init = np.linspace(0., 0., N)
phis_dot_init = np.linspace(0., 0., N)
px_init = np.linspace(0, 1, N)
py_init = np.linspace(0, 0, N) # np.exp(-time_init) * time_init
vx_init = np.linspace(0, 0, N) # np.exp(-time_init) * (1. - time_init)
vy_init = np.linspace(speed, speed, N) # np.linspace(1, 1, N)
ax_init = np.linspace(0, 0, N) # np.linspace(0, 0, N)
ay_init = np.linspace(0, 0, N) # np.exp(-time_init) * (time_init - 2.)
courses_init = np.linspace(course_init, 0, N)

# plot the initial values
# plt.plot(px_init, py_init)
# plt.axis('equal')
# plt.show()
# quit()

# define opti variables
t_end = opti.variable(
    init_guess=t_end_init, 
    lower_bound=0
)
time = np.linspace(0, t_end, N)
phis = opti.variable(
    init_guess=phis_init
)
courses = opti.variable(
    init_guess=courses_init
)
vx = np.cos(courses) * speed
vy = np.sin(courses) * speed

px = opti.variable(
    init_guess=px_init
)
opti.constrain_derivative(
    variable=px,
    with_respect_to=time,
    derivative=vx
)
py = opti.variable(
    init_guess=py_init
)
opti.constrain_derivative(
    variable=py,
    with_respect_to=time,
    derivative=vy
)

ax = opti.derivative_of(
    vx,
    with_respect_to=time,
    derivative_init_guess=ax_init
)
ay = opti.derivative_of(
    vy,
    with_respect_to=time,
    derivative_init_guess=ay_init
)
phi_dot = opti.derivative_of(
    phis,
    with_respect_to=time,
    derivative_init_guess=phis_dot_init
)

opti.subject_to([
    np.diff(phis) <= phi_dot_max * np.diff(time),
    np.diff(phis) >= -phi_dot_max * np.diff(time),
    phis <= phi_max,
    phis >= -phi_max,
])

# constrain initial values
opti.subject_to([
    px[0] == p_init.item(0),
    py[0] == p_init.item(1),
    courses[0] == courses_init[0],
    vx[0] == np.cos(courses_init[0]) * speed,
    vy[0] == np.sin(courses_init[0]) * speed,
    phis[0] == phis_init[0],
])

# constrain final values
opti.subject_to([
    py[-1] == 0, # final y position is 0
    vy[-1] == 0, # final y velocity is 0
    courses[-1] == 0, # final course is 0
    phis[-1] == 0, # final phi is 0
])

# constrain dynamics follow a plane turning model
omega = opti.derivative_of(
    courses,
    with_respect_to=time,
    derivative_init_guess=courses_init
)
opti.subject_to([
    omega == g * np.tan(phis),
])


# minimize time to reach the final position
opti.minimize(t_end)

sol = opti.solve(
    verbose=True,
    max_runtime=15.,
    behavior_on_failure='return_last'
)

phis = sol.value(phis)
px = sol.value(px)
py = sol.value(py)
courses = sol.value(courses)
time = sol.value(time)

# plot the results
plt.plot(px, py)
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.title('optimized path')
plt.axis('equal')
plt.show()

plt.plot(time, phis*180/np.pi)
plt.xlabel('time [s]')
plt.ylabel('phi [rad]')
plt.title('phi vs time')
plt.show()

plt.plot(time, courses*180/np.pi)
plt.xlabel('time [s]')
plt.ylabel('phi [rad]')
plt.title('phi vs time')
plt.show()