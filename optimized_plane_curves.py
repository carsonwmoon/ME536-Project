import aerosandbox as asb
import aerosandbox.numpy as np
import matplotlib.pyplot as plt

opti = asb.Opti()

courses_init = np.radians(np.linspace(0., 90., 4))
courses_init = np.array([np.radians(20)])

for course_start in courses_init:

    # course_init = np.radians(90) # [rad]

    phi_max = np.radians(20) # [rad]
    phi_dot_max = np.radians(360.*1) # [rad/s]
    p_init = np.array([0., 0.]) # [m]
    speed = 5. # [m/s]
    g = 9.79768 # [m/s^2]
    t_end_init = 1. # [s]
    N = 150 # [-]

    # just in case
    phi_max = np.clip(phi_max, -np.radians(90), np.radians(90))

    # set initial values
    time_init = np.linspace(0, t_end_init, N)
    zero_to_one = np.linspace(0, 1, N)
    phis_init = np.linspace(0., 0., N)
    phis_dot_init = np.linspace(0., 0., N)
    courses_init = np.linspace(course_start, 0, N)
    
    px_init = speed * np.cos(course_start) * np.linspace(0, t_end_init, N) # np.linspace(0, t_end_init * speed, N)
    py_init = speed * np.sin(course_start) * (time_init**3 - 2*time_init**2 + time_init) # np.linspace(0, 0, N) # np.exp(-time_init) * time_init
    vx_init = speed * np.cos(course_start) * np.ones(N)
    vy_init = speed * np.sin(course_start) * (3*time_init**2 - 4*time_init + 1)
    ax_init = speed * np.cos(course_start) * np.zeros(N)
    ay_init = speed * np.sin(course_start) * (6*time_init - 4)

    # plot the initial values
    # plt.plot(px_init, py_init)
    # plt.axis('equal')
    # plt.title('Initial Path')
    # plt.show()
    # quit()

    # define opti variables
    t_end = opti.variable(
        init_guess=t_end_init, 
        lower_bound=0.,
        category='t_end'
    )
    time = np.linspace(0, t_end, N)
    phis = opti.variable(
        init_guess=phis_init,
        category='phis'
    )
    courses = opti.variable(
        init_guess=courses_init,
        category='courses'
    )
    vx = np.cos(courses) * speed
    vy = np.sin(courses) * speed

    px = opti.variable(
        init_guess=px_init
    )
    opti.constrain_derivative(
        variable=px,
        with_respect_to=time,
        derivative=vx,
    )
    py = opti.variable(
        init_guess=py_init
    )
    opti.constrain_derivative(
        variable=py,
        with_respect_to=time,
        derivative=vy,
    )

    ax = opti.derivative_of(
        vx,
        with_respect_to=time,
        derivative_init_guess=ax_init,
    )
    ay = opti.derivative_of(
        vy,
        with_respect_to=time,
        derivative_init_guess=ay_init,
    )
    phis_dot = opti.derivative_of(
        phis,
        with_respect_to=time,
        derivative_init_guess=phis_dot_init,
        method='forward euler' # this helps with the solver to not be so shaky
    )

    opti.subject_to([
        phis_dot <= phi_dot_max,
        phis_dot >= -phi_dot_max,
        phis < phi_max,
        phis > -phi_max,
    ])

    # constrain initial values
    opti.subject_to([
        px[0] == p_init.item(0),
        py[0] == p_init.item(1),
        courses[0] == course_start,
    ])

    # constrain final values
    opti.subject_to([
        py[-1] == 0, # final y position is 0
        vy[-1] == 0, # final y velocity is 0
        courses[-1] == 0, # final course is 0
        phis[-1] == 0, # final phi is 0
        phis_dot[-1] == 0, # final phi_dot is 0
    ])

    # constrain dynamics follow a plane turning model
    omega = opti.derivative_of(
        courses,
        with_respect_to=time,
        derivative_init_guess=g * np.tan(phis_init),
    )
    opti.subject_to([
        omega == g * np.tan(phis),
    ])

    opti.subject_to([
        courses <= course_start,
        courses >= -course_start,
        py >= 0,
    ])

    # minimize time to reach the final position
    opti.minimize(t_end)

    sol = opti.solve(
        verbose=True,
        max_runtime=15.*4,
        max_iter=1000,
        behavior_on_failure='return_last'
    )

    # design_parameters = {k:sol(opti.variables_categorized[k]) for k in opti.variables_categorized}
    # print('Design Parameters:')
    # for k, v in design_parameters.items():
    #     print(f'{k}: {v}')
    # quit()

    # extract the results
    t_end = sol.value(t_end)
    time = sol.value(time)
    phis = sol.value(phis)
    courses = sol.value(courses)
    vx = sol.value(vx)
    vy = sol.value(vy)
    px = sol.value(px)
    py = sol.value(py)
    ax = sol.value(ax)
    ay = sol.value(ay)
    phis_dot = sol.value(phis_dot)
    omega = sol.value(omega)

    # computed values
    p = np.vstack((px, py))
    path_length = np.sum(np.linalg.norm(np.diff(p, axis=1), axis=0))
    status = sol.stats()['success']


    # print the results
    if status:
        print(f't_end = \n{t_end}')
        print(f'time = \n{time}')
        print(f'phis = \n{phis}')
        print(f'courses = \n{courses}')
        print(f'vx = \n{vx}')
        print(f'vy = \n{vy}')
        print(f'px = \n{px}')
        print(f'py = \n{py}')
        print(f'ax = \n{ax}')
        print(f'ay = \n{ay}')
        print(f'phis_dot = \n{phis_dot}')
        print(f'omega = \n{omega}')

    # print important values
    print(f"Solution Status: {status}")
    print(f't_end = \n{t_end}')
    print(f'path_length = \n{path_length}')

    # plot the results
    if status:
        plt.plot(px, py, label='Optimized Path')
    else:
        plt.plot(px, py, label='Unsuccessful Path')
    plt.plot(p_init[0], p_init[1], 'ro', label='Initial Position')
    plt.plot(px_init, py_init, 'k--', label='Initial Path')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.axis('equal')
    plt.legend()
    plt.show()

    plt.plot(time, phis*180/np.pi, label='phi')
    plt.plot(time, phis_dot*180/np.pi, label='phi_dot')
    plt.plot(time, courses*180/np.pi, label='course')
    plt.xlabel('time [s]')
    plt.legend()
    plt.show()
