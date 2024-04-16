import aerosandbox as asb
import aerosandbox.numpy as np
import matplotlib.pyplot as plt

opti = asb.Opti()

courses_init = np.radians(np.linspace(1., 90., 10))
courses_init = np.array([np.radians(90)])

phi_max = np.radians(10) # [rad]
phi_dot_max = np.radians(360.*.1) # [rad/s]
p_init = np.array([0., 0.]) # [m]
speed = 5. # [m/s]
g = 9.79768 # [m/s^2]
t_end_init = 1. # [s]
N = 250 # [-]

# just in case
phi_max = np.clip(phi_max, -np.radians(90), np.radians(90))

for course_end in courses_init:

    # course_init = np.radians(90) # [rad]

    # set initial values
    time_init = np.linspace(0, t_end_init, N)
    zero_to_one = np.linspace(0, 1, N)
    phis_init = np.linspace(0., 0., N)
    phis_dot_init = np.linspace(0., 0., N)
    courses_init = np.linspace(0, course_end, N)
    
    px_init = speed * (zero_to_one + .5 * (np.cos(course_end) - 1) * zero_to_one**2)
    py_init = -speed * np.sin(course_end) * (zero_to_one**3 - zero_to_one**4)
    vx_init = speed * (1 + (np.cos(course_end) - 1) * zero_to_one)
    vy_init = -speed * np.sin(course_end) * (3*zero_to_one**2 - 4*zero_to_one**3)
    
    # px_init = speed * (zero_to_one + .5 * (np.cos(course_end) - 1) * zero_to_one**2)
    # py_init = -speed * np.sin(course_end) * (zero_to_one**3 - zero_to_one**4)
    # vx_init = speed * (1 + (np.cos(course_end) - 1) * zero_to_one) * (1. - (10 - 10 * zero_to_one) / (np.sqrt(1+(10 - 10 * zero_to_one)**2)))
    # vy_init = -speed * np.sin(course_end) * (-15 * zero_to_one**4 + 28 * zero_to_one**3 - 12 * zero_to_one**2)

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

    # derived variables
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
    phis_dot = opti.derivative_of(
        phis,
        with_respect_to=time,
        derivative_init_guess=phis_dot_init,
        method='forward euler' # this helps with the solver to not be so shaky
    )

    # constrain initial values
    opti.subject_to([
        px[0] == p_init.item(0), # initial x position is the desired x position
        py[0] == p_init.item(1), # initial y position is the desired y position
        py[0] == 0, # initial y position is 0
        vy[0] == 0, # initial y velocity is 0
        vx[0] == speed, # initial x velocity is the desired speed
        courses[0] == 0, # initial course is 0
        phis[0] == 0, # initial phi is 0
    ])

    # constrain final values
    opti.subject_to([
        py[-1] == 0,
        courses[-1] == course_end,
    ])

    # constrain phis and phis_dot to be within max values
    opti.subject_to([
        phis_dot <= phi_dot_max,
        phis_dot >= -phi_dot_max,
        phis < phi_max,
        phis > -phi_max,
    ])

    # constrain dynamics to follow a plane turning model
    omega = opti.derivative_of(
        courses,
        with_respect_to=time,
        derivative_init_guess=g * np.tan(phis_init),
    )
    opti.subject_to([
        omega == g * np.tan(phis),
    ])

    # I found that constraining these helped the solver to converge faster
    opti.subject_to([
        py <= 0,
        courses <= np.radians(90),
        courses >= -np.radians(90),
        # phis_dot[0] <= 0.,
        phis_dot[0] == -phi_dot_max, # initial phi_dot is 0
        np.diff(phis_dot)[0] == 0, # initial phi_dot is 0
        np.diff(phis_dot)[-1] == 0, # initial phi_dot is 0
    ])

    # minimize time to reach the final position
    opti.minimize(t_end)

    fig, axes = plt.subplots(2)
    traj_axis = axes[0]
    curve_init = traj_axis.plot(px_init, py_init, 'k--', label='Initial Path')[0]
    curve = traj_axis.plot([], [], 'b-', label='Optimized Path')[0]
    traj_axis.set_xlabel('x [m]')
    traj_axis.set_ylabel('y [m]')
    traj_axis.axis('equal')
    traj_axis.legend()
    angle_axis = axes[1]
    phis_plot = angle_axis.plot(time_init, phis_init*180/np.pi, label='phi')[0]
    phis_dot_plot = angle_axis.plot(time_init, phis_dot_init*180/np.pi, label='phi_dot')[0]
    courses_plot = angle_axis.plot(time_init, courses_init*180/np.pi, label='course')[0]
    omegas_plot = angle_axis.plot(time_init, g*np.tan(phis_init), label='omega')[0]
    px_plot = angle_axis.plot(time_init, px_init, label='px')[0]
    py_plot = angle_axis.plot(time_init, py_init, label='py')[0]
    vx_plot = angle_axis.plot(time_init, vx_init, label='vx')[0]
    vy_plot = angle_axis.plot(time_init, vy_init, label='vy')[0]
    angle_axis.set_xlabel('time [s]')
    angle_axis.set_ylabel('angle [deg]')
    angle_axis.legend()
    plt.title(f'Intermediate Results for {round(course_end*180/np.pi,1)} deg course')
    plt.subplots_adjust(bottom=.06, hspace=.282, top=.97)

    def plot_trajectory(i=None):
        if i is None:
            curve.set_data(px, py)
            phis_plot.set_data(time, phis*180/np.pi)
            phis_dot_plot.set_data(time, phis_dot*180/np.pi)
            courses_plot.set_data(time, courses*180/np.pi)
            omegas_plot.set_data(time, omega*180/np.pi)
            px_plot.set_data(time, px)
            py_plot.set_data(time, py)
            vx_plot.set_data(time, vx)
            vy_plot.set_data(time, vy)
            traj_axis.relim()
            traj_axis.autoscale_view()
            angle_axis.relim()
            angle_axis.autoscale_view()
            if status:
                plt.title(f'Optimized Trajectory for {round(course_end*180/np.pi,1)} deg course')
            else:
                plt.title(f'Unsuccessful Trajectory for {round(course_end*180/np.pi,1)} deg course')
        else:
            curve.set_data(opti.value(px), opti.value(py))
            phis_plot.set_data(opti.value(time), opti.value(phis)*180/np.pi)
            phis_dot_plot.set_data(opti.value(time), opti.value(phis_dot)*180/np.pi)
            courses_plot.set_data(opti.value(time), opti.value(courses)*180/np.pi)
            omegas_plot.set_data(opti.value(time), opti.value(omega)*180/np.pi)
            px_plot.set_data(opti.value(time), opti.value(px))
            py_plot.set_data(opti.value(time), opti.value(py))
            vx_plot.set_data(opti.value(time), opti.value(vx))
            vy_plot.set_data(opti.value(time), opti.value(vy))
        
        traj_axis.relim()
        traj_axis.autoscale_view()
        angle_axis.relim()
        angle_axis.autoscale_view()

        if i is None:
            plt.show()
        else:
            plt.pause(0.01)

    sol = opti.solve(
        verbose=True,
        max_runtime=60.,
        max_iter=1000,
        behavior_on_failure='return_last',
        callback=plot_trajectory
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
    phis_dot = sol.value(phis_dot)
    omega = sol.value(omega)

    # computed values
    p = np.vstack((px, py))
    path_length = np.sum(np.linalg.norm(np.diff(p, axis=1), axis=0))
    status = sol.stats()['success']

    # print the results
    if status and False:
        print(f't_end = \n{t_end}')
        print(f'time = \n{time}')
        print(f'phis = \n{phis}')
        print(f'courses = \n{courses}')
        print(f'vx = \n{vx}')
        print(f'vy = \n{vy}')
        print(f'px = \n{px}')
        print(f'py = \n{py}')
        print(f'phis_dot = \n{phis_dot}')
        print(f'omega = \n{omega}')

    # print important values
    print(f"Solution Status: {status}")
    print(f't_end = \n{t_end}')
    print(f'path_length = \n{path_length}')

    # plot the final trajectory
    plot_trajectory()