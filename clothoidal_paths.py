import numpy as np
import matplotlib.pyplot as plt
from typing import Union, Callable, Optional, List
from scipy.special import fresnel
from scipy.optimize import fsolve

# goal: generate C1 continuous path for a 2D plane that can turn in a smallest circle of radius R from a set of waypoints

p0 = np.array([0, 0])
p1 = np.array([1, 0])
p2 = np.array([1, 1])
R_max = 2

waypoints = [p0, p1, p2]

plt.axis('equal')

# compute an euler spiral

def angle_to_t(angle: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    return np.sqrt(angle*2./np.pi)

def get_curve(angle: float, N=1000):
    def val(a):
        if a >= 0:
            return angle_to_t(a)
        else:
            return -angle_to_t(-a)
    ts = np.linspace(0, val(angle), N)
    ys, xs = fresnel(ts)
    ys = np.abs(ys)
    path = np.array([xs, ys]).T
    return path

def rot(theta: float, p: np.ndarray) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    rotated = np.dot(R, p.T).T
    return rotated

def move_to_end_of(mover: np.ndarray, stayer: np.ndarray, stayer_on_head=False, mover_on_head=True) -> np.ndarray:
    if stayer_on_head:
        stayer_translation = stayer[0].reshape(-1, 1)
    else:
        stayer_translation = stayer[-1].reshape(-1, 1)

    if mover_on_head:
        mover_translation = mover[0].reshape(-1, 1)
    else:
        mover_translation = mover[-1].reshape(-1, 1)

    return mover + np.tile(stayer_translation, mover.shape[0]).T - np.tile(mover_translation, mover.shape[0]).T

def get_half_curve(half_angle, angle_ratio) -> np.ndarray:

    if isinstance(angle_ratio, np.ndarray):
        angle_ratio = angle_ratio.item(0)

    angle_1 = -half_angle / (2 + angle_ratio)
    angle_2 = half_angle - 2*angle_1

    seg_1 = get_curve(angle_1)

    seg_2_raw = get_curve(-angle_1)
    seg_2_rotated = rot(2*angle_1, seg_2_raw)
    seg_2 = move_to_end_of(seg_2_rotated, seg_1, stayer_on_head=False, mover_on_head=False)
    seg_2 = np.flip(seg_2, axis=0)

    seg_3_raw = get_curve(angle_2)
    seg_3_rotated = rot(np.pi+2*angle_1, seg_3_raw)
    seg_3 = move_to_end_of(seg_3_rotated, seg_2, stayer_on_head=False, mover_on_head=True)

    half_seg = np.concatenate([seg_1, seg_2, seg_3])

    # plt.plot(*zip(*seg_1), lw=3, label='seg_1')
    # plt.scatter(*seg_1[0], c='r', s=100, label='seg_1_start')
    # # plt.plot(*zip(*seg_2_raw), lw=2, label='seg_2_raw')
    # # plt.scatter(*seg_2_raw[0], c='r', s=100, label='seg_2_raw_start')
    # # plt.plot(*zip(*seg_2_rotated), lw=2, label='seg_2_rotated')
    # plt.plot(*zip(*seg_2), lw=2, label='seg_2')
    # plt.scatter(*seg_2[0], c='r', s=100, label='seg_2_start')
    # plt.plot(*zip(*seg_3), lw=1, label='seg_3')
    # plt.scatter(*seg_3[0], c='r', s=100, label='seg_3_start')
    # plt.legend()
    # plt.show()

    return half_seg

def CI_SI(angle: Union[float, np.ndarray]) -> Union[tuple[float, float], tuple[np.ndarray, np.ndarray]]:
    t = angle_to_t(angle)
    S, C = fresnel(t)
    SI = t * S + (np.cos(np.pi/2*t**2) - 1.) / np.pi
    CI = t * C + np.sin(np.pi/2*t**2) / np.pi
    return SI, CI

def get_circle_center(angle: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    SI, CI = CI_SI(angle)
    cx = CI
    cy = SI + 1. / np.pi
    t = angle_to_t(angle)
    return np.array([cx, cy]) / t

half_angle = -90*np.pi/180
angle_ratio = 5 # angle_2/angle_1

N = 100
full_angles = np.radians(np.linspace(180, -180, N))
# full_angles = np.concatenate([full_angles, -full_angles])
# full_angles = np.radians(np.array([-150]))

angle_ratios = np.zeros_like(full_angles)

for i, full_angle in enumerate(full_angles):
    half_angle = full_angle/2
    def get_error(angle_ratio) -> float:
        half_seg = get_half_curve(half_angle, angle_ratio)
        error_pos = half_seg[-1][1]
        return error_pos

    angle_ratio = fsolve(get_error, x0=angle_ratio)[0]
    angle_ratios[i] = angle_ratio
    # print(f'{half_angle} {angle_ratio}')

    first_half_seg = get_half_curve(half_angle, angle_ratio)
    error_pos = first_half_seg[-1][1]

    second_half_seg = first_half_seg.copy()
    second_half_seg[:,1] = -second_half_seg[:,1]
    second_half_seg = rot(np.pi+2*half_angle, second_half_seg)
    second_half_seg = move_to_end_of(second_half_seg, first_half_seg, stayer_on_head=False, mover_on_head=False)
    second_half_seg = np.flip(second_half_seg, axis=0)

    full_seg = np.concatenate([first_half_seg, second_half_seg])

    plt.plot(*zip(*full_seg), lw=1, label=f'${np.degrees(full_angle):.0f}\degree$')
    # plt.plot([0, first_half_seg[-1][0], full_seg[-1][0]], [0, first_half_seg[-1][1], full_seg[-1][1]], lw=1, c='k', linestyle='--')
    plt.scatter(*second_half_seg[0], c='r', s=1)
# plt.legend()
# plt.title("Carson's Composite Clothoidal Curves")
plt.tight_layout()
plt.show()
# quit()

data = np.stack([full_angles, angle_ratios], axis=1)

print(data)

plt.plot(np.degrees(full_angles), angle_ratios, label='Actual', lw=6)
plt.xlabel('Half Angles [deg]')
plt.ylabel(r"Angle Ratios $\frac{\theta_2}{\theta_1}$")
plt.title('Angle Ratios vs Half Angles')
# plt.show()

# fit a parametric curve to the half_angles and angle_ratios
import numpy.polynomial.polynomial as poly
coefs = poly.polyfit(full_angles, angle_ratios, 4)
print(coefs)
# plot the fitted curve
fitted_angle_ratios = poly.polyval(full_angles, coefs)
plt.plot(np.degrees(full_angles), fitted_angle_ratios, label='Fitted', lw=2)
plt.legend()
plt.show()


plt.plot(np.degrees(full_angles), angle_ratios - fitted_angle_ratios, label='Error')
plt.xlabel('Half Angles [deg]')
plt.yscale('log')
plt.title('Error of Fitted Curve')
plt.show()


