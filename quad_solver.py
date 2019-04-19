import numpy as np

def solver(curr_pos, delta_y):
    a1 = 0.04476
    a2 = -1.869e-5
    b = 2*a2*curr_pos+a1
    d1 = (b)**2 + 4*a2*delta_y
    d2 = (b)**2 - 4*a2*delta_y
    sol1 = (-b + np.sqrt(d1))/(2*a2)
    sol2 = (-b + np.sqrt(d2))/(2*a2)
    sol3 = (-b - np.sqrt(d1))/(2*a2)
    sol4 = (-b - np.sqrt(d2))/(2*a2)
    return (sol1, sol2)
