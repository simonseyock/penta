import sympy as sp
from sympy import solve_poly_system
from sympy.solvers import solve
import matplotlib.pyplot as mpl
import numpy as np
from sympy.abc import x, y, a


def lin_eq_pp(p1, p2):
    return sp.Eq(y, (p2[1] - p1[1])/(p2[0] - p1[0]) * (x - p1[0]) + p1[1])


def lin_eq_sp(s, p):
    return sp.Eq(y, s * (x - p[0]) + p[1])


def mirror(p):
    return p[0], -p[1]


c = sp.Eq(1, x**2 + y**2)
f_1 = sp.Eq(y, - x / a + 1)

i_1 = solve_poly_system([c, f_1], x, y)

f_2 = lin_eq_pp((-a, 0), i_1[1]) # use second intersection

i_2 = solve_poly_system([c, f_2], x, y)

f_3 = lin_eq_sp(0, i_2[1]) # use second intersection

g = lin_eq_pp((-a, 0), mirror(i_1[1]))

sol = solve_poly_system([f_1, f_3], x, y)
subst = g.subs(x, sol[0][0]).subs(y, sol[0][1])
sol2 = solve(subst, a)

num_eval = [s.evalf() for s in sol2]
choose_a = num_eval[3]

print('Circle: ', c)
print('f_1: ', f_1)
print('intersections of Circle and f_1: ', i_1)
print('resulting equation f_2: ', f_2)
print('intersections of Circle and f_2 (subs 0.3): ', [(i[0].subs(a, 0.3), i[1].subs(a, 0.3)) for i in i_2])
print('first intersection of Circle and f_2: ', i_2[0])
print('resulting equation f_3: ', f_3)
print('mirrored equation f_2 (g)', g)
print('intersection of f_1 and f_3: ', sol)
print('subst: ', subst)
print('solve subst for a: ', sol2)
print('numerical solutions for a: ', num_eval)
print('-----------------------------------')
print('solution: ', sol2[3], '=', choose_a)

c_sol = solve(c, y)
lam_c_1 = sp.lambdify(x, c_sol[0], modules=['numpy'])
lam_c_2 = sp.lambdify(x, c_sol[1], modules=['numpy'])
lam_f_1 = sp.lambdify(x, solve(f_1.subs(a, choose_a), y)[0], modules=['numpy'])
lam_f_2 = sp.lambdify(x, solve(f_2.subs(a, choose_a), y)[0], modules=['numpy'])
f_3_y = solve(f_3.subs(a, choose_a), y)[0]
lam_f_3 = np.vectorize(lambda _: f_3_y)
lam_g = sp.lambdify(x, solve(g.subs(a, choose_a), y)[0], modules=['numpy'])

x_vals = np.linspace(-1, 1, 100)
c_1_vals = lam_c_1(x_vals)
c_2_vals = lam_c_2(x_vals)
f_1_vals = lam_f_1(x_vals)
f_2_vals = lam_f_2(x_vals)
f_3_vals = lam_f_3(x_vals)
g_vals = lam_g(x_vals)

mpl.axes(xlim=(-1,1), ylim=(-1,1))
mpl.plot(x_vals, f_1_vals)
mpl.plot(x_vals, f_2_vals)
mpl.plot(x_vals, f_3_vals)
mpl.plot(x_vals, g_vals)
mpl.plot(x_vals, c_1_vals, color='green')
mpl.plot(x_vals, c_2_vals, color='green')
mpl.show()
