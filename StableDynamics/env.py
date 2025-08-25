# Modified from: http://jakevdp.github.io/blog/2017/03/08/triple-pendulum-chaos/, https://github.com/locuslab/stable_dynamics.git
import numpy as np

from sympy.physics import mechanics
from sympy import Dummy, lambdify, srepr, symbols
from sympy.core.function import AppliedUndef
from sympy.core.sympify import sympify
from sympy.core.power import Pow
from sympy.physics.vector import Vector
from sympy.printing.printer import Printer

from scipy.integrate import odeint

import torch

########################
# Code for simulation

def integrate_pendulum(n, times,
                       initial_positions=135,
                       initial_velocities=0,
                       lengths=None, masses=1,
                       friction=0.3):
    """Integrate a multi-pendulum with `n` sections"""
    #-------------------------------------------------
    # Step 1: construct the pendulum model

    # Generalized coordinates and velocities
    # (in this case, angular positions & velocities of each mass)
    q = mechanics.dynamicsymbols('q:{0}'.format(n))
    u = mechanics.dynamicsymbols('u:{0}'.format(n))

    # mass and length
    m = symbols('m:{0}'.format(n))
    l = symbols('l:{0}'.format(n))

    # gravity and time symbols
    g, t = symbols('g,t')

    #--------------------------------------------------
    # Step 2: build the model using Kane's Method

    # Create pivot point reference frame
    A = mechanics.ReferenceFrame('A')
    P = mechanics.Point('P')
    P.set_vel(A, 0)

    # lists to hold particles, forces, and kinetic ODEs
    # for each pendulum in the chain
    particles = []
    forces = []
    kinetic_odes = []

    for i in range(n):
        # Create a reference frame following the i^th mass
        Ai = A.orientnew('A' + str(i), 'Axis', [q[i], A.z])
        Ai.set_ang_vel(A, u[i] * A.z)

        # Create a point in this reference frame
        Pi = P.locatenew('P' + str(i), l[i] * Ai.x)
        Pi.v2pt_theory(P, A, Ai)

        # Create a new particle of mass m[i] at this point
        Pai = mechanics.Particle('Pa' + str(i), Pi, m[i])
        particles.append(Pai)

        # Set forces & compute kinematic ODE
        forces.append((Pi, m[i] * g * A.x))
        # Add damping torque:
        #forces.append((Ai, -1 * friction * u[i] * A.z))

        kinetic_odes.append(q[i].diff(t) - u[i])

        P = Pi

    # Generate equations of motion
    KM = mechanics.KanesMethod(A, q_ind=q, u_ind=u,
                               kd_eqs=kinetic_odes)
    fr, fr_star = KM.kanes_equations(particles, forces)

    #-----------------------------------------------------
    # Step 3: numerically evaluate equations and integrate

    # initial positions and velocities – assumed to be given in degrees
    y0 = np.deg2rad(np.concatenate([np.broadcast_to(initial_positions, n),
                                    np.broadcast_to(initial_velocities, n)]))

    # lengths and masses
    if lengths is None:
        lengths = np.ones(n) / n
    lengths = np.broadcast_to(lengths, n)
    masses = np.broadcast_to(masses, n)

    # Fixed parameters: gravitational constant, lengths, and masses
    parameters = [g] + list(l) + list(m)
    parameter_vals = [9.81] + list(lengths) + list(masses)

    # define symbols for unknown parameters
    unknowns = [Dummy() for i in q + u]
    unknown_dict = dict(zip(q + u, unknowns))
    kds = KM.kindiffdict()

    # substitute unknown symbols for qdot terms
    mm_sym = KM.mass_matrix_full.subs(kds).subs(unknown_dict)
    fo_sym = KM.forcing_full.subs(kds).subs(unknown_dict)

    # create functions for numerical calculation
    mm_func = lambdify(unknowns + parameters, mm_sym)
    fo_func = lambdify(unknowns + parameters, fo_sym)

    # function which computes the derivatives of parameters
    def gradient(y, t, args):
        vals = np.concatenate((y, args))
        sol = np.linalg.solve(mm_func(*vals), fo_func(*vals))
        return np.array(sol).T[0]

    # ODE integration
    return odeint(gradient, y0, times, args=(parameter_vals,))
    #odeint

def simpleint(gradient, X, times, args=[]):
    rv = [X]
    for i in range(len(times) - 1):
        X = X + (times[i+1]-times[i]) * gradient(X, times[i+1], *args)
        rv.append(X)
    return rv

def get_xy_coords(p, lengths=None):
    """Get (x, y) coordinates from generalized coordinates p"""
    p = np.atleast_2d(p)
    n = p.shape[1] // 2
    if lengths is None:
        lengths = np.ones(n) / n
    zeros = np.zeros(p.shape[0])[:, None]
    x = np.hstack([zeros, lengths * np.sin(p[:, :n])])
    y = np.hstack([zeros, -lengths * np.cos(p[:, :n])])
    return np.cumsum(x, 1), np.cumsum(y, 1)

class TorchPrinter(Printer):
    printmethod = "_torchrepr"
    _default_settings = {
        "order": None
    }

    def _print_Add(self, expr, order=None):
        terms = list(map(self._print, self._as_ordered_terms(expr, order=order)))
        def __inner_sum(**kw):
            x = terms[0](**kw)
            for r in terms[1:]:
                v = r(**kw)
                x = torch.add(x, v)
            return x
        return __inner_sum

    def _print_Function(self, expr):
        __FUNCTION_MAP = { "sin": torch.sin, "cos": torch.cos, "tan": torch.tan, "Abs": torch.abs }

        if expr.func.__name__ in __FUNCTION_MAP:
            func = __FUNCTION_MAP[expr.func.__name__]
            args = [self._print(a) for a in expr.args]
            return lambda **kw: func(*(a(**kw) for a in args))
        else:
            key = f"{expr.func.__name__}_{', '.join([str(a) for a in expr.args])}"
            return lambda **kw: kw[key]

    def _print_Half(self, expr):
        return lambda **kw: 0.5

    def _print_Integer(self, expr):
        return lambda **kw: expr.p

    def _print_NaN(self, expr):
        return lambda **kw: float('nan')

    def _print_Mul(self, expr, order=None):
        assert len(expr.args) > 1
        terms = [self._print(a) for a in expr.args]

        def __inner_mul(**kw):
            x = terms[0](**kw)
            for r in terms[1:]:
                x = torch.mul(x, r(**kw))
            return x
        return __inner_mul

    def _print_Rational(self, expr):
        return lambda **kw: expr.p/expr.q

    def _print_Fraction(self, expr):
        numer = self._print(expr.numerator)
        denom = self._print(expr.denominator)
        return lambda **kw: numer(**kw), denom(**kw)

    def _print_Float(self, expr):
        return lambda **kw: float(expr.evalf())

    def _print_Symbol(self, expr):
        d = expr._assumptions.generator
        # print the dummy_index like it was an assumption
        if expr.is_Dummy or d != {}:
            raise NotImplementedError()

        return lambda **kw: kw[expr.name]

    def _print_FracElement(self, frac):
        numer_terms = list(frac.numer.terms())
        assert len(numer_terms) == 1
        #numer_terms.sort(key=frac.field.order, reverse=True)
        denom_terms = list(frac.denom.terms())
        assert len(denom_terms) == 1
        #denom_terms.sort(key=frac.field.order, reverse=True)
        numer = self._print(numer[0])
        denom = self._print(denom[0])
        return lambda *kw: torch.div(numer(*kw), denom(*kw))

    def _print_Expr(self, expr):
        base = self._print(expr.base)
        exp = self._print(expr.exp)
        if isinstance(expr, Pow):
            return lambda **kw: torch.pow(base(**kw), exp(**kw))

        raise NotImplementedError(f"No implementation of Expr {expr}")
        # return lambda **kw: kw[expr.name]

def sympy2torch(expr):
    return TorchPrinter()._print(expr)

def pendulum_gradient(n, lengths=None, masses=1, friction=0.3):
    """Integrate a multi-pendulum with `n` sections"""
    #-------------------------------------------------
    # Step 1: construct the pendulum model

    # Generalized coordinates and velocities
    # (in this case, angular positions & velocities of each mass)
    q = mechanics.dynamicsymbols('q:{0}'.format(n))
    u = mechanics.dynamicsymbols('u:{0}'.format(n))

    # mass and length
    m = symbols('m:{0}'.format(n))
    l = symbols('l:{0}'.format(n))

    # gravity and time symbols
    g, t = symbols('g,t')

    #--------------------------------------------------
    # Step 2: build the model using Kane's Method

    # Create pivot point reference frame
    A = mechanics.ReferenceFrame('A')
    P = mechanics.Point('P')
    P.set_vel(A, 0)

    # lists to hold particles, forces, and kinetic ODEs
    # for each pendulum in the chain
    particles = []
    forces = []
    kinetic_odes = []

    for i in range(n):
        # Create a reference frame following the i^th mass
        Ai = A.orientnew('A' + str(i), 'Axis', [q[i], A.z])
        Ai.set_ang_vel(A, u[i] * A.z)

        # Create a point in this reference frame
        Pi = P.locatenew('P' + str(i), l[i] * Ai.x)
        Pi.v2pt_theory(P, A, Ai)

        # Create a new particle of mass m[i] at this point
        Pai = mechanics.Particle('Pa' + str(i), Pi, m[i])
        particles.append(Pai)

        # Set forces & compute kinematic ODE
        forces.append((Pi, m[i] * g * A.x))
        # Add damping torque:
        forces.append((Ai, -1 * friction * u[i] * A.z))

        kinetic_odes.append(q[i].diff(t) - u[i])

        P = Pi

    # Generate equations of motion
    KM = mechanics.KanesMethod(A, q_ind=q, u_ind=u,
                               kd_eqs=kinetic_odes)
    fr, fr_star = KM.kanes_equations(particles, forces)

    #-----------------------------------------------------
    # Step 3: numerically evaluate equations and integrate

    # lengths and masses
    if lengths is None:
        lengths = np.ones(n) / n
    lengths = np.broadcast_to(lengths, n)
    masses = np.broadcast_to(masses, n)

    # Fixed parameters: gravitational constant, lengths, and masses
    parameters = [g] + list(l) + list(m)
    parameter_vals = [9.81] + list(lengths) + list(masses)

    # define symbols for unknown parameters
    unknowns = [Dummy() for i in q + u]
    unknown_dict = dict(zip(q + u, unknowns))
    kds = KM.kindiffdict()

    # substitute unknown symbols for qdot terms
    mm_sym = KM.mass_matrix_full.subs(kds).subs(unknown_dict)
    fo_sym = KM.forcing_full.subs(kds).subs(unknown_dict)

    # create functions for numerical calculation
    mm_func = lambdify(unknowns + parameters, mm_sym)
    fo_func = lambdify(unknowns + parameters, fo_sym)

    # function which computes the derivatives of parameters
    def gradient(y, *a, **kw):
        squeeze = False
        if len(y.shape) == 1:
            squeeze = True
            y = np.expand_dims(y, 0)
        rv = np.zeros_like(y)

        for i in range(y.shape[0]):
            # Assume in rad, rad/s:
            #y = np.concatenate([np.broadcast_to(initial_positions, n), np.broadcast_to(initial_velocities, n)])

            vals = np.concatenate((y[i,:], parameter_vals))
            sol = np.linalg.solve(mm_func(*vals), fo_func(*vals))
            rv[i,:] = np.array(sol).T[0]

        if squeeze:
            return rv[0,...]
        return rv

    # ODE integration
    return gradient

def _redim(inp):
    vec = np.array(inp)
    # Wrap all dimensions:
    n = vec.shape[1] // 2
    assert vec.shape[1] == n*2

    # Get angular positions:
    pos = vec[:,:n]
    l = 100

    if np.any(pos < -np.pi):
        # In multiples of 2pi
        adj, _ = np.modf((pos[pos < -np.pi] + np.pi) / (2*np.pi))
        # Scale it back
        pos[pos < -np.pi] = (adj * 2*np.pi) + np.pi
        assert not np.any(pos < -np.pi)

    if np.any(pos >= np.pi):
        # In multiples of 2pi
        adj, _ = np.modf((pos[pos >= np.pi] - np.pi) / (2*np.pi))
        # Scale it back
        pos[pos >= np.pi] = (adj * 2*np.pi) - np.pi
        assert not np.any(pos >= np.pi)

    vec[:,:n] = pos
    return vec

def pendulum_energy(n=1, lengths=1, masses=1, include_gpe=True, include_ke=True):
    # Generalized coordinates and velocities
    # (in this case, angular positions & velocities of each mass)
    q = mechanics.dynamicsymbols('q:{0}'.format(n))
    u = mechanics.dynamicsymbols('u:{0}'.format(n))

    # mass and length
    m = symbols('m:{0}'.format(n))
    l = symbols('l:{0}'.format(n))

    # gravity and time symbols
    g, t = symbols('g,t')

    #--------------------------------------------------
    # Step 2: build the model using Kane's Method

    # Create pivot point reference frame
    A = mechanics.ReferenceFrame('A')
    P = mechanics.Point('P')
    Origin = P
    P.set_vel(A, 0)

    gravity_direction = -A.x

    # lists to hold particles, forces, and kinetic ODEs
    # for each pendulum in the chain
    particles = []
    forces = []

    gpe = []
    ke = []

    cartVel = 0.0
    cartPos = 0.0

    for i in range(n):
        # Create a reference frame following the i^th mass
        Ai = A.orientnew('A' + str(i), 'Axis', [q[i], A.z])
        Ai.set_ang_vel(A, u[i] * A.z)

        # Create a point in this reference frame
        Pi = P.locatenew('P' + str(i), l[i] * Ai.x)
        Pi.v2pt_theory(P, A, Ai)

        # Create a new particle of mass m[i] at this point
        Pai = mechanics.Particle('Pa' + str(i), Pi, m[i])
        particles.append(Pai)

        # Calculate the cartesian position and velocity:
        # cartPos += l[i] * q[i]
        pos = Pi.pos_from(Origin)

        ke.append(1/n * Pai.kinetic_energy(A))
        gpe.append(m[i] * g * (Pi.pos_from(Origin) & gravity_direction))

        P = Pi

    # lengths and masses
    if lengths is None:
        lengths = np.ones(n) / n
    lengths = np.broadcast_to(lengths, n)
    masses = np.broadcast_to(masses, n)

    # Fixed parameters: gravitational constant, lengths, and masses
    parameters = [g] + list(l) + list(m)
    parameter_vals = [9.81] + list(lengths) + list(masses)

    # define symbols for unknown parameters
    unknowns = [Dummy() for i in q + u]
    unknown_dict = dict(zip(q + u, unknowns))

    # create functions for numerical calculation
    total_energy = 0
    if include_gpe:
        total_energy += (sum(gpe)).subs(zip(parameters, parameter_vals))
    if include_ke:
        total_energy += (sum( ke)).subs(zip(parameters, parameter_vals))

    total_energy_func = sympy2torch(total_energy)

    minimum_energy = total_energy_func(**fixvalue(n, torch.tensor([[0.]*2*n]))).detach()
    return lambda inp: (total_energy_func(**fixvalue(n, inp)) - minimum_energy.to(inp)).unsqueeze(1)

def fixvalue(n, value):
    keys = [f"q{i}_t" for i in range(n)] + [f"u{i}_t" for i in range(n)]
    rv = {}
    for i in range(2*n):
        if isinstance(value, list):
            rv[keys[i]] = value[i]
        else:
            rv[keys[i]] = value[:,i]
    return rv