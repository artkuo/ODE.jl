# Ordinary Differential Equation Solvers

module ODE

using Polynomial

## minimal function export list
export ode23, ode4, ode45, ode4s, ode4ms, ode78

## complete function export list
#export ode23, ode4,
#    oderkf, ode45, ode45_dp, ode45_fb, ode45_ck,
#    oderosenbrock, ode4s, ode4s_kr, ode4s_s,
#    ode4ms, ode_ms


#ODE23  Solve non-stiff differential equations.
#
#   ODE23(F,TSPAN,Y0) with TSPAN = [T0 TFINAL] integrates the system
#   of differential equations dy/dt = f(t,y) from t = T0 to t = TFINAL.
#   The initial condition is y(T0) = Y0.
#
#   The first argument, F, is a function handle or an anonymous function
#   that defines f(t,y).  This function must have two input arguments,
#   t and y, and must return a column vector of the derivatives, dy/dt.
#
#   With two output arguments, [T,Y] = ODE23(...) returns a column
#   vector T and an array Y where Y(:,k) is the solution at T(k).
#
#   More than four input arguments, ODE23(F,TSPAN,Y0,RTOL,P1,P2,...),
#   are passed on to F, F(T,Y,P1,P2,...).
#
#   ODE23 uses the Runge-Kutta (2,3) method of Bogacki and Shampine (BS23).
#
#   Example
#      tspan = [0, 2*pi]
#      y0 = [1, 0]
#      F = (t, y) -> [0 1; -1 0]*y
#      ode23(F, tspan, y0)
#
#   See also ODE23.

# Initialize variables.
# Adapted from Cleve Moler's textbook
# http://www.mathworks.com/moler/ncm/ode23tx.m
function ode23(F, y0, tspan; reltol = 1.e-5, abstol = 1.e-8)
    if reltol == 0
        warn("setting reltol = 0 gives a step size of zero")
    end

    threshold = abstol / reltol

    t = tspan[1]
    tfinal = tspan[end]
    tdir = sign(tfinal - t)
    hmax = abs(0.1*(tfinal-t))
    y = y0

    tout = t
    yout = Array(typeof(y0),1)
    yout[1] = y

    tlen = length(t)

    # Compute initial step size.

    s1 = F(t, y)
    r = norm(s1./max(abs(y), threshold), Inf) + realmin() # TODO: fix type bug in max()
    h = tdir*0.8*reltol^(1/3)/r

    # The main loop.

    while t != tfinal

        hmin = 16*eps()*abs(t)
        if abs(h) > hmax; h = tdir*hmax; end
        if abs(h) < hmin; h = tdir*hmin; end

        # Stretch the step if t is close to tfinal.

        if 1.1*abs(h) >= abs(tfinal - t)
            h = tfinal - t
        end

        # Attempt a step.

        s2 = F(t+h/2, y+h/2*s1)
        s3 = F(t+3*h/4, y+3*h/4*s2)
        tnew = t + h
        ynew = y + h*(2*s1 + 3*s2 + 4*s3)/9
        s4 = F(tnew, ynew)

        # Estimate the error.

        e = h*(-5*s1 + 6*s2 + 8*s3 - 9*s4)/72
        err = norm(e./max(max(abs(y), abs(ynew)), threshold), Inf) + realmin()

        # Accept the solution if the estimated error is less than the tolerance.

        if err <= reltol
            t = tnew
            y = ynew
            tout = [tout; t]
            push!(yout, y)
            s1 = s4   # Reuse final function value to start new step
        end

        # Compute a new step size.

        h = h*min(5, 0.8*(reltol/err)^(1/3))

        # Exit early if step size is too small.

        if abs(h) <= hmin
            println("Step size ", h, " too small at t = ", t)
            t = tfinal
        end

    end # while (t != tfinal)

    return tout, yout

end # ode23



# ode45 adapted from http://users.powernet.co.uk/kienzle/octave/matcompat/scripts/ode_v1.11/ode45.m
# (a newer version (v1.15) can be found here https://sites.google.com/site/comperem/home/ode_solvers)
#
# ode45 (v1.11) integrates a system of ordinary differential equations using
# 4th & 5th order embedded formulas from Dormand & Prince or Fehlberg.
#
# The Fehlberg 4(5) pair is established and works well, however, the
# Dormand-Prince 4(5) pair minimizes the local truncation error in the
# 5th-order estimate which is what is used to step forward (local extrapolation.)
# Generally it produces more accurate results and costs roughly the same
# computationally.  The Dormand-Prince pair is the default.
#
# This is a 4th-order accurate integrator therefore the local error normally
# expected would be O(h^5).  However, because this particular implementation
# uses the 5th-order estimate for xout (i.e. local extrapolation) moving
# forward with the 5th-order estimate should yield errors on the order of O(h^6).
#
# The order of the RK method is the order of the local *truncation* error, d.
# The local truncation error is defined as the principle error term in the
# portion of the Taylor series expansion that gets dropped.  This portion of
# the Taylor series exapansion is within the group of terms that gets multipled
# by h in the solution definition of the general RK method.  Therefore, the
# order-p solution created by the RK method will be roughly accurate to
# within O(h^(p+1)).  The difference between two different-order solutions is
# the definition of the "local error," l.  This makes the local error, l, as
# large as the error in the lower order method, which is the truncation
# error, d, times h, resulting in O(h^(p+1)).
# Summary:   For an RK method of order-p,
#            - the local truncation error is O(h^p)
#            - the local error (used for stepsize adjustment) is O(h^(p+1))
#
# This requires 6 function evaluations per integration step.
#
# The error estimate formula and slopes are from
# Numerical Methods for Engineers, 2nd Ed., Chappra & Cannle, McGraw-Hill, 1985
#
# Usage:
#         (tout, xout) = ode45(F, tspan, x0)
#
# INPUT:
# F     - User-defined function
#         Call: xprime = F(t,x)
#         t      - Time (scalar).
#         x      - Solution column-vector.
#         xprime - Returned derivative COLUMN-vector; xprime(i) = dx(i)/dt.
# tspan - [ tstart, tfinal ]
# x0    - Initial value COLUMN-vector.
#
# OUTPUT:
# tout  - Returned integration time points (column-vector).
# xout  - Returned solution, one solution column-vector per tout-value.
#
# Original Octave implementation:
# Marc Compere
# CompereM@asme.org
# created : 06 October 1999
# modified: 17 January 2001
function oderkf(F, x0, tspan, p, a, bs, bp; reltol = 1.0e-5, abstol = 1.0e-8,
  eventfun = None, eventargs = None nonnegativeids = [])
    # see p.91 in the Ascher & Petzold reference for more infomation.
    # p = polynomial order, a = matrix of coefficients, bs = b coefficients low,
    # bp = b coefficients high
    pow = 1/p   # use the higher order to estimate the next step size

    c = sum(a, 2)   # consistency condition

    # Initialization
    t = tspan[1]
    tfinal = tspan[end]
    tdir = sign(tfinal - t)
    hmax = abs(tfinal - t)/2.5
    hmin = abs(tfinal - t)/1e9
    h = tdir*abs(tfinal - t)/100  # initial guess at a step size
    x = x0
    tout = t            # first output time
    xout = Array(typeof(x0), 1)
    xout[1] = x         # first output solution

    k = Array(typeof(x0), length(c))
    k[1] = F(t,x) # first stage

    while abs(t) != abs(tfinal) && abs(h) >= hmin  # MAIN LOOP
        if abs(h) > abs(tfinal-t) # step takes us past tfinal
            h = tfinal - t        # adjust it so it takes us there exactly
        end

        #(p-1)th and pth order estimates
        xs = x + h*bs[1]*k[1]
        xp = x + h*bp[1]*k[1]
        for j = 2:length(c)
            dx = a[j,1]*k[1]
            for i = 2:j-1
                dx += a[j,i]*k[i]
            end
            k[j] = F(t + h*c[j], x + h*dx)

            # compute the (p-1)th order estimate
            xs = xs + h*bs[j]*k[j]
            # compute the pth order estimate
            xp = xp + h*bp[j]*k[j]
        end

        # estimate the local truncation error
        gamma1 = xs - xp

        # Estimate the error and the acceptable error
        delta = norm(gamma1, Inf)              # actual error
        tau   = max(reltol*norm(x,Inf),abstol) # allowable error

        # Update the solution only if the error is acceptable
        if delta <= tau
            tnew = t + h
            xnew = xp    # <-- using the higher order estimate is called 'local extrapolation'

            # Check events to see if there's a zero-crossing in this time step
            if haveevents
              (te, xe, ie, valt, stop) = odeevent(ntrp45, eventfun, eventargs,
                valt, t, x, tnew, xnew, t0, h, k, nonnegativeids)
              if !isempty(te) # event detected, so zero in on it
                push!(teout, te)
                push!(xeout, xe)
                push!(ieout, ie)
                if stop # event isterminal, so interpolate to te
                  taux = t + (te[end] - t)*c # use interpolating polynomial
                  # between now and the event time
                  (_, k[:,2:7]) =
                    ntrp45(taux, t, x, None, None, h, k, nonnegativeids)
                  tnew = te[end]
                  xnew = xe[:,end]
                  h = tnew - t
                  done = true
                end

              end # event detected

              #function odeevent(ntrpfun::Function, eventfun::Function, eventargs,
              #  v, t, x, tnew, xnew, t0, args...)
              #function ntrp45(tinterp,t,y,tnew,ynew,h,k; nonnegativeids::Array=[], derivs=false)

            # if haveEventFcn
            #   [te,ye,ie,valt,stop] = ...
            #       odezero(@ntrp45,eventFcn,eventArgs,valt,t,y,tnew,ynew,t0,h,f,idxNonNegative);
            #   if ~isempty(te)
            #     if output_sol || (nargout > 2)
            #       teout = [teout, te];
            #       yeout = [yeout, ye];
            #       ieout = [ieout, ie];
            #     end
            #     if stop               % Stop on a terminal event.
            #       % Adjust the interpolation data to [t te(end)].
            #
            #       % Update the derivatives using the interpolating polynomial.
            #       taux = t + (te(end) - t)*A;
            #       [~,f(:,2:7)] = ntrp45(taux,t,y,[],[],h,f,idxNonNegative);
            #
            #       tnew = te(end);
            #       ynew = ye(:,end);
            #       h = tnew - t;
            #       done = true;
            #     end
            #   end
            # end

            t = tnew
            x = xnew
            tout = [tout; t]
            push!(xout, x)

            # Compute the slopes by computing the k[:,j+1]'th column based on the previous k[:,1:j] columns
            # notes: k needs to end up as an Nxs, a is 7x6, which is s by (s-1),
            #        s is the number of intermediate RK stages on [t (t+h)] (Dormand-Prince has s=7 stages)
            if c[end] == 1
                # Assign the last stage for x(k) as the first stage for computing x[k+1].
                # This is part of the Dormand-Prince pair caveat.
                # k[:,7] has already been computed, so use it instead of recomputing it
                # again as k[:,1] during the next step.
                k[1] = k[end]
            else
                k[1] = F(t,x) # first stage
            end
        end
        # Update the step size
        h = min(hmax, 0.8*h*(tau/delta)^pow)
    end # while (t < tfinal) & (h >= hmin)       MAIN LOOP
    println(k)
    println(typeof(k))

    if abs(t) < abs(tfinal)
      println("Step size grew too small. t=", t, ", h=", abs(h), ", x=", x)
    end

    return tout, xout
end # oderkf

# Bogacki–Shampine coefficients
const bs_coefficients = (3,
                         [    0           0      0      0
                              1/2         0      0      0
                              0         3/4      0      0
                              2/9       1/3     4/9     0],
                         # 2nd order b-coefficients
                         [7/24 1/4 1/3 1/8],
                         # 3rd order b-coefficients
                         [2/9 1/3 4/9 0],
                         )
ode23_bs(F, x0, tspan; kwargs...) = oderkf(F, x0, tspan, bs_coefficients...; kwargs...)


# Both the Dormand-Prince and Fehlberg 4(5) coefficients are from a tableau in
# U.M. Ascher, L.R. Petzold, Computer Methods for  Ordinary Differential Equations
# and Differential-Agebraic Equations, Society for Industrial and Applied Mathematics
# (SIAM), Philadelphia, 1998
#
# Dormand-Prince coefficients
const dp_coefficients = (5,
                         [    0           0          0         0         0        0
                              1/5         0          0         0         0        0
                              3/40        9/40       0         0         0        0
                             44/45      -56/15      32/9       0         0        0
                          19372/6561 -25360/2187 64448/6561 -212/729     0        0
                           9017/3168   -355/33   46732/5247   49/176 -5103/18656  0
                             35/384       0        500/1113  125/192 -2187/6784  11/84],
                         # 4th order b-coefficients
                         [5179/57600 0 7571/16695 393/640 -92097/339200 187/2100 1/40],
                         # 5th order b-coefficients
                         [35/384 0 500/1113 125/192 -2187/6784 11/84 0],
                         )
ode45_dp(F, x0, tspan; kwargs...) = oderkf(F, x0, tspan, dp_coefficients...; kwargs...)

# Fehlberg coefficients
const fb_coefficients = (5,
                         [    0         0          0         0        0
                             1/4        0          0         0        0
                             3/32       9/32       0         0        0
                          1932/2197 -7200/2197  7296/2197    0        0
                           439/216     -8       3680/513  -845/4104   0
                            -8/27       2      -3544/2565 1859/4104 -11/40],
                         # 4th order b-coefficients
                         [25/216 0 1408/2565 2197/4104 -1/5 0],
                         # 5th order b-coefficients
                         [16/135 0 6656/12825 28561/56430 -9/50 2/55],
                         )
ode45_fb(F, x0, tspan; kwargs...) = oderkf(F, x0, tspan, fb_coefficients...; kwargs...)

# Cash-Karp coefficients
# Numerical Recipes in Fortran 77
const ck_coefficients = (5,
                         [   0         0       0           0          0
                             1/5       0       0           0          0
                             3/40      9/40    0           0          0
                             3/10     -9/10    6/5         0          0
                           -11/54      5/2   -70/27       35/27       0
                          1631/55296 175/512 575/13824 44275/110592 253/4096],
                         # 4th order b-coefficients
                         [37/378 0 250/621 125/594 0 512/1771],
                         # 5th order b-coefficients
                         [2825/27648 0 18575/48384 13525/55296 277/14336 1/4],
                         )
ode45_ck(F, x0, tspan; kwargs...) = oderkf(F, x0, tspan, ck_coefficients...; kwargs...)


# Fehlberg 7(8) coefficients
# Values from pag. 65, Fehlberg, Erwin. "Classical fifth-, sixth-, seventh-, and eighth-order Runge-Kutta formulas with stepsize control".
# National Aeronautics and Space Administration.
const fb_coefficients_78 = (8,
                            [     0      0      0       0        0         0       0       0     0      0    0 0
                                  2/27   0      0       0        0         0       0       0     0      0    0 0
                                  1/36   1/12   0       0        0         0       0       0     0      0    0 0
                                  1/24   0      1/8     0        0         0       0       0     0      0    0 0
                                  5/12   0    -25/16   25/16     0         0       0       0     0      0    0 0
                                  1/20   0      0       1/4      1/5       0       0       0     0      0    0 0
                                -25/108  0      0     125/108  -65/27    125/54    0       0     0      0    0 0
                                 31/300  0      0       0       61/225    -2/9    13/900   0     0      0    0 0
                                  2      0      0     -53/6    704/45   -107/9    67/90    3     0      0    0 0
                                -91/108  0      0      23/108 -976/135   311/54  -19/60   17/6  -1/12   0    0 0
                               2383/4100 0      0    -341/164 4496/1025 -301/82 2133/4100 45/82 45/164 18/41 0 0
                                  3/205  0      0       0        0        -6/41   -3/205  -3/41  3/41   6/41 0 0
                              -1777/4100 0      0    -341/164 4496/1025 -289/82 2193/4100 51/82 33/164 12/41 0 1],
                            # 7th order b-coefficients
                            [41/840 0 0 0 0 34/105 9/35 9/35 9/280 9/280 41/840 0 0],
                            # 8th order b-coefficients
                            [0 0 0 0 0 34/105 9/35 9/35 9/280 9/280 0 41/840 41/840],
                            )
ode78_fb(F, x0, tspan; kwargs...) = oderkf(F, x0, tspan, fb_coefficients_78...; kwargs...)

# Use Fehlberg version of ode78 by default
const ode78 = ode78_fb

# Use Dormand Prince version of ode45 by default
const ode45 = ode45_dp

# some higher-order embedded methods can be found in:
# P.J. Prince and J.R.Dormand: High order embedded Runge-Kutta formulae, Journal of Computational and Applied Mathematics 7(1), 1981.


#ODE4  Solve non-stiff differential equations, fourth order
#   fixed-step Runge-Kutta method.
#
#   [T,X] = ODE4(ODEFUN, TSPAN, X0) with TSPAN = [T0:H:TFINAL]
#   integrates the system of differential equations x' = f(t,x) from time
#   T0 to TFINAL in steps of H with initial conditions X0. Function
#   ODEFUN(T,X) must return a column vector corresponding to f(t,x). Each
#   row in the solution array X corresponds to a time returned in the
#   column vector T.
function ode4(F, x0, tspan)
    h = diff(tspan)
    x = Array(typeof(x0), length(tspan))
    x[1] = x0

    midxdot = Array(typeof(x0), 4)
    for i = 1:length(tspan)-1
        # Compute midstep derivatives
        midxdot[1] = F(tspan[i],         x[i])
        midxdot[2] = 2*F(tspan[i]+h[i]./2, x[i] + midxdot[1].*h[i]./2)
        midxdot[3] = 2*F(tspan[i]+h[i]./2, x[i] + midxdot[2].*h[i]./2)
        midxdot[4] = F(tspan[i]+h[i],    x[i] + midxdot[3].*h[i])

        # Integrate
        x[i+1] = x[i] + 1/6 .*h[i].*sum(midxdot)
    end
    return [tspan], x
end

#ODEROSENBROCK Solve stiff differential equations, Rosenbrock method
#    with provided coefficients.
function oderosenbrock(F, x0, tspan, gamma, a, b, c; jacobian=nothing)
    # Crude forward finite differences estimator as fallback
    # FIXME: This doesn't really work if x is anything but a Vector or a scalar
    function fdjacobian(F, x::Number, t)
        ftx = F(t, x)

        # The 100 below is heuristic
        dx = (x .+ (x==0))./100
        dFdx = (F(t,x+dx)-ftx)./dx

        return dFdx
    end

    function fdjacobian(F, x, t)
        ftx = F(t, x)
        lx = max(length(x),1)
        dFdx = zeros(eltype(x), lx, lx)
        for j = 1:lx
            # The 100 below is heuristic
            dx = zeros(eltype(x), lx)
            dx[j] = (x[j] .+ (x[j]==0))./100
            dFdx[:,j] = (F(t,x+dx)-ftx)./dx[j]
        end
        return dFdx
    end

    if typeof(jacobian) == Function
        G = jacobian
    else
        G = (t, x)->fdjacobian(F, x, t)
    end

    h = diff(tspan)
    x = Array(typeof(x0), length(tspan))
    x[1] = x0

    solstep = 1
    while tspan[solstep] < maximum(tspan)
        ts = tspan[solstep]
        hs = h[solstep]
        xs = x[solstep]
        dFdx = G(ts, xs)
        # FIXME
        if size(dFdx,1) == 1
            jac = 1/gamma/hs - dFdx[1]
        else
            jac = eye(dFdx)/gamma/hs - dFdx
        end

        g = Array(typeof(x0), size(a,1))
        g[1] = (jac \ F(ts + b[1]*hs, xs))
        x[solstep+1] = x[solstep] + b[1]*g[1]

        for i = 2:size(a,1)
            dx = zero(x0)
            dF = zero(x0/hs)
            for j = 1:i-1
                dx += a[i,j]*g[j]
                dF += c[i,j]*g[j]
            end
            g[i] = (jac \ (F(ts + b[i]*hs, xs + dx) + dF/hs))
            x[solstep+1] += b[i]*g[i]
        end
        solstep += 1
    end
    return [tspan], x
end


# Kaps-Rentrop coefficients
const kr4_coefficients = (0.231,
                          [0              0             0 0
                           2              0             0 0
                           4.452470820736 4.16352878860 0 0
                           4.452470820736 4.16352878860 0 0],
                          [3.95750374663  4.62489238836 0.617477263873 1.28261294568],
                          [ 0               0                0        0
                           -5.07167533877   0                0        0
                            6.02015272865   0.1597500684673  0        0
                           -1.856343618677 -8.50538085819   -2.08407513602 0],)

ode4s_kr(F, x0, tspan; jacobian=nothing) = oderosenbrock(F, x0, tspan, kr4_coefficients...; jacobian=jacobian)

# Shampine coefficients
const s4_coefficients = (0.5,
                         [ 0    0    0 0
                           2    0    0 0
                          48/25 6/25 0 0
                          48/25 6/25 0 0],
                         [19/9 1/2 25/108 125/108],
                         [   0       0      0   0
                            -8       0      0   0
                           372/25   12/5    0   0
                          -112/125 -54/125 -2/5 0],)

ode4s_s(F, x0, tspan; jacobian=nothing) = oderosenbrock(F, x0, tspan, s4_coefficients...; jacobian=jacobian)

# Use Shampine coefficients by default (matching Numerical Recipes)
const ode4s = ode4s_s

# ODE_MS Fixed-step, fixed-order multi-step numerical method with Adams-Bashforth-Moulton coefficients
function ode_ms(F, x0, tspan, order::Integer)
    h = diff(tspan)
    x = Array(typeof(x0), length(tspan))
    x[1] = x0

    if 1 <= order <= 4
        b = [ 1      0      0     0
             -1/2    3/2    0     0
             5/12  -16/12  23/12 0
             -9/24   37/24 -59/24 55/24]
    else
        for steporder = size(b,1):order
            s = steporder - 1
            for j = 0:s
                # Assign in correct order for multiplication below
                #                    (a factor depending on j and s)      .* (an integral of a polynomial with -(0:s), except -j, as roots)
                b[steporder,s-j+1] = (-1).^j./factorial[j]./factorial(s-j).*diff(polyval(polyint(poly(diagm(-[0:j-1; j+1:s]))),0:1))
            end
        end
    end

    # TODO: use a better data structure here (should be an order-element circ buffer)
    xdot = similar(x)
    for i = 1:length(tspan)-1
        # Need to run the first several steps at reduced order
        steporder = min(i, order)
        xdot[i] = F(tspan[i], x[i])

        x[i+1] = x[i]
        for j = 1:steporder
            x[i+1] += h[i]*b[steporder, j]*xdot[i-(steporder-1) + (j-1)]
        end
    end
    return [tspan], x
end

# Use order 4 by default
ode4ms(F, x0, tspan) = ode_ms(F, x0, tspan, 4)

function mycumprod(x)
    y = copy(x)
    for i = 2:length(syx)
        y[i] = y[i-1] .* x[i]
    end
    return y
end

#function ntrp45(tinterp,t,y::Array,tnew,ynew::Array,h,k::Array; nonnegativeids::Array=[], derivs=false)
function ntrp45(tinterp,t,y,tnew,ynew,h,k; nonnegativeids::Array=[], derivs=false)
#NTRP45  Interpolation helper function for ODE45.
#   YINTERP = NTRP45(TINTERP,T,Y,TNEW,YNEW,H,F,IDX) uses data computed in ODE45
#   to approximate the solution at time TINTERP.  TINTERP may be a scalar
#   or a row vector.
#   The arguments TNEW and YNEW do not affect the computations. They are
#   required for consistency of syntax with other interpolation functions.
#   Any values entered for TNEW and YNEW are ignored.
#   t = current time, y = current y
#   h = current step size
#   F = matrix of values (N x 7) where N is order of system
#   similar to k = array of arrays (7 x N)
#
#   [YINTERP,YPINTERP] = NTRP45(TINTERP,T,Y,TNEW,YNEW,H,F,IDX) returns also the
#   derivative of the polynomial approximating the solution.
#
#   IDX has indices of solution components that must be non-negative. Negative
#   YINTERP(IDX) are replaced with zeros and the derivative YPINTERP(IDX) is
#   set to zero.
#
#   See also ODE45, DEVAL.

#   Mark W. Reichelt and Lawrence F. Shampine, 6-13-94
#   Copyright 1984-2009 The MathWorks, Inc.
#   $Revision: 1.13.4.7 $  $Date: 2009/11/16 22:26:19 $

# from Jimenez JC, Sotolongo A, Sanchez-Bornot JM (2014)
# Localy linearized Runge Kutta method of Dormand and Prince
# Applied Mathematics & Computation 247: 589-606, Table 2
#
const  BI = [
      1       -183/64      37/12       -145/128
      0          0           0            0
      0       1500/371    -1000/159    1000/371
      0       -125/32       125/12     -375/64
      0       9477/3392   -729/106    25515/6784
      0        -11/7        11/3        -55/28
      0         3/2         -4            5/2
      ]

  Ntimes = length(tinterp) # may be fed a scalar time or an array of times

  s = (tinterp' - t)/h  # row of interpolation time pts
  # normalized by h for numerical precision because
  # we're going to take this to the fourth power

# y + h * sum(b_j(tau) k_j)
  cinterp = h*BI*cumprod([s;s;s;s]) # interpolation coefficients

  yinterp = [y + cinterp[:,i]'*k for i=1:Ntimes]

  cpinterp = BI*[ones(1,Ntimes); cumprod([2*s;3/2*s;4/3*s])]

  if derivs # nargout > 1
    ypinterp = [0*y + cpinterp[:,i]'*k for i=1:Ntimes]
    # the 0*y is needed to ensure the type is an array
  end

# Zero out non-negative solutions
  for i in nonnegativeids
    for j = 1:Ntimes
      if yinterp[j][i] < 0
        yinterp[j][i] = 0
        if getderiv
          ypinterp[j][i] = 0
        end
      end
    end # loop over interpolated times
  end # loop over non-negative indices

  if derivs
    return yinterp,ypinterp
  else
    return yinterp
  end # getderiv

end # ntrp45


# function [tout,yout,iout,vnew,stop] = ...
#     odezero(ntrpfun,eventfun,eventargs,v,t,y,tnew,ynew,t0,varargin)
function odeevent(ntrpfun::Function, eventfun::Function, eventargs,
  v, t, x, tnew, xnew, t0, args...)
# ODEEVENT Locate any zero-crossings of event functions in a time step.
#   ODEEVENT is an event location helper function for the ODE Suite.  ODEEVENT
#   uses Regula Falsi and information passed from the ODE solver to locate
#   any zeros in the half open time interval (T,TNEW] of the event functions
#   coded in eventfun.

# improve zhang's http://www.cscjournals.org/csc/manuscript/Journals/IJEA/volume4/Issue1/IJEA-33.pdf
# num recipes code: http://apps.nrbook.com/empanel/index.html#pg=454
# wikipedia http://en.wikipedia.org/wiki/Brent%27s_method
# wikipedia http://en.wikipedia.org/wiki/Ridders%27_method (sq rt)


# Initialize
  tol = 128*max(eps(t),eps(tnew))
  tol = min(tol, abs(tnew - t))
  tout = []
  xout = []
  iout = []
  tdir = sign(tnew - t) # are we integrating fwd or bwd in time
  stop = 0
  rmin = realmin(v)

  # Set up tL, tR, xL, xR, vL, vR, isterminal and direction.
  tL, xL, vL = t, x, v
  (vnew,isterminal,direction) = feval(eventfun,tnew,xnew,eventargs...)
  if isempty(direction)
    direction = zeros(vnew)   # zeros crossings in any direction
  end
  tR, xR, vR = tnew, xnew, vnew

  # Initialize ttry so that we won't extrapolate if vL or vR is zero.
  ttry = tR

  # Find all events before tnew or the first terminal event.
  while true # Outer while

    lastmoved = 0
    while true     # Inner while: Bracket in until we reach tolerance
      #      detect   zero-crossing             in the intended direction
      indzc = find((sign(vL) != sign(vR)) && (direction .* (vR - vL) >= 0))
      # Events of interest shouldn't have disappeared, but new ones might
      # be found in other elements of the v vector:
      if isempty(indzc)    # nothing in bracket
        if lastmoved != 0  # but we had one that we were trying to close in on
          error("ODEEVENT: lost event") # which evidently disappeared
        end
        # seems like there are no events so go home
        return tout,xout,iout,vnew,stop
      end

      # Check if the time interval is too short to continue looking.
      bracketwidtht = tR - tL
      if abs(bracketwidtht) <= tol
        break # give up, only small steps to perform here
      end

      # are we sitting on an event (at left bracket, with non-event at right?)
      if (tL == t) && any(vL[indzc] == 0. & vR[indzc] != 0.)
        ttry = tL + tdir*0.5*tol # then try to move forward a tiny bit

      else # try to bracket inward
        # Compute Regula Falsi change, using leftmost possibility.
        change = 1
        for j in indzc
          # If either bracket is stuck on zero, try using old bracket to guess
          # a crossing or departure from zero.
          if vL[j] == 0.0 # left bracket stuck on zero?
            # did we previously move the right bracket in?
            if (tdir*ttry > tdir*tR) && (vtry[j] != vR[j])
              # then use the two rights (vtry, vR) to get a secant solution
              # to serve as the next guess
              maybe = 1.0 - vR[j] * (ttry-tR) / ((vtry[j]-vR[j]) * bracketwidtht)
              if (maybe < 0) || (maybe > 1) # avoid going outside bracket
                 maybe = 0.5 # so conservatively fall back on bisection
              end
            else # also fall back on bisection we had either moved the left
              maybe = 0.5 # bracket or if no deriv to be gained from right
            end
          elseif vR[j] == 0.0 # stuck on zero at vR
            if (tdir*ttry < tdir*tL) && (vtry[j] != vL[j])
              maybe = vL[j] * (tL-ttry) / ((vtry[j]-vL[j]) * bracketwidtht)
              if (maybe < 0) || (maybe > 1)
                maybe = 0.5
              end
            else
              maybe = 0.5
            end

          else # both vL and vR are non-zero, so linearly interpolate the root
            maybe = -vL[j] / (vR[j] - vL[j]) # Note vR(j) != vL(j)
          end

          if maybe < change # accept as a change as long as it's interpolation
            change = maybe  # fraction of the bracket where zero is expected
          end
        end # loop through event indices
        changet = change * abs(bracketwidtht) # convert into the time step

        # Enforce minimum and maximum change in time interval.
        # At least a smidge over zero (0.5*tol) and at most (a smidge under)
        # bracketwidtht
        changet = max(0.5*tol, min(changet, abs(bracketwidtht) - 0.5*tol))

        ttry = tL + tdir * changet
      end # if need to change

      # Compute vtry.
      xtry = feval(ntrpfun,ttry,t,x,tnew,xnew,args...)
      vtry = feval(eventfun,ttry,xtry,eventargs...)

      # Check for any crossings between tL and ttry.
      indzc = find((sign(vL) != sign(vtry)) & (direction .* (vtry - vL) >= 0))
      if !isempty(indzc) # root appears to be to the left of ttry
        #  Move right end of bracket leftward, remembering the old value in try
        tR, ttry = ttry, tR
        xR, xtry = xtry, xR
        vR, vtry = vtry, vR
        # Illinois method.  If we've moved leftward twice, halve
        # vL so we'll move closer next time.
        if lastmoved == 2
          # Watch out for underflow and signs disappearing.
          maybe = 0.5 * vL
          i = find(abs(maybe) >= rmin)
          vL[i] = maybe[i] # accept adjustments that are big enough
        end
        lastmoved = 2  # meaning we just moved the right bracket left
      else # root must be to the right
        # Move left end of bracket rightward, remembering the old value.
        tL, ttry = ttry, tL
        xL, xtry = xtry, xL
        vL, vtry = vtry, vL
        # Illinois method.  If we've moved rightward twice, halve
        # vR so we'll move closer next time.
        if lastmoved == 1 # already moved left bracket right
          # Watch out for underflow and signs disappearing.
          maybe = 0.5 * vR
          i = find(abs(maybe) >= rmin)
          vR[i] = maybe[i]
        end
        lastmoved = 1 # record that we moved the left bracket right
      end # crossings between tL and ttry
    end # inner while, got close

    # great, now let's record the event at tR
    j = ones(1,length(indzc))
    push!(tout, fill(tR, length(indzc))    # tout = [tout, tR(j)]
    push!(xout, fill(xR, [:,j])  # yout = [yout, yR(:,j)]
    push!(iout, indzc)   # iout = [iout, indzc']
    if any(isterminal[indzc]) # isterminal means we need to quit
      if tL != t0 # have we gone anywhere yet? (only stop if event is past i.c.)
        stop = 1  # this signals ode to stop
      end
      break # out of outer while to quit
      # and otherwise we need to keep going
    elseif abs(tnew - tR) <= tol # and can do so if we're close enough
      #  We're not going to find events closer than tol.
      break # out of outer while
    else
      # Shift whole bracket rightward from [tL tR] to [tR+0.5*tol tnew].
      ttry, xtry, vtry = tR, xR, vR
      tL = tR + tdir*0.5*tol # just a smidge past our most recent event
      xL = feval(ntrpfun,tL,t,x,tnew,ynew,args...) # right-hand-side approx
      vL = feval(eventfun,tL,xL,eventargs...)      # event value
      tR, xR, vR = tnew, xR, vR
    end # okay our event was before tnew, so keep going

  end # outer while, look for next event

  # one way out
  return tout,xout,iout,vnew,stop
  end # function odeevent

end # module ODE

# Doub zbrent(T &func, const Doub x1, const Doub x2, const Doub tol)
# {
#   const Int ITMAX=100;
#   const Doub EPS=numeric_limits<Doub>::epsilon();
#   Doub a=x1,b=x2,c=x2,d,e,fa=func(a),fb=func(b),fc,p,q,r,s,tol1,xm;
#   if ((fa > 0.0 && fb > 0.0) || (fa < 0.0 && fb < 0.0))
#     throw("Root must be bracketed in zbrent");
#   fc=fb;
#   for (Int iter=0;iter<ITMAX;iter++) {
#     if ((fb > 0.0 && fc > 0.0) || (fb < 0.0 && fc < 0.0)) {
#       c=a;
#       fc=fa;
#       e=d=b-a;
#     }
#     if (abs(fc) < abs(fb)) {
#       a=b;
#       b=c;
#       c=a;
#       fa=fb;
#       fb=fc;
#       fc=fa;
#     }
#     tol1=2.0*EPS*abs(b)+0.5*tol;
#     xm=0.5*(c-b);
#     if (abs(xm) <= tol1 || fb == 0.0) return b;
#     if (abs(e) >= tol1 && abs(fa) > abs(fb)) {
#       s=fb/fa;
#       if (a == c) {
#         p=2.0*xm*s;
#         q=1.0-s;
#       } else {
#         q=fa/fc;
#         r=fb/fc;
#         p=s*(2.0*xm*q*(q-r)-(b-a)*(r-1.0));
#         q=(q-1.0)*(r-1.0)*(s-1.0);
#       }
#       if (p > 0.0) q = -q;
#       p=abs(p);
#       Doub min1=3.0*xm*q-abs(tol1*q);
#       Doub min2=abs(e*q);
#       if (2.0*p < (min1 < min2 ? min1 : min2)) {
#         e=d;
#         d=p/q;
#       } else {
#         d=xm;
#         e=d;
#       }
#     } else {
#       d=xm;
#       e=d;
#     }
#     a=b;
#     fa=fb;
#     if (abs(d) > tol1)
#       b += d;
#     else
#       b += SIGN(tol1,xm);
#       fb=func(b);
#   }
#   throw("Maximum number of iterations exceeded in zbrent");
# }
