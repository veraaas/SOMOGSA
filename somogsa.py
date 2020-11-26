"""
TBD
"""


from math import cos, degrees, pi
from cma import bbobbenchmarks as bn
import matplotlib.pyplot as plt
import numpy as np
import random, xlsxwriter


bbob_fun = bn.F22(1)  # single-objective two-dimensional bbob problem


def rastrOneD(x):
    # Single-Objective Rastrigin Problem
    y = 20 + (pow(x[0], 2)-10*cos(2*pi*x[0])) + (pow(x[1], 2)-10*cos(2*pi*x[1]))
    return y


def bBOP(x):
   # Single-Objective bbob Problem
   return bbob_fun(x)


def dimSphere(x, shift):
   return sum([pow(x[i] - shift[i], 2) for i in range(len(x))])


def multiobjectivisation(x, func, shift):
   """Transforming a single-objective into a bi-objective problem by adding a sphere function"""
   y = np.empty(2)
   y[0] = func(x)
   y[1] = dimSphere(x, shift)
   return y


def init_random(bounds, dim):
    """Initializing a random point within the boundaries in the respective dimension"""
    v = []
    for i in range(dim):
        v.append(bounds[i, 0] + random.random()*(bounds[i, 1]-bounds[i, 0]))
    return np.array(v)


def angle(u, v):
    skal_u = np.sqrt((u * u).sum())
    skal_v = np.sqrt((v * v).sum())

    cos = np.dot(u, v) / (skal_u * skal_v)

    # check for rounding errors
    if cos > 1:
        cos = 1
    elif cos < -1:
        cos = -1

    arc = degrees(np.arccos(cos))  # angle between u and v
    return arc


def checkbounds(bounds, x, x_new, g):
    """if x is out of bounds, place it on the point of the boundary where it was crossed"""
    d = len(x)
    restarted = False

    # lower bounds crossed?
    stepMax = 100
    crossed = False
    for i in range(d):
        if x_new[i] < bounds[i, 0]:
            crossed = True
            step = (x[i] - bounds[i, 0]) / abs(g[i])
            if step < stepMax:
                stepMax = step
                stepPos = i
    if crossed:
        x_new = x - g / abs(g[stepPos]) * (x[stepPos] - bounds[stepPos, 0])

    # upper bounds crossed?
    stepMax = 100
    crossed = False
    for i in range(d):
        if x_new[i] > bounds[i, 1]:
            crossed = True
            step = (bounds[i, 1] - x[i]) / abs(g[i])
            if step < stepMax:
                stepMax = step
                stepPos = i
    if crossed:
        x_new = x - g / abs(g[stepPos]) * (bounds[stepPos, 1] - x[stepPos])

    # trapped at bounds? restart randomly
    if np.array_equal(x_new, x):
        restarted = True
        x_new = init_random(bounds, d)

    return x_new, restarted


def norm(x):
    """compute the norm/length of a given vector as well as the unit vector"""

    vlength = np.linalg.norm(x)

    if vlength > 0:
        uv = x/vlength # [q/vlength for q in x]
    else:
        uv = np.array([0.0]*len(x)) # nullarray
    return (vlength, uv)


def gradient(x, problem, shift, h=0.0001, singleObj = False):
    '''The bi-objective gradient of the problem at point x is estimated'''

    grads = [[],[]]

    for i in range(len(x)):
        xp = x.copy() + [0.0, 0.0]
        xm = x.copy() + [0.0, 0.0]
        xp[i] = xp[i] + h
        xm[i] = xm[i] - h

        if singleObj:
            fp = multiobjectivisation(xp, problem, shift)
            fm = multiobjectivisation(xm, problem, shift)
        else:
            fp = problem(xp)
            fm = problem(xm)
        gradient = (fp-fm)/(2*h)

        grads[0].append(-gradient[0])  # stores all (negated) gradients of the first function [[x1, x2], [x1, x2]]
        grads[1].append(-gradient[1])  # stores all (negated) gradients of the second function

    return np.array(grads)


def interv_bisection(problem, x_tm, mog_tm, x_t, mog_t, SO):

    count = 0
    found = False

    while count < 100:
        count += 1

        x_tp = x_tm + (x_t - x_tm) * np.linalg.norm(mog_tm) / (np.linalg.norm(mog_t) + np.linalg.norm(mog_tm))

        gs = gradient(x_tp, problem, shift, singleObj=SO)  # approximation of MOP's SO gradients
        g1 = norm(gs[0]) # [length, unit vector]
        g2 = norm(gs[1])
        mog_tp = g1[1] + g2[1]

        if np.linalg.norm(mog_tp) <= 0.000001: # point on efficient set?
            found = True
            break

        w = angle(mog_tm, mog_tp)

        if w < 90: #  x_tm and x_tp are pointing in the same direction meaning they are on the same side of the efficient set
            x_tm = x_tp
            mog_tm = mog_tp
        else: #  x_tm and x_tp are located on opposing sides of the efficient set
            x_t = x_tp
            mog_t = mog_tp

    return x_tp, mog_tp, found


def MOGSA(start, problem, bounds, step_mo=0.001, ang_term=170, maxSteps=1000, precision=0.000001, singleObj = False):

    x = start
    path = [[x[0]],[x[1]]]

    stepcount = 0
    l_mog = 1  # length of the MO gradient, initialization with 1
    ang = 1  # angle between g1 an g2, initialization with 1

    success = False

    x_old = np.nan
    mog_old = np.nan

    while l_mog > precision and ang <= ang_term and stepcount < maxSteps:

        stepcount += 1

        gs = gradient(x, problem, shift=shift, singleObj=singleObj)  # approximation of MOP's SO gradients; gs[0] and gs[1] are the single gradients
        g1 = norm(gs[0])  # norm returns length & unit vector of first functions' gradient
        g2 = norm(gs[1])  # norm returns length/ unit vector of second functions' gradient
        mog = g1[1] + g2[1]  # MO gradient at the current position (summed up normalized SO gradients (unit vectors))

        l_mog = np.linalg.norm(mog)  # length of the MO gradient

        if l_mog <= precision:
            success = True  # locally efficient point is found
            break  # downhill search stops

        g_mog = mog * step_mo
        x_new = x + g_mog  # perform gradient-descent step using the gradient length scaled by step_mo

        x_new, restarted = checkbounds(bounds, x, x_new, g_mog)

        # perform an interval bisection if MOGSA jumped over an efficient set
        if stepcount > 1:
            w = angle(mog, mog_old)
            if w > 90:
                x, mog, found = interv_bisection(problem=problem, x_tm=x_old, mog_tm=mog_old, x_t=x, mog_t=mog, SO=singleObj)
                if found:
                    path[0].append(x[0])
                    path[1].append(x[1])
                    break

        x_old = x
        mog_old = mog
        x = x_new

        # store gradient path
        path[0].append(x[0])
        path[1].append(x[1])

    return success, x, path


def localsearch(x, problem, shift, singleObj=False, mu=0.01):
    """problems' gradient is required"""

    grad = gradient(x, problem, singleObj=singleObj, shift=shift)

    g1 = grad[0]  # gradient of f_1
    g1_l = norm(g1)  # length of g1

    if g1_l[0] == 0:
        print("gradient of f1 zero - breaking.")
        return x

    g1_old = g1 / g1_l[0]

    v_mu = mu
    #    print(x)

    stepcount = 0
    success_count = 0

    """perform step in f1-direction with step-size length of g1"""
    while g1_l[0] > 0.01 and stepcount < 100:

        grad = gradient(x, problem, singleObj=singleObj, shift=shift)

        g1 = grad[0]  # gradient of f_1
        g1_l = norm(g1)  # length of g1

        if g1_l[0] == 0:
            print("gradient of f1 zero - breaking.")
            break

        g1_v = g1 / g1_l[0]  # unit vector of g1

        if abs(angle(g1_v, g1_old)) > 90:
            #            print("bisec: ", abs(angle(g1_v, g1_old)))
            success_count = 0
            v_mu = v_mu * 0.5

        else:
            success_count = success_count + 1
            if success_count > 2:
                v_mu = v_mu * 2

        x = x + v_mu * g1_v  # step in g1 direction
        g1_old = g1_v

        stepcount = stepcount + 1
        if stepcount == 100:
            print("SO local search forced to terminate - not convergence after 100 steps.")

    #    print("niter(LS): ", stepcount, " [mu_last,mu_init] = [", v_mu, ",", mu, "]")
    return x


def somogsa(start_point, problem, term_ang, shift, bounds, step_mo, step_so, singleObj = False, steps=1000):

    gamma = 0.000001  # maximum length of a (local efficient) individual's gradient
    epsilon = 0.000001  # maximum difference between individuals to be considered identical

    x = start_point

    archive = []  # saves all possible optima
    opt1 = False  # indicates whether the optimum of f1 was found
    opt2 = False  # indicates whether the optimum of f2 was found

    plt.scatter(x[0], x[1])
    way = [[x[0]], [x[1]]]

    count = 0

    while count < steps and not (opt1 and opt2):

        count += 1

        opt1, opt2 = False, False
        runmog = False  # indicates whether MOGSA should be run after having followed the gradient of the first function (this is the case, when the algorithm jumped over a ridge)

        """run MOGSA"""
        success, x, path = MOGSA(start=x, problem=problem, bounds=bounds, step_mo=step_mo, ang_term=170, precision=gamma, singleObj=singleObj)  # Multi-Objective Gradient Sliding Algorithm
        plt.plot(path[0], path[1])
        for i in range(len(path[0])):
            way[0].append(path[0][i])
            way[1].append(path[1][i])

        x_start = x  # starting point on the found efficient set
        x_archive = x

        gs = gradient(x, problem, shift, singleObj=singleObj) # approximation of MOP's SO gradients; gs[0] and gs[1] are the single gradients
        g1 = norm(gs[0])  # norm returns length & unit vector of first functions' gradient
        g2 = norm(gs[1])  # norm returns length & unit vector of second functions' gradient

        g1_start = g1
        g2_start = g2

        """explore the efficient set by following the normalized gradient of the first objective"""
        while (count < steps):

            count += 1

            g_f1 = g1[1] * step_so
            x_new = x + g_f1

            x_new, restarted = checkbounds(bounds, x, x_new, g_f1)
            if restarted:
                x = x_new
                runmog = True  # random restart: an efficient set has to be found (run MOGSA)
                break

            plt.scatter(x_new[0], x_new[1])
            way[0].append(x_new[0])
            way[1].append(x_new[1])

            # no step performed? - end of efficient set is reached
            dist = np.linalg.norm((x-x_new))
            if dist <= epsilon:
                x_archive = x_new
                break

            gs_new = gradient(x_new, problem, shift, singleObj=singleObj)  # approximation of MOP's SO gradients; gs[0] and gs[1] are the single gradients
            g1_new = norm(gs_new[0])  # norm returns length & unit vector of first functions' gradient
            g2_new = norm(gs_new[1])  # norm returns length & unit vector of second functions' gradient

            # SO optimum in x_new? (length of gradient is smaller/equals the precision)
            if (g1_new[0] <= gamma) or (g2_new[0] <= gamma):
                x_archive = x_new
                opt1 = True
                break

            alpha = angle(g1[1], g1_new[1])  # angle between g1 and g1_new


            if alpha >= 90:  # same basin -> local search for f1
                archive.append(x)
                ls_initial_step = norm(x_new - x)[0]
                res = localsearch(x, problem, shift, singleObj=singleObj, mu=ls_initial_step)
                #res = sc.minimize(problem, x, method='nelder-mead')
                x_archive = res
                #x_archive = res.x
                plt.scatter(x_archive[0], x_archive[1])
                way[0].append(x_archive[0])
                way[1].append(x_archive[1])
                opt1 = True
                break
            else:
                beta = angle(g1_new[1], g2_new[1])  # angle between g1_new and g2_new
                if beta <= 90:  # jumped over a ridge
                    runmog = True
                    x = x_new
                    break

            x = x_new
            gs = gs_new
            g1 = g1_new
            g2 = g2_new


        if np.any(np.where(x_archive == archive)):
            x = init_random(bounds, len(x))
            plt.scatter(x[0],x[1])
            runmog = True
        else:
            archive.append(x_archive)


        if not runmog:
            x = x_start
            g1 = g1_start
            g2 = g2_start

        """explore the efficient set from the first found point of the set in the direction of the gradient of the second function"""
        while (count < steps and not runmog):

            count += 1

            g_f2 = g2[1] * step_so
            x_new = x + g_f2

            x_new, restarted = checkbounds(bounds, x, x_new, g_f2)
            plt.scatter(x_new[0], x_new[1])
            way[0].append(x_new[0])
            way[1].append(x_new[1])

            if restarted:
                x = x_new
                break

            # no step performed? - end of efficient set is reached / zulaessigen Bereich verlassen?
            dist = np.linalg.norm((x - x_new))
            if dist <= epsilon:
                break

            gs_new = gradient(x_new, problem, shift, singleObj=singleObj)
            g1_new = norm(gs_new[0])
            g2_new = norm(gs_new[1])

            # SO optimum in x_new? (length of gradient is smaller/equals the precision)
            if (g1_new[0] <= gamma) or (g2_new[0] <= gamma):
                archive.append(x_new)
                opt2 = True
                break

            alpha = angle(g2[1], g2_new[1])  # angle between g2 and g2_new

            # jumped over an optimum
            if (alpha >= 90):
                archive.append(x_new)
                opt2 = True
                x = x_new
                break
            else:
                beta = angle(g1_new[1], g2_new[1])  # angle between g1_new and g2_new
                if beta <= 90:  # jumped over a ridge
                    x = x_new
                    break

            x = x_new
            g2 = g2_new

    if not archive:
        plt.show()
        return archive, np.nan, np.nan

    # save the points and the corresponding function values of the first function
    f1archive = [[], []]
    for i in range(len(archive)):
        f1archive[0].append(archive[i])  # x-value
        fvalue = problem(archive[i])
        f1archive[1].append(fvalue)  # f1(x)
    fmini = min(f1archive[1])  # smallest function value
    xmini = archive[f1archive[1].index(fmini)]  # x-value for minimum function value
    print("min x: ", xmini, "with function value:", fmini, "after", count, "steps")
    print("archive: ", archive)
    print("farchive: ", f1archive)
    plt.show()

    workbook = xlsxwriter.Workbook('test3.xlsx')
    worksheet = workbook.add_worksheet()
    for row_num, row_data in enumerate(way):
        for col_num, col_data in enumerate(row_data):
            worksheet.write(col_num, row_num, col_data)
    workbook.close()

    return archive, xmini, fmini




if __name__ == '__main__':

    x = np.array([2, -2])  # startpoint

    # define the single-objective problem f1
    problem = rastrOneD
    # problem = bBOP

    term_ang = 170  # termination angle for switching to local search w.r.t. f1

    step_mo = 0.1  # scaling factor for step-size for MO gradient descent
    step_so = 0.1  # scaling factor for step-size for SO gradient descent

    shift = [-3.5, -2.5]  # location of the optimum of the sphere function f2

    l = -5
    u = 5
    bounds = np.array([[l, u], [l, u]])  # boundary of the search space

    opt, globoptx, globoptf = somogsa(x, problem, term_ang=term_ang, shift=shift, bounds=bounds, step_mo=step_mo, step_so=step_so, singleObj=True)

    pass







