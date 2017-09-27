import numpy as np
import os, inspect
import dogs, uq
import scipy.io as io
from pathlib import Path
import shutil
import tr


##########  Initialize function ##########
def __main___:

    n = 3  # Dimension of data
    K = 3  # Tuning parameter for continuous search function
    Nm = 8  # Initial mesh grid size
    L = 1  # Tuning parameter for discrete search function
    flag = 1  # Identify
    method = "NPS"  # The strategy for regression function, you can choose NPS or MAPS
    user = 'Imperial College'
    # fe_times = np.array([])  # Represents the times of function evaluation at this point.

    # The following lines represents the initial points:
    # bnd1: lower bounds for physical data
    # bnd2: upper bounds for physical data
    # xE: initial interested points
    # y0: estimate value for minimum
    if n == 1:
        xE = np.array([[0.5, 0.75]])
        bnd2 = np.array([30])
        bnd1 = np.array([24])

    elif n == 2:
        xE = np.array([[0.5, 0.75, 0.5], [0.5, 0.5, 0.75]])
        bnd2 = np.array([30, 30])
        bnd1 = np.array([24, 24])
    elif n == 3:
        xE = np.array([[0.5, 0.5, 0.5, 0.75], [0.5, 0.5, 0.75, 0.5], [0.5, 0.75, 0.5, 0.5]])
        bnd2 = np.array([30, 30, 30])
        bnd1 = np.array([24, 24, 24])

    xU = dogs.bounds(np.zeros([n, 1]), np.ones([n, 1]), n)

    xE = dogs.physical_bounds(xE, bnd1, bnd2)
    xU = dogs.physical_bounds(xU, bnd1, bnd2)

    k = 0  # times of iteration, start with 0
    iter_max = 50  # maximum iteration steps
    idx = 0

    print('==================================================')

    current_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

    pre_opt = current_path + "/allpoints/pre_opt_IC.mat"
    pre_opt_path = Path(pre_opt)

    pre_Y = current_path + "/allpoints/Yall.mat"
    pre_Y_path = Path(pre_Y)

    pre_J = current_path + "/allpoints/surr_J_new.dat"
    pre_J_path = Path(pre_J)
    # Check whether or not it is the first iteraiton, if the optimizaton information file pre_opt_IC doesn't exist, then
    # generate that file:

    var_opt = io.loadmat("allpoints/pre_opt_IC")
    k = var_opt['iter'][0, 0]
    flag = var_opt['flag'][0, 0]

    # initilizaiton
    if k == 0:  # k is the number of iteration. k = 0 means that the initialization is not finished.

        if pre_Y_path.is_file():  # The file 'Yall' exists.
            # Read from the previous points
            data = io.loadmat("allpoints/Yall")
            yE = data['yE'][0]
            SigmaT = data['SigmaT'][0]
            T = data['T'][0]

            # if sign == 1:
            zs = np.loadtxt("allpoints/surr_J_new.dat")
            xx = uq.data_moving_average(zs, 40).values
            ind = tr.transient_removal(xx)
            sig = np.sqrt(uq.stationary_statistical_learning_reduced(xx[ind:], 18)[0])
            t = len(zs)  # not needed for Alpha-DOGS  # TODO fix the length of data
            J = np.abs(np.mean(xx[ind:]))

            yE = np.hstack((yE, J))
            SigmaT = np.hstack((SigmaT, sig))
            T = np.hstack((T, t))

            data = {'yE': yE, 'SigmaT': SigmaT, 'T': T}
            io.savemat("allpoints/Yall", data)

            print(' len of yE = ', len(yE))

        else:
            if not pre_J_path.is_file():
                # The very first iteration, the file 'Yall' doesn't exist.
                yE = np.array([])
                print("The first time running the iteration")
                print(' len of yE = ', len(yE))
                print('iter k = ', k)
            else:
                # The second time of running the algorithm.
                yE = np.array([])
                SigmaT = np.array([])
                T = np.array([])

                # Read from surr_J_new.
                zs = np.loadtxt("allpoints/surr_J_new.dat")

                xx = uq.data_moving_average(zs, 40).values
                ind = tr.transient_removal(xx)
                sig = np.sqrt(uq.stationary_statistical_learning_reduced(xx[ind:], 18)[0])
                t = len(zs)  # not needed for Alpha-DOGS
                J = np.abs(np.mean(xx[ind:]))

                yE = np.hstack((yE, J))
                SigmaT = np.hstack((SigmaT, sig))
                T = np.hstack((T, t))  # not needed for Alpha-DOGS
                data = {'yE': yE, 'SigmaT': SigmaT, 'T': T}
                io.savemat("allpoints/Yall", data)

                print("The second time running the iteration")
                print(' len of yE = ', len(yE))
                print('iter k = ', k)
                print('function evaluation at this iteration: ', J)

        # we read pre_opt_IC.mat
        # untill len(yE) < n+1 then k=1
        # var_opt = io.loadmat("allpoints/pre_opt_IC")
        # The following variables are needed for initialization:

        xE = var_opt['xE']
        n = var_opt['n'][0, 0]

        if len(yE) < xE.shape[1]:
            # Generate the point that we want to evaluate.
            xcurr = np.copy(xE[:, len(yE)])
            fout = open("allpoints/pts_to_eval.dat", 'w')
            keywords = ['Awin', 'lambdain', 'fanglein']
            fout.write(str('flagin') + '=' + str(int(flag)) + "\n")
            fout.write(str('IDin') + '=' + str(int(len(yE))) + "\n")
            for j in range(n):
                fout.write(keywords[j] + '=' + str(float(xcurr[j])) + "\n")
            fout.close()

            var_opt['iter'] = 0
            var_opt['num_point'] = len(yE)
            io.savemat("allpoints/pre_opt_IC", var_opt)

            print('point to eval at this iteration x = ', xcurr)
            print('len of yE = ', len(yE))

            return
        else:

            # Initialization complete
            var_opt['iter'] = 1
            io.savemat("allpoints/pre_opt_IC", var_opt)
            # Run one iteration after initialization.
            if pre_J_path.is_file():
                dogs.DOGS_standalone_IC()
                os.remove(current_path + "/allpoints/surr_J_new.dat")
                return
            else:
                return

    else:
        if pre_J_path.is_file():  # If surr_J_new exists, function evaluation is succeeded.
            dogs.DOGS_standalone_IC()
            var_opt = io.loadmat("allpoints/pre_opt_IC")
            flag = var_opt['flag'][0, 0]
            if flag != 2:  # If flag == 2, mesh refinement, perform one more iteration.
                os.remove(current_path + "/allpoints/surr_J_new.dat")
            return
        else:  # function evaluation is failed, reperform functon evaluation
            return


def run_opti():
    current_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    # generate the directory
    apts = current_path + "/allpoints"
    if not os.path.exists(apts):
        os.makedirs(apts)
    # create the stop file
    stp = current_path + "allpoints/stop.dat"
    stp_path = Path(stp)
    if not stp_path.is_file():
        stop = 0
        fout = open("allpoints/stop.dat", 'w')
        fout.write(str(stop) + "\n")
        fout.close()

    while stop == 0:
        # DOGS_standalone() will first generate the initial points, then run alpha-DOGS algorithm.
        DOGS_standlone()

        # The "solver_lorenz" function will read from "pts_to_eval" and generate a file named "surr_J_new.dat"
        # containing all information about function evaluation.
        dogs.solver_lorenz()


        # Read stop file
        stop = int(np.loadtxt("allpoints/stop.dat"))
        print('stop = ', stop)
    return


################################################################################################################
run_opti()
DOGS_standlone()

# Delete the directory of allpoints
current_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
shutil.rmtree(current_path + "/allpoints")



# TODO FIXME: nff is deleted
from dogs import *
'''
  This function reads the set of evaluated points and writes them into the desired file to perform function evaluations
  Note: DOGS_standalone() only exists at the inactivated iterations.
  :return: points that needs to be evaluated
  '''
# For future debugging, remind that xc and xd generate by DOGS_standalone() is set to be a one dimension row vector.
# While lb and ub should be a two dimension matrix, i.e. a column vector.
var_opt = io.loadmat("allpoints/pre_opt")
n = var_opt['n'][0, 0]
K = var_opt['K'][0, 0]
L = var_opt['L'][0, 0]
Nm = var_opt['Nm'][0, 0]
bnd2 = var_opt['ub'][0]
bnd1 = var_opt['lb'][0]
lb = np.zeros(n)
ub = np.ones(n)
user = var_opt['user'][0]
idx = var_opt['num_point'][0, 0]
flag = var_opt['flag'][0, 0]
T_lorenz = var_opt['T_lorenz']
h = var_opt['h_lorenz']
method = var_opt['inter_par_method']
xE = var_opt['xE']
xU = var_opt['xU']
if xU.shape[1] == 0:
    xU = xU.reshape(n, 0)

Data = io.loadmat("allpoints/Yall")
yE = Data['yE'][0]
SigmaT = Data['SigmaT'][0]


xE = np.array([[0.5, 0.5, 0.5, 0.75], [0.5, 0.5, 0.75, 0.5], [0.5, 0.75, 0.5, 0.5]])
bnd2 = np.array([30, 30, 30])
bnd1 = np.array([24, 24, 24])

xU = dogs.bounds(np.zeros([n, 1]), np.ones([n, 1]), n)

xE = dogs.physical_bounds(xE, bnd1, bnd2)
xU = dogs.physical_bounds(xU, bnd1, bnd2)

# initilization
Ain = np.concatenate((np.identity(n), -np.identity(n)), axis=0)
Bin = np.concatenate((np.ones((n, 1)), np.zeros((n, 1))), axis=0)
yE = []
for ll in range(xE.shape[1]):
    J = mpl_cost(xE[:,ll])
    yE = np.hstack([yE,J])


inter_par = Inter_par(method='NPS')


[inter_par, yp] = interpolateparameterization(xE, yE, inter_par)


K0 = 20  # K0 = np.ptp(yE, axis=0)
# Calculate the discrete function.

yd = np.amin(sd)
ind_exist = np.argmin(yE)



#################################### Adaptive K method ####################################
def tringulation_search_bound(inter_par, xi, y0, ind_min):
    inf = 1e+20
    n = xi.shape[0]
    xm, ym = inter_min(xi[:, ind_min], inter_par)
    ym = ym[0, 0]  # If using scipy package, ym would first be a two dimensions array.
    sc_min = inf
    # cse=1
    if ym > y0:
        ym = inf
    # cse =2
    # construct Deluanay tringulation
    if n == 1:
        sx = sorted(range(xi.shape[1]), key=lambda x: xi[:, x])
        tri = np.zeros((xi.shape[1] - 1, 2))
        tri[:, 0] = sx[:xi.shape[1] - 1]
        tri[:, 1] = sx[1:]
        tri = tri.astype(np.int32)
    else:
        options = 'Qt Qbb Qc' if n <= 3 else 'Qt Qbb Qc Qx'
        # tri = scipy.spatial.Delaunay(xi.T, qhull_options=options).simplices
        tri = scipy.spatial.Delaunay(xi.T).simplices
        keep = np.ones(len(tri), dtype=bool)
        for i, t in enumerate(tri):
            if abs(np.linalg.det(np.hstack((xi.T[t], np.ones([1, n + 1]).T)))) < 1E-15:
                keep[i] = False  # Point is coplanar, we don't want to keep it
        tri = tri[keep]

    Sc = np.zeros([np.shape(tri)[0]])
    Scl = np.zeros([np.shape(tri)[0]])
    for ii in range(np.shape(tri)[0]):
        R2, xc = circhyp(xi[:, tri[ii, :]], n)
        # if R2 != np.inf:
        if R2 < inf:
            # initialze with body center of each simplex
            x = np.dot(xi[:, tri[ii, :]], np.ones([n + 1, 1]) / (n + 1))
            Sc[ii] = (interpolate_val(x, inter_par) - y0) / (R2 - np.linalg.norm(x - xc) ** 2)
            if np.sum(ind_min == tri[ii, :]):
                Scl[ii] = np.copy(Sc[ii])
            else:
                Scl[ii] = inf
        else:
            Scl[ii] = inf
            Sc[ii] = inf

    # Global one
    ind = np.argmin(Sc)
    R2, xc = circhyp(xi[:, tri[ind, :]], n)
    x = np.dot(xi[:, tri[ind, :]], np.ones([n + 1, 1]) / (n + 1))
    xm, ym = Adoptive_K_Search(x, inter_par, xc, R2, y0, K0)
    # Local one
    ind = np.argmin(Scl)
    R2, xc = circhyp(xi[:, tri[ind, :]], n)
    # Notice!! ind_min may have a problen as an index
    x = np.copy(xi[:, ind_min])
    xml, yml = Adoptive_K_Search(x, inter_par, xc, R2, y0, K0)
    if yml < 2 * ym:
        xm = np.copy(xml)
        ym = np.copy(yml)
    return xm, ym







xd = xE[:, ind_exist]

    # Calcuate the unevaluated function:
yu = np.zeros([1, xU.shape[1]])
xc, yc =  tringulation_search_bound(inter_par, np.hstack([xE, xU]), 0, 0)
# xc, yc = tringulation_search_bound_constantK(inter_par, np.hstack([xE, xU]), 1, 0)
yc = yc[0, 0]
if interpolate_val(xc, inter_par) < min(yp):
    xc = np.round(xc * Nm) / Nm
        break

            else:
                xc = np.round(xc * Nm) / Nm
                if mindis(xc, xE)[0] < 1e-6:
                    break
                xc, xE, xU, success, _ = points_neighbers_find(xc, xE, xU, Bin, Ain)
                xc = xc.T[0]
                if success == 1:
                    break
                else:
                    yu = np.hstack([yu, (interpolate_val(xc, inter_par) - min(yp)) / mindis(xc, xE)[0]])

        if xU.shape[1] != 0:
            tmp = (interpolate_val(xc, inter_par) - min(yp)) / mindis(xc, xE)[0]
            if np.amin(yu) < tmp:
                ind = np.argmin(yu)
                xc = np.copy(xU[:, ind])
                yc = -np.inf
                xU = scipy.delete(xU, ind, 1)  # create empty array

    if mindis(xc, xE)[0] < 1e-6:
        K = 2 * K
        Nm = 2 * Nm
        L += 1
        flag = 2  # flag = 2 represents mesh refinement, in this step we don't have function evaluation.



