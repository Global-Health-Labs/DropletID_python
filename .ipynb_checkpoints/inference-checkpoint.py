__author__ = 'Toan Huynh'
__copyright__ = 'Copyright 2019, Intellectual Ventures'
__credits__ = ['Toan Huynh']
__license__ = 'Apache 2.0'
__version__ = '1.0'
__maintainer__ = 'Toan Huynh'
__email__ = 'toanhuynh@intven.com'
__status__ = 'Development'
__fullname__ = 'Digital Variable-Volume'

import numpy as np
from scipy.optimize import fsolve
from scipy.special import erfc
import warnings


'''
in here: log is natural log (ln), base-10 log is log10
all array should be numpy arrays
lam: input concentration, copy number per volume unit
Lam: log(lam)
lam_array: array of input concentration, useful for simulations
Vm: array of volumes of sample compartments to characterize the volume distribution
Vn: array of volumes of the assay
Va: array of volumes of positive compartments
Vmu: mean of Vm
Vsigma: stdev of Vm
Vtotal: sum of Vn
n: total number of compartments
a: number of positive compartments
A: array of well results (matching V), each element: 1 or 0
'''

###########
# functions to calculate concentration estimates
###########

all_methods = ['cnt', 'amv', 'gmv', 'dvv', 'dvva', 'pp', 'ppa', 'hcf']

#quick approximations
lam_cnt = lambda Vtotal, a: a/Vtotal
lam_mean = lambda Vmu, n, a: -np.log(1-a/n)/Vmu if a < n else np.inf
lam_amv = lambda Vm, n, a: lam_mean(Vm.mean(),n,a)
lam_gmv = lambda Vm, n, a: lam_mean(np.exp(np.log(Vm).mean()),n,a)


def lam_dvv(Va, Vtotal):
    '''
    to calculate when we know the total volume, and the volume of each positive compartment
    :param Va: 1D array of volumes of ON compartments
    :param Vtotal: total volume of the compartments
    :return: inferred concentration, lam
    '''
    if Va.sum() == 0:
        out = 0
    elif Va.sum() == Vtotal:
        out = np.inf
    else:
        Flam_dvv = lambda lam, Va, Vtotal: (Va / (1 - np.exp(-Va * lam))).sum() - Vtotal  # function to optimize

        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                lam_ini_dvv = lam_amv(Va, Va.shape[0] * Vtotal / Va.sum(), Va.shape[0])
                out = fsolve(lambda lam: Flam_dvv(lam, Va, Vtotal), lam_ini_dvv)[0]
            except Warning as e:
                print('error:', e)
                print('amv failed as initial value, try gmv')
                try:
                    lam_ini_dvv = lam_gmv(Va, Va.shape[0] * Vtotal / Va.sum(), Va.shape[0])
                    out = fsolve(lambda lam: Flam_dvv(lam, Va, Vtotal), lam_ini_dvv)[0]
                except Warning as e:
                    print('error:', e)
                    print('gmv failed as initial value, try cnt')
                    try:
                        lam_ini_dvv = lam_cnt(Vtotal, Va.shape[0])
                        out = fsolve(lambda lam: Flam_dvv(lam, Va, Vtotal), lam_ini_dvv)[0]
                    except Warning as e:
                        print('error:', e)
                        print('cnt failed, give up and return nan')
                        out = np.nan
    return out


def lam_dvva(Vm, n, a):
    '''
    to calculate when the distribution is known, but not each volume
    :param Vm: volume of 
    :param n: total number of compartments
    :param a: number of ON compartments
    :return: inferred concentration, lam
    '''
    if a == 0:
        out = 0
    elif a == n:
        out = np.inf
    else:
        Flam_dvva = lambda lam, Vm, n, a: np.exp(-Vm*lam).mean() - 1 + a/n #function to optimize
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                lam_ini_dvva = lam_amv(Vm, n, a)
                out = fsolve(lambda lam: Flam_dvva(lam, Vm, n, a), lam_ini_dvva)[0]
            except Warning as e:
                print('error:', e)
                print('amv failed as initial value, try gmv')
                try:
                    lam_ini_dvva = lam_gmv(Vm, n, a)
                    out = fsolve(lambda lam: Flam_dvva(lam, Vm, n, a), lam_ini_dvva)[0]
                except Warning as e:
                    print('error found:', e)
                    print('gmv failed as initial value, try cnt')
                    try:
                        lam_ini_dvva = lam_cnt(Vm.mean()*n, a)
                        out = fsolve(lambda lam: Flam_dvva(lam, Vm, n, a), lam_ini_dvva)[0]
                    except Warning as e:
                        print('error found:', e)
                        print('cnt failed, give up and return nan')
                        out = np.nan
    return out


def lam_ppa(Vmu, Vsigma, n, a):
    '''
    Poisson Plus Approximation, based on the assumption of Gaussian distribution
    doi:10.1038/s41598-017-09183-4
    use equation 10
    :param Vmu: mean of volume distribution
    :param Vsigma: standard deviation of volume distribution
    :param n: total number of compartments
    :param a: number of ON compartments
    :return: inferred concentration, lam
    '''
    if a == 0:
        out = 0
    elif a == n:
        out = np.inf
    else:
        out = (Vmu-(Vmu**2 + 2*(Vsigma**2)*np.log(1-a/n))**0.5)/(Vsigma**2.0)
    return out


def lam_pp(Vmu, Vsigma, n, a):
    '''
    Poisson Plus, based on the assumption of truncated Gaussian distribution
    doi:10.1038/s41598-017-09183-4
    use equation 13
    :param Vmu: mean of volume distribution
    :param Vsigma: standard deviation of volume distribution
    :param n: total number of compartments
    :param a: number of ON compartments
    :return: inferred concentration, lam
    '''
    if a == 0:
        out = 0
    elif a == n:
        out = np.inf
    else:
        Poff_gauss_truncated = lambda Vmu, Vsigma, lam: \
            erfc(-1/(2**0.5)*(Vmu/Vsigma-lam*Vsigma))/erfc(-1/(2**0.5)*Vmu/Vsigma)*np.exp(-lam*Vmu+1/2*(Vsigma*lam)**2)
        lam_ini_pp = lam_mean(Vmu, n, a)
        out = fsolve(lambda lam: Poff_gauss_truncated(Vmu, Vsigma, lam)-1+a/n, lam_ini_pp)[0]
    return out


def lam_hcf(Vmu, Vsigma, n, a):
    '''
    doi.org/10.1373/clinchem.2014.221366
    equation 7 in supplementary materials
    :param Vmu: mean of volume distribution
    :param Vsigma: standard deviation of volume distribution
    :param n: total number of compartments
    :param a: number of ON compartments
    :return: inferred concentration, lam
    '''
    if a == 0:
        out = 0
    elif a == n:
        out = np.inf
    else:
        out = ((Vsigma/Vmu)**(-2))*((1-a/n)**(-((Vsigma/Vmu)**2))-1)/Vmu
    return out


# standard errors, confidence intervals
sigmaLam_dvv = lambda lam, Vn, n: (lam**2*(Vn**2/(np.exp(Vn*lam-1))).sum())**-0.5 # if each volume is known
sigmaLam_dvva = lambda lam, Vm, n: np.exp(-Vm*lam).mean()**0.5 * (1-np.exp(-Vm*lam).mean())**0.5 / n**0.5 / (lam*Vm*np.exp(-lam*Vm)).mean()
sigmalam_hfc = lambda lam, Vmu, Vsigma, n: ((n-1)**-0.5)*(1+Vsigma**2/Vmu*lam)*((1+Vsigma**2/Vmu*lam)**(Vmu**2/Vsigma**2)-1)**0.5

#get 2-sided z from confidence interval size
z = lambda conf: st.norm.ppf((1+conf)/2)
boundlam = lambda lam, sigmaLam, conf: np.exp(np.log(lam) + np.array([-1,1])*z(conf)*sigmaLam)

def lam_from_full(Vm, Vn, A, methods=all_methods):
    '''
    calculate concentrations using different methods
    :param Vm: volumes of pre-sample compartments
    :param Vn: volumes of compartments
    :param A: True/False array of reaction results from compartments
    :param methods: methods to calculate
    :return: dictionary containing results obtained using different methods
    '''
    n = Vn.shape[0]
    a = A.sum()
    Vtotal = Vn.sum()
    Va = Vn[A]
    Vmu = Vm.mean()
    Vsigma = Vm.std()

    lam={}
    if 'cnt' in methods:
        lam['cnt'] = lam_cnt(Vtotal, a)
    if 'amv' in methods:
        lam['amv'] = lam_amv(Vm, n, a)
    if 'gmv' in methods:
        lam['gmv'] = lam_gmv(Vm, n, a)
    if 'dvv' in methods:
        lam['dvv'] = lam_dvv(Va, Vtotal)
    if 'dvva' in methods:
        lam['dvva'] = lam_dvva(Vm, n, a)
    if 'pp' in methods:
        lam['pp'] = lam_pp(Vmu, Vsigma, n, a)
    if 'ppa' in methods:
        lam['ppa'] = lam_ppa(Vmu, Vsigma, n, a)
    if 'hcf' in methods:
        lam['hcf'] = lam_hcf(Vmu, Vsigma, n, a)

    return lam

