import numpy as np
import math
import os, ctypes
from scipy import LowLevelCallable
import scipy.integrate as si
from scipy.stats import norm
from scipy.stats import uniform
from scipy.special import gamma, kv
# from scipy.stats import uniform
import scipy.interpolate as interp
from scipy.stats import genextreme
import sys





## -------------------------------------------------------------------------- ##
## -------------------------------------------------------------------------- ##
## --------------------------- integration.cpp ------------------------------ ##
## -------------------------------------------------------------------------- ##
## -------------------------------------------------------------------------- ##

## -------------------------------------------------------------------------- ##
## Generate Pareto random variables
##

def rPareto(n, location, shape = 1):
    if(isinstance(n, (int, np.int64, float))): n=np.array([n])
    if n.size < 1:
        sys.exit("'n' must be non-empty.")
    if n.size > 1:
        n = n.size
    else:
        if np.isnan(n) or n <= 0:
            sys.exit("'n' must be a positive integer or vector.")
    
    return location*(1-uniform.rvs(0,1,n))**(-1/shape)
    
##
## -------------------------------------------------------------------------- ##









## -------------------------------------------------------------------------- ##
## Calculate CDF of scale mixture marginals
##

def asymptotic_p(xval, delta):
    if abs(delta-0.5)<1e-9:
        result = 1-(math.log(xval)+1)/xval
    else:
        result = 1-(delta/(2*delta-1))*xval**((delta-1)/delta)+((1-delta)/(2*delta-1))/xval
    return result



## Approach 1: LowLevelCallable from C++
## gcc -shared -fPIC -o d_integrand.so d_integrand.c
lib = ctypes.CDLL(os.path.abspath('./scalemixture_temp/p_integrand.so'))
lib.f.restype = ctypes.c_double
lib.f.argtypes = (ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.c_void_p)

c = ctypes.c_double(0.55)
user_data = ctypes.cast(ctypes.pointer(c), ctypes.c_void_p)

func_p = LowLevelCallable(lib.f, user_data)

def pmixture_me_C(xval, delta, tau_sqd):
    tmp1 = delta/(2*delta-1)
    tmp2 = -(1-delta)/delta
    tmp3 = (1-delta)/(2*delta-1)
    tmp4 = 1/(2*tau_sqd)
    sd = np.sqrt(tau_sqd)
    
    # Nugget error gets to 100 times the sd? Negligible!
    if tau_sqd < 0.05:
        tmp = 0
        if xval>1:
            tmp = asymptotic_p(xval, delta)
        return tmp
    my_lower_bound = 100*sd
    I_1 = si.quad(func_p, -my_lower_bound, xval-1, args=(xval, tau_sqd, tmp1, tmp2, tmp3, tmp4), full_output=1)
    tmp = - I_1[0]/(np.sqrt(2*np.pi)*sd)
    if tmp<1e-5 and xval>10:
        return asymptotic_p(xval, delta)
    else:
        return tmp

## Vectorize pmixture_me_C
pmixture_me = np.vectorize(pmixture_me_C,otypes=[np.float])




## Approach 2: define integrand in Python
def mix_distn_integrand(t, xval, delta, tau_sqd, tmp1, tmp3, tmp4):
    tmp2 = (xval-t)**(-1)
    if abs(delta-0.5)<1e-9:
        half_result = (np.log(xval-t)+1)*(xval-t)**(-1)
    else:
        half_result = tmp3*tmp2**((1-delta)/delta)-tmp4*tmp2
    dnorm = math.exp(-t**2/(2*tau_sqd))
    return dnorm*half_result


def pmixture_me_pre(xval, my_lower_bound, delta, tau_sqd, tmp1, tmp3, tmp4):
    I_1 = si.quad(mix_distn_integrand, -my_lower_bound, xval-1, args=(xval, delta, tau_sqd, tmp1, tmp3, tmp4), full_output=1)
   
    return I_1



def pmixture_me_uni(xval, delta, tau_sqd):
    # Randomly assign 'tmp1' a value when 'delta==0.5' because it is not needed.
    if delta==0.5:
        tmp1 = np.inf
    else:
        tmp1 = 1/(2*delta-1)
    tmp3 = delta*tmp1
    tmp4 = (1-delta)*tmp1
    sd = math.sqrt(tau_sqd)
    
    # Nugget error gets to 100 times the sd? Negligible!
    my_lower_bound = 100*sd
    if tau_sqd < 0.05:
        tmp = 0
        if xval>0:
            tmp = asymptotic_p(xval, delta)
        return tmp
    I_1 = si.quad(mix_distn_integrand, -my_lower_bound, xval-1, args=(xval, delta, tau_sqd, tmp1, tmp3, tmp4), full_output=1)
    I_2 = norm.cdf(xval-1, loc=0.0, scale=sd)
    tmp = I_2 - I_1[0]/math.sqrt(2*math.pi*tau_sqd)
    if tmp<0.975:
        return tmp
    else:
        return asymptotic_p(xval, delta)

## --------------  Vectorize pmixture_me_uni ------------------
## This is equivalent to defining the following function:
  # def pmixture_me(xvals, delta, tau_sqd):
  #  n = xvals.shape[0]
  #  resultVec = np.zeros(n)
  #  for idx, xval in enumerate(xvals):
  #      resultVec[idx] = pmixture_me_uni(xval, delta, tau_sqd)
  #  return resultVec
## Test the function outputs:
## pmixture_me(np.array([3,3.4,5.4]),0.55,4)
pmixture_me_py = np.vectorize(pmixture_me_uni)

    
##
## -------------------------------------------------------------------------- ##



## -------------------------------------------------------------------------- ##
## This only makes sense if we want to search over x > 0.  Since we are interested
## in extremes, this seems okay -- things are not so extreme if they are in the
## lower half of the support of the (copula) distribution.
##
## Would be nice to control the relerr of 'pmixture_me' but I never used it.
##                                                                            ##

def find_xrange_pmixture_me(min_p, max_p, x_init, delta, tau_sqd):#, relerr = 1e-10):
    x_range = np.zeros(2)
    min_x = x_init[0]
    max_x = x_init[1]
    
    # if min_x <= 0 or min_p <= 0.15:
    #     sys.exit('This will only work for x > 0, which corresponds to p > 0.15.')
    if min_x >= max_x:
        sys.exit('x_init[0] must be smaller than x_init[1].')
    
    ## First the min
    p_min_x = pmixture_me_C(min_x, delta, tau_sqd)
    while p_min_x > min_p:
        # print('left x is {}'.format(min_x))
        # print('F({})={}'.format(min_x, p_min_x))
        min_x = min_x-5/delta
        p_min_x = pmixture_me_C(min_x, delta, tau_sqd)
    
    x_range[0] = min_x
    
    ## Now the max
    p_max_x = pmixture_me_C(max_x, delta, tau_sqd)
    while p_max_x < max_p:
        # print(' right x is {}'.format(max_x))
        # print('F({})={}'.format(max_x, p_max_x))
        max_x = max_x*1.5
        p_max_x = pmixture_me_C(max_x, delta, tau_sqd)
    
    x_range[1] = max_x
    return x_range

##                                                                            ##
## -------------------------------------------------------------------------- ##



import sklearn
import sklearn.isotonic
# from sklearn.experimental import enable_hist_gradient_boosting  # noqa
# from sklearn.ensemble import HistGradientBoostingRegressor
# x_vals_tmp = x_vals.reshape(-1,1)
# cdf_gbdt = HistGradientBoostingRegressor(monotonic_cst=[1]).fit(x_vals_tmp, cdf_vals)
# cdf_vals_1 = cdf_gbdt.predict(x_vals_tmp)


## -------------------------------------------------------------------------- ##
## Approximates the marginal quantile function by taking values of the
## marginal CDF of X and doing linear interpolation.  If no values of the CDF
## are supplied, it computes n.x of them, for x in (lower, upper).
##
##
def qmixture_me_interp(p, delta, tau_sqd, cdf_vals = np.nan, x_vals = np.nan,
                               n_x=200, lower=5, upper=20):
    
  if type(p).__module__!='numpy':
      p = np.array(p)
  large_delta_large_x = False
  if np.any(np.isnan(x_vals)):
    x_range = find_xrange_pmixture_me(np.nanmin(p),np.nanmax(p), np.array([lower,upper]), delta, tau_sqd)
    if np.isinf(x_range[1]):
        x_range[1] = 10^20; large_delta_large_x = True
    if np.any(x_range<=0):
            x_vals = np.concatenate((np.linspace(x_range[0], 0.0001, num=150),
                           np.exp(np.linspace(np.log(0.0001001), np.log(x_range[1]), num=n_x))))
    else:
            x_vals = np.exp(np.linspace(np.log(x_range[0]), np.log(x_range[1]), num=n_x))
    cdf_vals = pmixture_me(x_vals, delta, tau_sqd)
  else:
    if np.any(np.isnan(cdf_vals)):
      cdf_vals = pmixture_me(x_vals, delta, tau_sqd)
  
  # Obtain the quantile level using the interpolated function
  if not large_delta_large_x:
     zeros = sum(cdf_vals<1e-70)
     try:
         tck = interp.pchip(cdf_vals[zeros:], x_vals[zeros:]) # 1-D monotonic cubic interpolation.
     except ValueError:
         ir = sklearn.isotonic.IsotonicRegression(increasing=True)
         ir.fit(x_vals[zeros:], cdf_vals[zeros:])
         cdf_vals_1 = ir.predict(x_vals[zeros:])
         indices = np.where(np.diff(cdf_vals_1)==0)[0]+1
         tck = interp.pchip(np.delete(cdf_vals_1,indices), np.delete(x_vals[zeros:],indices))
     q_vals = tck(p)
  else:
     which = p>cdf_vals[-1]
     q_vals = np.repeat(np.nan, np.shape(p)[0])
     q_vals[which] = x_range[1]
     if np.any(~which):
         # tck = interp.interp1d(cdf_vals, x_vals, kind = 'cubic')
         tck = interp.pchip(cdf_vals, x_vals)
         q_vals[~which] = tck(p[~which])
  return q_vals
  
      
##
## -------------------------------------------------------------------------- ##



# # The CDF function can't deal with delta = 0.5
# import matplotlib.pyplot as plt
# axes = plt.gca()
# # axes.set_ylim([0,0.125])
# X_vals = np.linspace(-10,300,num=300)

# import time
# delta=0.3; tau_sqd=10

# start_time = time.time()
# D_asym = asymptotic_p(X_vals, delta)
# time.time() - start_time

# start_time = time.time()
# D_mix = pmixture_me(X_vals, delta, tau_sqd)
# time.time() - start_time

# start_time = time.time()
# D_py = pmixture_me_py(X_vals, delta, tau_sqd)
# time.time() - start_time

# fig, ax = plt.subplots()
# ax.plot(X_vals[3:], D_asym[3:], 'b', label="Smooth R^phi*W")
# ax.plot(X_vals, D_mix, 'r',linestyle='--', label="With nugget: C++ lowlevel callable")
# ax.plot(X_vals, D_py, 'g',linestyle=':', label="With nugget: numerical int")
# legend = ax.legend(loc = "lower right",shadow=True)
# plt.title(label="Delta")
# plt.show()


# q_vals = qmixture_me_interp(np.array([0.2,0.6,0.9]), delta, tau_sqd)
# plt.plot(D_mix, X_vals, 'r',linestyle='--')
# plt.scatter(np.array([0.2,0.6,0.9]), q_vals)



## -------------------------------------------------------------------------- ##
## Calculate PDF of scale mixture marginals
##
def asymptotic_d(xval, delta):
    if abs(delta-0.5)<1e-9:
        result=xval**(-2)*math.log(xval)
    else:
        result = ((1-delta)/(2*delta-1))*(xval**(-1/delta)-xval**(-2))
    return result



## Approach 1: LowLevelCallable from C++
## gcc -shared -fPIC -o d_integrand.so d_integrand.c
libd = ctypes.CDLL(os.path.abspath('./scalemixture_temp/d_integrand.so'))
libd.f.restype = ctypes.c_double
libd.f.argtypes = (ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.c_void_p)

c = ctypes.c_double(0.55)
user_data = ctypes.cast(ctypes.pointer(c), ctypes.c_void_p)
func_d = LowLevelCallable(libd.f, user_data)


def dmixture_me_C(xval, delta, tau_sqd):
    tmp1 = (1-delta)/(2*delta-1)
    tmp2 = -1/delta
    tmp3 = 1/(2*tau_sqd)
    sd = np.sqrt(tau_sqd)

    # Nugget error gets to 100 times the sd? Negligible!
    if tau_sqd<0.05:
        tmp = 0
        if xval>1:
            tmp = asymptotic_d(xval, delta)
        return tmp
    my_lower_bound = 100*sd
    I_1 = si.quad(func_d, -my_lower_bound, xval-1, args=(xval, tau_sqd, tmp1, tmp2, tmp3), full_output=1)
    tmp_res = - I_1[0]/(np.sqrt(2*np.pi)*sd)
    if xval>1 and tmp_res<1e-4:
        tmp_res_asymp = asymptotic_d(xval, delta)
        if tmp_res/tmp_res_asymp<0.1:
            tmp_res = tmp_res_asymp
    return tmp_res


## Vectorize dmixture_me_C
dmixture_me = np.vectorize(dmixture_me_C,otypes=[np.float])



## Approach 2: define integrand in Python
def mix_dens_integrand(t, xval, delta, tau_sqd, tmp1):
    tmp2 = xval-t
    if abs(delta-0.5)<1e-9:
        half_result = -np.log(tmp2)*tmp2**(-2)
    else:
        half_result = tmp1*tmp2**(-2)-tmp1*tmp2**(-1/delta)
    dnorm = math.exp(-t**2/(2*tau_sqd))
    return dnorm*half_result


def dmixture_me_uni(xval, delta, tau_sqd):
    # Randomly assign 'tmp1' a value when 'delta==0.5' because it is not needed.
    if delta==0.5:
        tmp1 = np.inf
    else:
        tmp1 = (1-delta)/(2*delta-1)
    sd = math.sqrt(tau_sqd)
    # Nugget error gets to 100 times the sd? Negligible!
    my_lower_bound = 100*sd
    I_1 = si.quad(mix_dens_integrand, -my_lower_bound, xval-1, args=(xval, delta, tau_sqd, tmp1), full_output=1)
    
    tmp_res = -I_1[0]/math.sqrt(2*math.pi*tau_sqd)
    if xval>1 and tmp_res<1e-4:
        tmp_res_asymp = asymptotic_d(xval, delta)
        if tmp_res/tmp_res_asymp<0.1:
            tmp_res = tmp_res_asymp
    return tmp_res

    
## --------------  Vectorize dmixture_me_uni ------------------
## This is equivalent to defining the following function:
  # def dmixture_me(xvals, delta, tau_sqd):
  #   n = xvals.shape[0]
  #   resultVec = np.zeros(n)
  #   for idx, xval in enumerate(xvals):
  #       tmp_res  = dmixture_me_uni(xval, delta, tau_sqd)
  #       if(tmp_res<1e-4):
  #           tmp_res_asymp = asymptotic_d(xval, delta)
  #           if tmp_res/tmp_res_asymp<0.1:
  #               tmp_res = tmp_res_asymp
  #       resultVec[idx] = tmp_res
  #   return resultVec
## Test the function outputs
## dmixture_me(np.array([3,3.4,5.4]),0.55,4)
dmixture_me_py = np.vectorize(dmixture_me_uni)

##
## -------------------------------------------------------------------------- ##


## Approach 3: interpolate using a grid of density values
def density_interp_grid(delta, tau_sqd, grid_size=400):
    xp_0 = np.linspace(-100, -15, int(grid_size/8), endpoint = False)
    xp_1 = np.linspace(-15, 80, grid_size, endpoint = False)
    xp_2 = np.linspace(80, 200, int(grid_size/8), endpoint = False)
    xp_3 = np.linspace(200, 1000, int(grid_size/20), endpoint = False)
    xp_4 = np.linspace(1000, 15000, int(grid_size/40), endpoint = False)
    xp = np.concatenate((xp_0, xp_1, xp_2, xp_3, xp_4))
    
    xp = np.ascontiguousarray(xp, np.float64) #C contiguous order: xp.flags['C_CONTIGUOUS']=True?
    den_p = dmixture_me(xp, delta, tau_sqd)
    return (xp, den_p)


def dmixture_me_interpo(xvals, xp, den_p):
    if type(xvals).__module__!='numpy':
        xvals = np.array(xvals)
    
    den_vec = np.empty(xvals.shape)
    tck = interp.PchipInterpolator(xp, den_p, extrapolate=True)
    den_vec = tck(xvals)
        
    return den_vec        



# import matplotlib.pyplot as plt
# axes = plt.gca()
# # axes.set_ylim([0,0.125])
# X_vals = np.linspace(0.001,300,num=300)
# delta=0.3; tau_sqd=0.001
# grid = density_interp_grid(delta, tau_sqd)
# xp = grid[0]; den_p = grid[1]

# import time

# start_time = time.time()
# D_asym = asymptotic_d(X_vals, delta)
# time.time() - start_time

# start_time = time.time()
# D_mix = dmixture_me(X_vals, delta, tau_sqd)
# time.time() - start_time

# start_time = time.time()
# D_py = dmixture_me_py(X_vals, delta, tau_sqd)
# time.time() - start_time

# start_time = time.time()
# D_interp = dmixture_me_interpo(X_vals, xp, den_p)
# time.time() - start_time

# fig, ax = plt.subplots()
# ax.plot(X_vals[3:], D_asym[3:], 'b', label="Smooth R^phi*W")
# ax.plot(X_vals, D_mix, 'r',linestyle='--', label="With nugget: C++ lowlevel callable")
# ax.plot(X_vals[1:], D_py[1:], 'g',linestyle=':', label="With nugget: numerical int")
# ax.plot(X_vals[1:], D_interp[1:], 'y',linestyle='-.', label="With nugget: grid interpolation")
# legend = ax.legend(loc = "upper right",shadow=True)
# plt.title(label="Delta")
# plt.show()

def qRW_me_bisection(p, delta, tau_sqd, x_range, n_x=100):
    m = (x_range[0]+x_range[1])/2
    iter=0
    new_F = pmixture_me_C(m, delta, tau_sqd)-p
    while iter<100 and np.abs(new_F) > 1e-04:
        if new_F>0:
            x_range[1] = m
        else:
            x_range[0] = m
        m = (x_range[0]+x_range[1])/2
        new_F = pmixture_me_C(m, delta, tau_sqd)-p
        iter += 1
    return m

def qRW_me_newton(p, delta, tau_sqd, x_range, n_x=100):
    current_x = x_range[0]; iter=0; error=1
    while iter<400 and error > 1e-08:
        tmp = (pmixture_me_C(current_x, delta, tau_sqd)-p)/dmixture_me_C(current_x, delta, tau_sqd)
        new_x = current_x - tmp
        error = np.abs(new_x-current_x)
        iter += 1
        current_x = max(x_range[0],new_x)
        if(current_x==x_range[0]): current_x = qRW_me_bisection(p, delta, tau_sqd, x_range)
        
    return current_x
            
# Vectorize
def qmixture_me_Newton(p, delta, tau_sqd, n_x=400):
    if type(p).__module__!='numpy':
      p = np.array(p)
    
    x_range = find_xrange_pmixture_me(np.nanmin(p),np.nanmax(p), np.array([5,20]), delta, tau_sqd)
    results = np.empty(p.shape)
    
    for x in np.ndindex(p.shape):
        results[x] = qRW_me_newton(p[x], delta, tau_sqd, x_range, n_x)
        
    return results






## -------------------------------------------------------------------------- ##
## -------------------------------------------------------------------------- ##
## --------------------------- scalemix_utils.R ----------------------------- ##
## -------------------------------------------------------------------------- ##
## -------------------------------------------------------------------------- ##




      


## -------------------------------------------------------------------------- ##
## Compute the Matern correlation function from a matrix of pairwise distances
## and a vector of parameters
##

def corr_fn(r, theta, sill=1):
    if type(r).__module__!='numpy' or isinstance(r, np.float64):
      r = np.array(r)
    if np.any(r<0):
      sys.exit('Distance argument must be nonnegative.')
    r[r == 0] = 1e-10

    range = theta[0]
    nu = theta[1]
    part1 = 2 ** (1 - nu) / gamma(nu)
    part2 = (np.sqrt(2 * nu) * r / range) ** nu
    part3 = kv(nu, np.sqrt(2 * nu) * r / range)
    return sill*part1 * part2 * part3

##
## -------------------------------------------------------------------------- ##



## -------------------------------------------------------------------------- ##
## Compute the density of the mixing distribution in Huser-Wadsworth (2017).
## delta in the notation in the manuscript.
## The support of r is [1, infinity)
##
## R should ONLY be a vector.
##

def dhuser_wadsworth(R, delta, log=True):
    if type(R).__module__!='numpy' or isinstance(R, np.float64):
      R = np.array(R)
    if ~np.all(R>1):
      return -np.inf
    n_t = R.size

    if log:
      dens = n_t*np.log((1-delta)/delta)-np.sum(np.log(R))/delta
    else:
      dens = ((1-delta)/delta)**n_t*(np.prod(R))**(-1/delta)
    
    return dens
##
## -------------------------------------------------------------------------- ##



## -------------------------------------------------------------------------- ##
## For MVN
##

## Assumes that A = VDV', where D is a diagonal vector of eigenvectors of A, and
## V is a matrix of normalized eigenvectors of A.
## Computes A^{-1}x
##
def eig2inv_times_vector(V, d_inv, x):
  return V@np.diag(d_inv)@V.T@x


## Computes y=A^{-1}x via solving linear system Ay=x
from scipy.linalg import lapack
def inv_timeol_vector(A, x):
  inv = lapack.dposv(A,x)
  return inv


## Assumes that A = VDV', where D is a diagonal vector of eigenvectors of A, and
## V is a matrix of normalized eigenvectors of A.
##
## log(|A|)
##
def eig2logdet(d):
  return sum(np.log(d))


## Multivariate normal log density of R, where each column of
## R iid N(mean,VDV'), where VDV' is the covariance matrix
## It essentially computes the log density of each column of R, then takes
## the sum.  Faster than looping over the columns, but not as transparent.
##
## Ignore the coefficient: -p/2*log(2*pi)
##
def dmvn_eig(R, V, d_inv, mean=0):
  if len(R.shape)==1:
    n_rep = 1
  else:
    n_rep = R.shape[1]
  res = -0.5*n_rep*eig2logdet(1/d_inv) - 0.5 * np.sum((R-mean) * eig2inv_times_vector(V, d_inv, R-mean))
  return res

def dmvn(R, Cor, mean=0, cholesky_inv = None):## cholesky_inv is the output of inv_times_vector()
  if len(R.shape)==1:
    n_rep = 1
  else:
    n_rep = R.shape[1]
    
  if cholesky_inv is None:
      inv = lapack.dposv(Cor, R-mean)
  else:
      sol = lapack.dpotrs(cholesky_inv[0],R-mean) #Solve Ax = b using factorization
      inv = (cholesky_inv[0],sol[0])
  logdet = 2*sum(np.log(np.diag(inv[0])))
  res = -0.5*n_rep*logdet - 0.5 * np.sum((R-mean) * inv[1])
  return res

def dmvn_diag(R, sigma_squared, mean=0): #one-dim vec
  n_rep = R.shape[0]
  
  scale=np.sqrt(sigma_squared)
  vec = R-mean
  res = [norm.logpdf(vec[i], loc=0, scale=scale) for i in np.arange(n_rep)]
  return sum(res)

## Assumes that A = VDV', where D is a diagonal vector of eigenvectors of A, and
## V is a matrix of normalized eigenvectors of A.
##
## Computes x'A^{-1}x
##
def eig2inv_quadform_vector(V, d_inv, x):
  cp = V@np.diag(d_inv)@V.T@x
  return sum(x*cp)

def inv_quadform_vector(Cor, x, cholesky_inv = None):
  if cholesky_inv is None:
      inv = lapack.dposv(Cor, x)
      cp = inv[1]
  else:
      sol = lapack.dpotrs(cholesky_inv[0],x) #Solve Ax = b using factorization
      cp = sol[0]
  return sum(x*cp)

##
## -------------------------------------------------------------------------- ##


## -------------------------------------------------------------------------- ##
## -------------------------------------------------------------------------- ##
##                    Transform Normal to Standard Pareto
## -------------------------------------------------------------------------- ##
## -------------------------------------------------------------------------- ##
##
##
## i.e., Stable variables with alpha=1/2
##
def norm_to_Pareto(z):
    if(isinstance(z, (int, np.int64, float))): z=np.array([z])
    tmp = norm.cdf(z)
    if np.any(tmp==1): tmp[tmp==1]=1-1e-9
    return 1/(1-tmp)


def pareto_to_Norm(W):
    if(isinstance(W, (int, np.int64, float))): W=np.array([W])
    if np.any(W<1): sys.exit("W must be greater than 1")
    tmp = 1-1/W
    return norm.ppf(tmp)
        
## -------------------------------------------------------------------------- ##




## -------------------------------------------------------------------------- ##
## For generalized extreme value (GEV) distribution
## Negative shape parametrization in scipy.genextreme
##

def dgev(yvals, Loc, Scale, Shape, log=False):
    if log:
        return genextreme.logpdf(yvals, c=-Shape, loc=Loc, scale=Scale)  # Opposite shape
    else:
        return genextreme.pdf(yvals, c=-Shape, loc=Loc, scale=Scale)  # Opposite shape

def pgev(yvals, Loc, Scale, Shape, log=False):
    if log:
        return genextreme.logcdf(yvals, c=-Shape, loc=Loc, scale=Scale)  # Opposite shape
    else:
        return genextreme.cdf(yvals, c=-Shape, loc=Loc, scale=Scale)  # Opposite shape

def qgev(p, Loc, Scale, Shape):
    if type(p).__module__!='numpy':
      p = np.array(p)
    return genextreme.ppf(p, c=-Shape, loc=Loc, scale=Scale)  # Opposite shape

##
## -------------------------------------------------------------------------- ##






## -------------------------------------------------------------------------- ##
## -------------------------------------------------------------------------- ##
## ------------------------ scalemix_likelihoods.R -------------------------- ##
## -------------------------------------------------------------------------- ##
## -------------------------------------------------------------------------- ##



## -------------------------------------------------------------------------- ##
## Transforms observations from a Gaussian scale mixture to a GPD, or vice versa
##

def scalemix_me_2_gev(X, delta, tau_sqd, Loc, Scale, Shape):
    unifs = pmixture_me(X, delta, tau_sqd)
    gevs = qgev(unifs, Loc=Loc, Scale=Scale, Shape=Shape)
    return gevs

def gev_2_scalemix_me(Y, delta, tau_sqd, Loc, Scale, Shape):
    unifs = pgev(Y, Loc, Scale, Shape)
    scalemixes = qmixture_me_interp(unifs, delta, tau_sqd)
    return scalemixes

## After GEV params are updated, the 'cen' should be re-calculated.
def which_censored(Y, Loc, Scale, Shape, prob_below):
    unifs = pgev(Y, Loc, Scale, Shape)
    return unifs<prob_below

## Only calculate the un-censored elements
def X_update(Y, cen, cen_above, delta, tau_sqd, Loc, Scale, Shape):
    X = np.empty(Y.shape)
    X[:] = np.nan
    
    if np.any(~cen & ~cen_above):
        X[~cen & ~cen_above] = gev_2_scalemix_me(Y[~cen & ~cen_above], delta, tau_sqd,
                                    Loc[~cen & ~cen_above], Scale[~cen & ~cen_above], Shape[~cen & ~cen_above])
    return X

##
## -------------------------------------------------------------------------- ##




## -------------------------------------------------------------------------- ##
## The log likelihood of the data, where the data comes from a scale mixture
## of Gaussians, transformed to GPD (matrix/vector input)
##
## NOT ACTUALLY depending on X. X and cen need to be calculated in advance.
##
##

def marg_transform_data_mixture_me_likelihood(Y, X, X_s, cen, cen_above,
                prob_below, prob_above, Loc, Scale, Shape,
                delta, tau_sqd, thresh_X=np.nan, thresh_X_above=np.nan):
  if np.isnan(thresh_X):
     if prob_below==0:
         thresh_X = -np.inf
     else:
         thresh_X = qmixture_me_interp(prob_below, delta = delta, tau_sqd = tau_sqd)
     if prob_above==1:
         thresh_X_above = np.inf
     else:
         thresh_X_above = qmixture_me_interp(prob_above, delta = delta, tau_sqd = tau_sqd)
  sd = np.sqrt(tau_sqd)
  
  ## Initialize space to store the log-likelihoods for each observation:
  ll = np.empty(Y.shape)
  ll[:] = np.nan
  if np.any(cen):
     ll[cen] = norm.logcdf(thresh_X, loc=X_s[cen], scale=sd)
  if np.any(cen_above):
     ll[cen_above] = norm.logsf(thresh_X_above, loc=X_s[cen_above], scale=sd)
  
  if np.any(~cen & ~cen_above):
     # # Sometimes pgev easily becomes 1, which causes the gev_2_scalemix to become nan
     # if np.any(np.isnan(X[~cen])):
     #     return -np.inf
     ll[~cen & ~cen_above] = norm.logpdf(X[~cen & ~cen_above], loc=X_s[~cen & ~cen_above], scale=sd
               )+dgev(Y[~cen & ~cen_above], Loc=Loc[~cen & ~cen_above], Scale=Scale[~cen & ~cen_above], Shape=Shape[~cen & ~cen_above], log=True
               )-np.log(dmixture_me(X[~cen & ~cen_above], delta = delta, tau_sqd = tau_sqd))
     
  #which = np.isnan(ll)
  #if np.any(which):
  #   ll[which] = -np.inf  # Normal density larger order than marginal density of scalemix
  return np.nansum(ll)


## Univariate version
def marg_transform_data_mixture_me_likelihood_uni(Y, X, X_s, cen, cen_above,
                   prob_below, prob_above, Loc, Scale, Shape,
                   delta, tau_sqd, thresh_X=np.nan, thresh_X_above=np.nan):
  if np.isnan(thresh_X):
     if prob_below==0:
         thresh_X = -np.inf
     else:
         thresh_X = qmixture_me_interp(prob_below, delta = delta, tau_sqd = tau_sqd)
     if prob_above==1:
         thresh_X_above = np.inf
     else:
         thresh_X_above = qmixture_me_interp(prob_above, delta = delta, tau_sqd = tau_sqd)
  sd = math.sqrt(tau_sqd)
  
  ll=np.array(np.nan)
  if cen:
     ll = norm.logcdf(thresh_X, loc=X_s, scale=sd)
  elif cen_above:
     ll = norm.logsf(thresh_X_above, loc=X_s, scale=sd)
  else:
     ll = norm.logpdf(X, loc=X_s, scale=sd
        )+dgev(Y, Loc=Loc, Scale=Scale, Shape=Shape, log=True
        )-np.log(dmixture_me(X, delta = delta, tau_sqd = tau_sqd))
  #if np.isnan(ll):
  #   ll = -np.inf
  return np.nansum(ll)


def marg_transform_data_mixture_me_likelihood_interp(Y, X, X_s, cen, cen_above,
                prob_below, prob_above, Loc, Scale, Shape, delta, tau_sqd, 
                xp=np.nan, den_p=np.nan, thresh_X=np.nan, thresh_X_above=np.nan):
  if np.isnan(thresh_X):
     if prob_below==0:
         thresh_X = -np.inf
     else:
         thresh_X = qmixture_me_interp(prob_below, delta = delta, tau_sqd = tau_sqd)
     if prob_above==1:
         thresh_X_above = np.inf
     else:
         thresh_X_above = qmixture_me_interp(prob_above, delta = delta, tau_sqd = tau_sqd)
  if np.any(np.isnan(xp)):
      grid = density_interp_grid(delta, tau_sqd)
      xp = grid[0]; den_p = grid[1]
  sd = np.sqrt(tau_sqd)
  
  ## Initialize space to store the log-likelihoods for each observation:
  ll = np.empty(Y.shape)
  ll[:] = np.nan
  if np.any(cen):
     ll[cen] = norm.logcdf(thresh_X, loc=X_s[cen], scale=sd)
  if np.any(cen_above):
     ll[cen_above] = norm.logsf(thresh_X_above, loc=X_s[cen_above], scale=sd)
  
  if np.any(~cen & ~cen_above):
     ll[~cen & ~cen_above] = norm.logpdf(X[~cen & ~cen_above], loc=X_s[~cen & ~cen_above], scale=sd
               )+dgev(Y[~cen & ~cen_above], Loc=Loc[~cen & ~cen_above], Scale=Scale[~cen & ~cen_above], Shape=Shape[~cen & ~cen_above], log=True
               )-np.log(dmixture_me_interpo(X[~cen & ~cen_above], xp, den_p))
     
  return np.nansum(ll)

##
## -------------------------------------------------------------------------- ##



## -------------------------------------------------------------------------- ##
## Updates the X.s, for the generic Metropolis sampler
## Samples from the scaled Gaussian process (update the smooth process).
## The mixing distribution comes from from the Huser-wadsworth scale mixing distribution.
##
## Update ONE time, and 'thresh_X' is required.

def X_s_likelihood_conditional(X_s, R, V, d):
    tmp = norm.ppf(1-R/X_s)
    if np.any(np.isnan(tmp)):
        return -np.inf
    else:
        part1 = -0.5*eig2inv_quadform_vector(V, 1/d, tmp)-0.5*np.sum(np.log(d))
        part2 = 0.5*np.sum(tmp*tmp)+np.sum(np.log(R)-2*np.log(X_s))
        return part1+part2


def X_s_update_onetime(Y, X, X_s, cen, cen_above, prob_below, prob_above, Loc, Scale, Shape, delta, tau_sqd, thresh_X, thresh_X_above,
                       R, V, d, Sigma_m, random_generator):
    
    n_s = X.size
    prop_X_s = np.empty(X.shape)
    accept = np.zeros(n_s)
    
    log_num=0; log_denom=0 # sd= np.sqrt(tau_sqd)
    for idx, X_s_idx in enumerate(X_s):
        # tripped : X = Y, changing X will change Y as well.
        prop_X_s[:] = X_s
        # temp = X_s(iter)+v_q(iter)*R::rnorm(0,1);
        temp = X_s_idx + Sigma_m[idx]*random_generator.standard_normal(1)
        prop_X_s[idx] = temp
        log_num = marg_transform_data_mixture_me_likelihood_uni(Y[idx], X[idx], prop_X_s[idx],
                       cen[idx], cen_above[idx], prob_below, prob_above, Loc[idx], Scale[idx], Shape[idx], delta, tau_sqd,
                       thresh_X, thresh_X_above) + X_s_likelihood_conditional(prop_X_s, R, V, d);
        log_denom = marg_transform_data_mixture_me_likelihood_uni(Y[idx], X[idx], X_s_idx,
                       cen[idx], cen_above[idx], prob_below, prob_above, Loc[idx], Scale[idx], Shape[idx], delta, tau_sqd,
                       thresh_X, thresh_X_above) + X_s_likelihood_conditional(X_s, R, V, d);
        
        r = np.exp(log_num - log_denom)
        if ~np.isfinite(r):
            r = 0
        if random_generator.uniform(0,1,1)<r:
            X_s[idx] = temp  # changes argument 'X_s' directly
            accept[idx] = accept[idx] + 1
    
    #result = (X_s,accept)
    return accept



def Z_likelihood_conditional_eigen(Z, V, d):
    # R_powered = R**phi
    part1 = -0.5*eig2inv_quadform_vector(V, 1/d, Z)-0.5*np.sum(np.log(d))
    return part1

def Z_likelihood_conditional(Z, Cor, cholesky_inv):
    # R_powered = R**phi
    part1 = -0.5*inv_quadform_vector(Cor, Z, cholesky_inv)-0.5*2*sum(np.log(np.diag(cholesky_inv[0])))
    return part1


def Z_update_onetime(Y, X, R, Z, cen, cen_above, prob_below, prob_above,
                delta, tau_sqd, Loc, Scale, Shape, thresh_X, thresh_X_above,
                Cor_1t, cholesky_inv_1t, nonMissing_1t, Sigma_m, random_generator):
    
    n_s = X.size
    prop_Z = np.empty(X.shape)
    accept = np.zeros(n_s)
    ## Generate X_s
    X_s = (R**(delta/(1-delta)))*norm_to_Pareto(Z)
    
    log_num=0; log_denom=0 # sd= np.sqrt(tau_sqd)
    for Z_idx in nonMissing_1t:
        # tripped : X = Y, changing X will change Y as well.
        prop_Z[:] = Z
        # temp = X_s(iter)+v_q(iter)*R::rnorm(0,1);
        temp = Z[Z_idx] + Sigma_m[Z_idx]*random_generator.standard_normal(1)
        prop_Z[Z_idx] = temp
        prop_X_s_idx = (R**(delta/(1-delta)))*norm_to_Pareto(temp)
        
        log_num = marg_transform_data_mixture_me_likelihood_uni(Y[Z_idx], X[Z_idx], prop_X_s_idx,
                       cen[Z_idx], cen_above[Z_idx], prob_below, prob_above, Loc[Z_idx], Scale[Z_idx], Shape[Z_idx], delta, tau_sqd,
                       thresh_X, thresh_X_above) + Z_likelihood_conditional(prop_Z[nonMissing_1t], Cor_1t, cholesky_inv_1t);
        log_denom = marg_transform_data_mixture_me_likelihood_uni(Y[Z_idx], X[Z_idx], X_s[Z_idx],
                       cen[Z_idx], cen_above[Z_idx], prob_below, prob_above, Loc[Z_idx], Scale[Z_idx], Shape[Z_idx], delta, tau_sqd,
                       thresh_X, thresh_X_above) + Z_likelihood_conditional(Z[nonMissing_1t], Cor_1t, cholesky_inv_1t);
        
        with np.errstate(over='raise'):
            try:
                r = np.exp(log_num - log_denom)  # this gets caught and handled as an exception
            except FloatingPointError:
                # print(" -- idx="+str(idx)+", Z="+str(Z[idx])+", prop_Z="+str(temp)+", log_num="+str(log_num)+", log_denom="+str(log_denom))
                r=1
    
        if random_generator.uniform(0,1,1)<r:
            Z[Z_idx] = temp  # changes argument 'Z' directly
            X_s[Z_idx] = prop_X_s_idx
            accept[Z_idx] = accept[Z_idx] + 1
    
    #result = (X_s,accept)
    return accept


def Rt_update_mixture_me_likelihood(data, params, X, Z, cen, cen_above,
                prob_below, prob_above, Loc, Scale, Shape, delta, tau_sqd,
                thresh_X, thresh_X_above):
  Y = data
  R = params
    
  if R < 1:
      return -np.inf
  else:
      ## Generate X_s
      X_s = (R**(delta/(1-delta)))*norm_to_Pareto(Z)
      ll = marg_transform_data_mixture_me_likelihood(Y, X, X_s, cen, cen_above,
                prob_below, prob_above, Loc, Scale, Shape, delta, tau_sqd,
                thresh_X, thresh_X_above)
      return ll

def Rt_update_mixture_me_likelihood_interp(data, params, X, Z, cen, cen_above,
                prob_below, prob_above, Loc, Scale, Shape, delta, tau_sqd,
                xp, den_p, thresh_X, thresh_X_above):
  Y = data
  R = params
    
  if R < 1:
      return -np.inf
  else:
      ## Generate X_s
      X_s = (R**(delta/(1-delta)))*norm_to_Pareto(Z)
      ll = marg_transform_data_mixture_me_likelihood_interp(Y, X, X_s, cen, cen_above,
                prob_below, prob_above, Loc, Scale, Shape, delta, tau_sqd,
                xp, den_p, thresh_X, thresh_X_above)
      return ll
  
# def Rt_update_mixture_me_likelihood(data, params, delta, V, d):
#   X_s = data
#   Rt = params
#   if Rt < 1:
#       return -np.inf
#   else:
#       ll = X_s_likelihood_conditional(X_s, Rt, V, d)
#       return ll

##
## -------------------------------------------------------------------------- ##



## -------------------------------------------------------------------------- ##
## For the generic Metropolis sampler
## Samples from the parameters of the mixing distribution, for the scale
## mixture of Gaussians, where the mixing distribution comes from
## Huser-wadsworth scale mixture.
##
##

def delta_update_mixture_me_likelihood(data, params, R, Z, cen, cen_above,
                                       prob_below, prob_above,
                                       Loc, Scale, Shape, tau_sqd):
  Y = data
  delta = params
  if delta < 0 or delta > 1:
      return -np.inf
  
  X = X_update(Y, cen, cen_above, delta, tau_sqd, Loc, Scale, Shape)
  X_s = (R**(delta/(1-delta)))*norm_to_Pareto(Z)
  
  ll = marg_transform_data_mixture_me_likelihood(Y, X, X_s, cen, cen_above, prob_below, prob_above, Loc,
                Scale, Shape, delta, tau_sqd)

  return ll

def delta_update_mixture_me_likelihood_interp(data, params, R, Z, cen, cen_above,
                                       prob_below, prob_above,
                                       Loc, Scale, Shape, tau_sqd):
  Y = data
  delta = params
  if delta < 0 or delta > 1:
      return -np.inf
  
  X = X_update(Y, cen, cen_above, delta, tau_sqd, Loc, Scale, Shape)
  X_s = (R**(delta/(1-delta)))*norm_to_Pareto(Z)
  
  ll = marg_transform_data_mixture_me_likelihood_interp(Y, X, X_s, cen, cen_above, prob_below, prob_above, Loc,
                Scale, Shape, delta, tau_sqd)

  return ll                                                                             
##
## -------------------------------------------------------------------------- ##



##
## -------------------------------------------------------------------------- ##
## For the generic Metropolis sampler
## Samples from the measurement error variance (on the X scale), for the scale
## mixture of Gaussians, where the
## mixing distribution comes from Huser-wadsworth scale mixture.
## Just a wrapper for marg.transform.data.mixture.me.likelihood
##
##   *********** If we do end up updating the prob.below parameter, this
##   *********** is a good place to do it.
##
## data............................... a n.t vector of scaling factors
## params............................. tau_sqd
## Y ................................. a (n.s x n.t) matrix of data that are
##                                     marginally GPD, and conditionally
##                                     independent given X(s)
## X.s ............................... the latent Gaussian process, without the
##                                     measurement error
## cen
## prob.below
## theta.gpd
## delta
##
def tau_update_mixture_me_likelihood(data, params, X_s, cen, cen_above, prob_below, prob_above,
                                   Loc, Scale, Shape, delta):
  Y = data
  tau_sqd = params

  X = X_update(Y, cen, cen_above, delta, tau_sqd, Loc, Scale, Shape)
  
  ll = marg_transform_data_mixture_me_likelihood(Y, X, X_s, cen, cen_above, prob_below, prob_above, Loc,
                                            Scale, Shape, delta, tau_sqd)
  
  return ll

def tau_update_mixture_me_likelihood_interp(data, params, X_s, cen, cen_above, prob_below, prob_above,
                                   Loc, Scale, Shape, delta):
  Y = data
  tau_sqd = params

  X = X_update(Y, cen, cen_above, delta, tau_sqd, Loc, Scale, Shape)
  
  ll = marg_transform_data_mixture_me_likelihood_interp(Y, X, X_s, cen, cen_above, prob_below, prob_above, Loc,
                                            Scale, Shape, delta, tau_sqd)
  
  return ll
##
## -------------------------------------------------------------------------- ##



## -------------------------------------------------------------------------- ##
## Update covariance parameters. For the generic Metropolis sampler
## Samples from the parameters of the underlying Gaussian process, for the scale
## mixture of Gaussians, where the
## mixing distribution comes from Huser-wadsworth scale mixture.
##
##
def theta_c_update_mixture_me_likelihood_eigen(data, params, S, V=np.nan, d=np.nan):
  Z = data
  range = params[0]
  nu = params[1]
  if len(Z.shape)==1:
      Z = Z.reshape((Z.shape[0],1))
  n_t = Z.shape[1]
  
  if np.any(np.isnan(V)):
    Cor = corr_fn(S, np.array([range,nu]))
    eig_Cor = np.linalg.eigh(Cor) #For symmetric matrices
    V = eig_Cor[1]
    d = eig_Cor[0]

  ll = np.empty(n_t)
  ll[:]=np.nan
  for idx in np.arange(n_t):
    ll[idx] = Z_likelihood_conditional(Z[:,idx], V, d)
  return np.sum(ll)


def range_update_mixture_me_likelihood_eigen(data, params, nu, S, V=np.nan, d=np.nan):
  Z = data
  range = params
  
  if len(Z.shape)==1:
      Z = Z.reshape((Z.shape[0],1))
  n_t = Z.shape[1]
  
  if np.any(np.isnan(V)):
    Cor = corr_fn(S, np.array([range,nu]))
    eig_Cor = np.linalg.eigh(Cor) #For symmetric matrices
    V = eig_Cor[1]
    d = eig_Cor[0]

  ll = np.empty(n_t)
  ll[:]=np.nan
  for idx in np.arange(n_t):
    ll[idx] = Z_likelihood_conditional(Z[:,idx], V, d)
  return np.sum(ll)


def range_update_mixture_me_likelihood(data, params, nu, S, Cor=None, cholesky_inv=None):
  Z = data
  range = params
  
  if len(Z.shape)==1:
      Z = Z.reshape((Z.shape[0],1))
  n_t = Z.shape[1]
  
  if Cor is None:
    Cor = corr_fn(S, np.array([range,nu]))
    cholesky_inv = lapack.dposv(Cor,Z[:,0])

  ll = np.empty(n_t)
  ll[:]=np.nan
  for idx in np.arange(n_t):
    ll[idx] = Z_likelihood_conditional(Z[:,idx], Cor, cholesky_inv)
  return np.sum(ll)
##
## -------------------------------------------------------------------------- ##


################################################################################
################################################################################
################################################################################
##                                                                            ##
##                    GEV_params = beta0 + beta1 * x(s)                       ##
##                                                                            ##
################################################################################
################################################################################
################################################################################
## For the generic Metropolis sampler
## Samples from the parameters of the GEV response distribution
##

## For the intercept of the location parameter
def loc0_gev_update_mixture_me_likelihood(data, params, Y, X_s, cen, cen_above, prob_below, prob_above,
                     delta, tau_sqd, loc1, Scale, Shape, Time, thresh_X, thresh_X_above):
  
  ## Design_mat = data
  ## For the time being, assume that the intercept, slope are CONSTANTS
  beta_loc0 = params
  loc0 = data@beta_loc0  # mu = Xb
  
  if len(X_s.shape)==1:
      X_s = X_s.reshape((X_s.shape[0],1))
  n_t = X_s.shape[1]
  n_s = X_s.shape[0]
  Loc = np.tile(loc0, n_t) + np.tile(loc1, n_t)*np.repeat(Time,n_s)
  Loc = Loc.reshape((n_s,n_t),order='F')
  
  max_support = Loc - Scale/Shape
  max_support[Shape>0] = np.inf
  
  tmp=pgev(Y[~cen & ~cen_above], Loc[~cen & ~cen_above], Scale[~cen & ~cen_above], Shape[~cen & ~cen_above])
  
  # If the parameters imply support that is not consistent with the data,
  # then reject the parameters.
  if np.any(Y[~np.isnan(Y)] > max_support[~np.isnan(Y)]) or np.nanmin(tmp)<prob_below-0.001 or np.nanmax(tmp)>prob_above+0.001:
      return -np.inf
  
  X = X_update(Y, cen, cen_above, delta, tau_sqd, Loc, Scale, Shape)
  ll = marg_transform_data_mixture_me_likelihood(Y, X, X_s,  cen, cen_above, prob_below, prob_above,
                                    Loc, Scale, Shape, delta, tau_sqd, thresh_X, thresh_X_above)
  return ll


## For the slope wrt T of the location parameter
def loc1_gev_update_mixture_me_likelihood(data, params, Y, X_s, cen, cen_above, prob_below, prob_above,
                     delta, tau_sqd, loc0, Scale, Shape, Time, thresh_X, thresh_X_above):
  
  ##Design_mat = data
  ## For the time being, assume that the intercept, slope are CONSTANTS
  beta_loc1 = params
  loc1 = data@beta_loc1  # mu = Xb
  
  if len(X_s.shape)==1:
      X_s = X_s.reshape((X_s.shape[0],1))
  n_t = X_s.shape[1]
  n_s = X_s.shape[0]
  Loc = np.tile(loc0, n_t) + np.tile(loc1, n_t)*np.repeat(Time,n_s)
  Loc = Loc.reshape((n_s,n_t),order='F')
  
  max_support = Loc - Scale/Shape
  max_support[Shape>0] = np.inf
  
  tmp=pgev(Y[~cen & ~cen_above], Loc[~cen & ~cen_above], Scale[~cen & ~cen_above], Shape[~cen & ~cen_above])
  
  # If the parameters imply support that is not consistent with the data,
  # then reject the parameters.
  if np.any(Y[~np.isnan(Y)] > max_support[~np.isnan(Y)]) or np.nanmin(tmp)<prob_below-0.001 or np.nanmax(tmp)>prob_above+0.001:
      return -np.inf
  
  X = X_update(Y, cen, cen_above, delta, tau_sqd, Loc, Scale, Shape)
  ll = marg_transform_data_mixture_me_likelihood(Y, X, X_s, cen, cen_above, prob_below, prob_above,
                                    Loc, Scale, Shape, delta, tau_sqd, thresh_X, thresh_X_above)
  return ll


## For the scale parameter
def scale_gev_update_mixture_me_likelihood(data, params, Y, X_s, cen, cen_above, prob_below, prob_above,
                     delta, tau_sqd, Loc, Shape, Time, thresh_X, thresh_X_above):
  
  ## Design_mat = data
  ## For the time being, assume that the intercept, slope are CONSTANTS
  beta_scale = params
  scale = data@beta_scale  # mu = Xb
  if np.any(scale < 0):
      return -np.inf
  
  if len(X_s.shape)==1:
      X_s = X_s.reshape((X_s.shape[0],1))
  n_t = X_s.shape[1]
  n_s = X_s.shape[0]
  Scale = np.tile(scale, n_t)
  Scale = Scale.reshape((n_s,n_t),order='F')
  
  max_support = Loc - Scale/Shape
  max_support[Shape>0] = np.inf
  
  tmp=pgev(Y[~cen & ~cen_above], Loc[~cen & ~cen_above], Scale[~cen & ~cen_above], Shape[~cen & ~cen_above])
  
  # If the parameters imply support that is not consistent with the data,
  # then reject the parameters.
  if np.any(Y[~np.isnan(Y)] > max_support[~np.isnan(Y)]) or np.nanmin(tmp)<prob_below-0.001 or np.nanmax(tmp)>prob_above+0.001:
      return -np.inf
  
  X = X_update(Y, cen, cen_above, delta, tau_sqd, Loc, Scale, Shape)
  ll = marg_transform_data_mixture_me_likelihood(Y, X, X_s, cen, cen_above, prob_below, prob_above,
                                    Loc, Scale, Shape, delta, tau_sqd, thresh_X, thresh_X_above)
  return ll


## For the shape parameter
def shape_gev_update_mixture_me_likelihood(data, params, Y, X_s, cen, cen_above, prob_below, prob_above,
                     delta, tau_sqd, Loc, Scale, Time, thresh_X, thresh_X_above):
  
  ## Design_mat = data
  ## For the time being, assume that the intercept, slope are CONSTANTS
  beta_shape = params
  shape = data@beta_shape  # mu = Xb
  
  if len(X_s.shape)==1:
      X_s = X_s.reshape((X_s.shape[0],1))
  n_t = X_s.shape[1]
  n_s = X_s.shape[0]
  Shape = np.tile(shape, n_t)
  Shape = Shape.reshape((n_s,n_t),order='F')
  
  max_support = Loc - Scale/Shape
  max_support[Shape>0] = np.inf
  
  tmp=pgev(Y[~cen & ~cen_above], Loc[~cen & ~cen_above], Scale[~cen & ~cen_above], Shape[~cen & ~cen_above])
  
  # If the parameters imply support that is not consistent with the data,
  # then reject the parameters.
  if np.any(Y[~np.isnan(Y)] > max_support[~np.isnan(Y)]) or np.nanmin(tmp)<prob_below-0.001 or np.nanmax(tmp)>prob_above+0.001:
      return -np.inf
  
  X = X_update(Y, cen, cen_above, delta, tau_sqd, Loc, Scale, Shape)
  ll = marg_transform_data_mixture_me_likelihood(Y, X, X_s, cen, cen_above, prob_below, prob_above,
                                    Loc, Scale, Shape, delta, tau_sqd, thresh_X, thresh_X_above)
  return ll
  
  
## For all GEV parameter
def gev_update_mixture_me_likelihood(params, data, Y, X_s, cen, cen_above, prob_below, prob_above,
                     delta, tau_sqd, Time, thresh_X, thresh_X_above):
  
  ## Design_mat = data
  ## For the time being, assume that the intercept, slope are CONSTANTS
  beta_loc0 = params[0:2]
  beta_loc1 = params[2:4]
  beta_scale = params[4:6]
  beta_shape = params[6:8]
  
  loc0 = data@beta_loc0
  loc1 = data@beta_loc1
  scale = data@beta_scale
  shape = data@beta_shape  # mu = Xb
  if np.any(scale < 0):
      return -np.inf
  
  if len(X_s.shape)==1:
      X_s = X_s.reshape((X_s.shape[0],1))
  n_t = X_s.shape[1]
  n_s = X_s.shape[0]
  Loc = np.tile(loc0, n_t) + np.tile(loc1, n_t)*np.repeat(Time,n_s)
  Loc = Loc.reshape((n_s,n_t),order='F')
  Scale = np.tile(scale, n_t)
  Scale = Scale.reshape((n_s,n_t),order='F')
  Shape = np.tile(shape, n_t)
  Shape = Shape.reshape((n_s,n_t),order='F')
  
  max_support = Loc - Scale/Shape
  max_support[Shape>0] = np.inf
  
  tmp=pgev(Y[~cen & ~cen_above], Loc[~cen & ~cen_above], Scale[~cen & ~cen_above], Shape[~cen & ~cen_above])
  
  # If the parameters imply support that is not consistent with the data,
  # then reject the parameters.
  if np.any(Y[~np.isnan(Y)] > max_support[~np.isnan(Y)]) or np.nanmin(tmp)<prob_below-0.001 or np.nanmax(tmp)>prob_above+0.001:
      return -np.inf
    
  X = X_update(Y, cen, cen_above, delta, tau_sqd, Loc, Scale, Shape)
  ll = marg_transform_data_mixture_me_likelihood(Y, X, X_s, cen, cen_above, prob_below, prob_above,
                                    Loc, Scale, Shape, delta, tau_sqd, thresh_X, thresh_X_above)
  return ll


##
## -------------------------------------------------------------------------- ##






################################################################################
################################################################################
################################################################################
##                                                                            ##
##           GEV_params = N(beta0 + beta1 * x(s), Cov(theta_c_param))         ##
##                                                                            ##
################################################################################
################################################################################
################################################################################



## -------------------------------------------------------------------------- ##
## ----------------------  beta_param + theta_c_param ----------------------- ##
## -------------------------------------------------------------------------- ##
def cluster_mvn(gev_vector, mean, Cluster_which, Cor_clusters, inv_cluster):
    ll = 0
    for idx, bool_vec in enumerate(Cluster_which):
        current_gev_vector = gev_vector[bool_vec]
        current_mean = mean[bool_vec]
        ll += dmvn(current_gev_vector, Cor_clusters[idx], mean=current_mean, cholesky_inv =inv_cluster[idx])
    return ll
    
def beta_param_update_me_likelihood(data, params, Design_mat, Cluster_which, Cor_clusters, inv_cluster):
    mean = Design_mat @ params
    return cluster_mvn(data, mean, Cluster_which, Cor_clusters, inv_cluster)

from scipy.linalg import cholesky
## Fix the shape parameter
def theta_c_param_updata_me_likelihood(data, param, nu, mean, Cluster_which, S_clusters):
    Cor_clusters=list()
    inv_cluster=list()
    theta_c = np.array([param[0], nu])
    sill = param[1]
    for idx, bool_vec in enumerate(Cluster_which):
        Cor_tmp = corr_fn(S_clusters[idx], theta_c, sill)
        cholesky_inv = (cholesky(Cor_tmp,lower=False),np.repeat(1,Cor_tmp.shape[0]))
        Cor_clusters.append(Cor_tmp)
        inv_cluster.append(cholesky_inv)
    return cluster_mvn(data, mean, Cluster_which, Cor_clusters, inv_cluster)

## Fix the sill parameter
def theta_c_update_mixture_me_likelihood_1t(Z_onetime, params, sill, Cluster_which, S_clusters_nonMissing, nonMissing_1t_cluster):
    if len(Z_onetime.shape)==1:
        Z_onetime = Z_onetime.reshape((Z_onetime.shape[0],1))
    n_clusters = len(S_clusters_nonMissing)
    ll = np.empty(n_clusters)
    ll[:]=np.nan
    for cluster_num in np.arange(n_clusters):
        which = Cluster_which[cluster_num]
        tmp_nonMissing = nonMissing_1t_cluster[cluster_num]
        Cor_tmp = corr_fn(S_clusters_nonMissing[cluster_num], params)
        cholesky_inv = (cholesky(Cor_tmp,lower=False),np.repeat(1,Cor_tmp.shape[0]))
        ll[cluster_num] = dmvn(Z_onetime[which][tmp_nonMissing], Cor_tmp, mean = 0, cholesky_inv = cholesky_inv)
        
    return np.sum(ll)
 

##
## -------------------------------------------------------------------------- ##


## -------------------------------------------------------------------------- ##
## --------------------------  Latent Z process ----------------------------- ##
## -------------------------------------------------------------------------- ##

## Update Z_onetime vector by cluster
## --- Z_onetime: vector with length n_s
## --- Cluster_which: bool list for identifying cluster labels
## --- cluster_num: which cluster to update?
## --- inv_Z_cluster: Cov cholesky matrix list for all clusters
## --- lambda_current_cluster: the random walk variance
def update_Z_1t_one_cluster(Z, Cluster_which, cluster_num, Cor_Z_clusters, inv_Z_cluster, inv_Z_cluster_proposal,
                                 Y, X, R, cen, cen_above, prob_below, prob_above, delta, tau_sqd,
                                 Loc, Scale, Shape, thresh_X, thresh_X_above,
                                 lambda_current_cluster, random_generator):
    # 1. Obtain the current parameters in the chosen cluster
    which = Cluster_which[cluster_num]
    current_params = Z[which]
    n_current_cluster = len(current_params)
    accept = 0
    
    # 2. Propose parameters
    tmp_parmas_star = np.matmul(np.linalg.inv(inv_Z_cluster_proposal[cluster_num][0].T) , current_params) + lambda_current_cluster*random_generator.standard_normal(n_current_cluster)
    params_star = np.matmul(inv_Z_cluster_proposal[cluster_num][0].T , tmp_parmas_star)
    
    # plt.plot(np.arange(n_current_cluster), current_params, np.arange(n_current_cluster),params_star)
    # plt.plot(np.arange(n_current_cluster), np.matmul(np.linalg.inv(inv_Z_cluster[cluster_num][0].T) , current_params) , np.arange(n_current_cluster),tmp_parmas_star)
    
    # 3. Calculate likelihoods
    X_s_which = (R**(delta/(1-delta)))*norm_to_Pareto(current_params)
    prop_X_s_which = (R**(delta/(1-delta)))*norm_to_Pareto(params_star)
    log_num = marg_transform_data_mixture_me_likelihood(Y[which], X[which], prop_X_s_which,
                       cen[which], cen_above[which], prob_below, prob_above, Loc[which], Scale[which], Shape[which], delta, tau_sqd,
                       thresh_X, thresh_X_above)  + dmvn(params_star, Cor_Z_clusters[cluster_num],
                                                         mean=0, cholesky_inv =inv_Z_cluster[cluster_num])
    log_denom = marg_transform_data_mixture_me_likelihood(Y[which], X[which], X_s_which,
                       cen[which], cen_above[which], prob_below, prob_above, Loc[which], Scale[which], Shape[which], delta, tau_sqd,
                       thresh_X, thresh_X_above) + dmvn(current_params, Cor_Z_clusters[cluster_num],
                                                         mean=0, cholesky_inv =inv_Z_cluster[cluster_num])
    
    # 4. Decide whether to update or not
    r = np.exp(log_num - log_denom)
    if ~np.isfinite(r):
        r = 0
    if random_generator.uniform(0,1,1)<r:
        Z[which] = params_star  # changes argument 'Z' directly
        accept = 1
    
    #result = (X_s,accept)
    return accept


def update_Z_1t_one_cluster_interp(Z, Cluster_which, cluster_num, nonMissing_1t_cluster, Cor_Z_clusters_nonMissing,
                                 inv_Z_cluster_nonMissing, inv_Z_cluster_proposal_nonMissing, Y, X, R, cen,
                                 cen_above, prob_below, prob_above, delta, tau_sqd,
                                 Loc, Scale, Shape, xp, den_p, thresh_X, thresh_X_above,
                                 lambda_current_cluster, random_generator):
    # 1. Obtain the current parameters in the chosen cluster
    which = Cluster_which[cluster_num]
    which_nonMissing = nonMissing_1t_cluster[cluster_num]
    current_params = Z[which]
    params_star = np.empty(current_params.shape)
    n_current_cluster = sum(which_nonMissing)
    accept = 0
    
    # 2. Propose parameters
    tmp_parmas_star = np.matmul(np.linalg.inv(inv_Z_cluster_proposal_nonMissing[cluster_num][0].T) , current_params[which_nonMissing]) + lambda_current_cluster*random_generator.standard_normal(n_current_cluster)
    params_star[which_nonMissing] = np.matmul(inv_Z_cluster_proposal_nonMissing[cluster_num][0].T , tmp_parmas_star)
    
    # plt.plot(np.arange(n_current_cluster), current_params, np.arange(n_current_cluster),params_star)
    # plt.plot(np.arange(n_current_cluster), np.matmul(np.linalg.inv(inv_Z_cluster[cluster_num][0].T) , current_params) , np.arange(n_current_cluster),tmp_parmas_star)
    
    # 3. Calculate likelihoods
    X_s_which = (R**(delta/(1-delta)))*norm_to_Pareto(current_params)
    prop_X_s_which = (R**(delta/(1-delta)))*norm_to_Pareto(params_star)
    log_num = marg_transform_data_mixture_me_likelihood_interp(Y[which], X[which], prop_X_s_which,
                       cen[which], cen_above[which], prob_below, prob_above, Loc[which], Scale[which], Shape[which], delta, tau_sqd,
                       xp, den_p, thresh_X, thresh_X_above)  + dmvn(params_star[which_nonMissing], Cor_Z_clusters_nonMissing[cluster_num],
                                                         mean=0, cholesky_inv =inv_Z_cluster_nonMissing[cluster_num])
    log_denom = marg_transform_data_mixture_me_likelihood_interp(Y[which], X[which], X_s_which,
                       cen[which], cen_above[which], prob_below, prob_above, Loc[which], Scale[which], Shape[which], delta, tau_sqd,
                       xp, den_p, thresh_X, thresh_X_above) + dmvn(current_params[which_nonMissing], Cor_Z_clusters_nonMissing[cluster_num],
                                                         mean=0, cholesky_inv =inv_Z_cluster_nonMissing[cluster_num])
    
    # 4. Decide whether to update or not
    r = np.exp(log_num - log_denom)
    if ~np.isfinite(r):
        r = 0
    if random_generator.uniform(0,1,1)<r:
        Z[which] = params_star  # changes argument 'Z' directly
        accept = 1
    
    #result = (X_s,accept)
    return accept
##
## -------------------------------------------------------------------------- ##





## -------------------------------------------------------------------------- ##
## -----------------------------  loc0 vector ------------------------------- ##
## -------------------------------------------------------------------------- ##
## For the full conditionals of loc0 vector
def loc0_vec_gev_update_mixture_me_likelihood(data, params, X_s, cen, cen_above, prob_below, prob_above,
                     delta, tau_sqd, loc1, Scale, Shape, Time, thresh_X, thresh_X_above):
  Y = data
  
  if len(X_s.shape)==1:
      X_s = X_s.reshape((X_s.shape[0],1))
  n_t = X_s.shape[1]
  n_s = X_s.shape[0]
  Loc = np.tile(params, n_t) + np.tile(loc1, n_t)*np.repeat(Time,n_s)
  Loc = Loc.reshape((n_s,n_t),order='F')
  
  max_support = Loc - Scale/Shape
  max_support[Shape>0] = np.inf
  
  tmp=pgev(Y[~cen & ~cen_above], Loc[~cen & ~cen_above], Scale[~cen & ~cen_above], Shape[~cen & ~cen_above])
  
  # If the parameters imply support that is not consistent with the data,
  # then reject the parameters.
  if np.any(Y[~np.isnan(Y)] > max_support[~np.isnan(Y)]) or np.nanmin(tmp)<prob_below-0.001 or np.nanmax(tmp)>prob_above+0.001:
      return -np.inf
  
  X = X_update(Y, cen, cen_above, delta, tau_sqd, Loc, Scale, Shape)
  ll = marg_transform_data_mixture_me_likelihood(Y, X, X_s, cen, cen_above, prob_below, prob_above,
                                    Loc, Scale, Shape, delta, tau_sqd, thresh_X, thresh_X_above)
  return ll

def beta_loc0_vec_gev_update_mixture_me_likelihood_interp(data, params, Design_mat, sbeta_loc0, X_s, cen, cen_above, prob_below, prob_above, delta, tau_sqd, loc1, loc2, loc3, Scale, Shape, WMGHGs, PDSI, ELI_summer_average, xp, den_p, thresh_X, thresh_X_above):
  Y = data
  
  if len(X_s.shape)==1:
      X_s = X_s.reshape((X_s.shape[0],1))
  n_t = X_s.shape[1]
  n_s = X_s.shape[0]
  loc0 = Design_mat @params
  Loc = np.tile(loc0, n_t) + np.tile(loc1, n_t)*np.repeat(WMGHGs,n_s) + np.tile(loc2,
        n_t)*PDSI.flatten(order='F') + np.tile(loc3, n_t)*np.repeat(ELI_summer_average,n_s)
  Loc = Loc.reshape((n_s,n_t),order='F')
  
  max_support = Loc - Scale/Shape
  max_support[Shape>0] = np.inf
  
  tmp=pgev(Y[~cen & ~cen_above], Loc[~cen & ~cen_above], Scale[~cen & ~cen_above], Shape[~cen & ~cen_above])
  
  # If the parameters imply support that is not consistent with the data,
  # then reject the parameters.
  if np.any(Y[~np.isnan(Y)] > max_support[~np.isnan(Y)]) or np.nanmin(tmp)<prob_below-0.001 or np.nanmax(tmp)>prob_above+0.001:
      return -np.inf
  
  X = X_update(Y, cen, cen_above, delta, tau_sqd, Loc, Scale, Shape)
  ll = marg_transform_data_mixture_me_likelihood_interp(Y, X, X_s, cen, cen_above, prob_below, prob_above,
            Loc, Scale, Shape, delta, tau_sqd, xp, den_p, thresh_X, thresh_X_above) + dmvn_diag(
                    params, sbeta_loc0)
  return ll

def mu_loc0_vec_gev_update_mixture_me_likelihood_interp(data, params, Design_mat, beta_loc0, sbeta_loc0, X_s, cen, cen_above, prob_below, prob_above,
                     delta, tau_sqd, loc1, Scale, Shape, Time, xp, den_p, thresh_X, thresh_X_above):
  Y = data
  
  if len(X_s.shape)==1:
      X_s = X_s.reshape((X_s.shape[0],1))
  n_t = X_s.shape[1]
  n_s = X_s.shape[0]
  loc0 = params +Design_mat @beta_loc0
  Loc = np.tile(loc0, n_t) + np.tile(loc1, n_t)*np.repeat(Time,n_s)
  Loc = Loc.reshape((n_s,n_t),order='F')
  
  max_support = Loc - Scale/Shape
  max_support[Shape>0] = np.inf
  
  tmp=pgev(Y[~cen & ~cen_above], Loc[~cen & ~cen_above], Scale[~cen & ~cen_above], Shape[~cen & ~cen_above])
  
  # If the parameters imply support that is not consistent with the data,
  # then reject the parameters.
  if np.any(Y[~np.isnan(Y)] > max_support[~np.isnan(Y)]) or np.nanmin(tmp)<prob_below-0.001 or np.nanmax(tmp)>prob_above+0.001:
      return -np.inf
  
  X = X_update(Y, cen, cen_above, delta, tau_sqd, Loc, Scale, Shape)
  ll = marg_transform_data_mixture_me_likelihood_interp(Y, X, X_s, cen, cen_above, prob_below, prob_above,
            Loc, Scale, Shape, delta, tau_sqd, xp, den_p, thresh_X, thresh_X_above)
  return ll

## Update loc0 vector by cluster
## --- loc0: vector with length n_s
## --- Cluster_which: bool list for identifying cluster labels
## --- cluster_num: which cluster to update?
## --- inv_loc0_cluster: Cov cholesky matrix list for all clusters
## --- lambda_current_cluster: the random walk variance
def update_loc0_GEV_one_cluster(loc0, Cluster_which, cluster_num, Cor_loc0_clusters, inv_loc0_cluster, inv_loc0_cluster_proposal,
                                 Y, X_s, cen, cen_above, prob_below, prob_above, delta, tau_sqd,
                                 loc1, Scale, Shape, Time, thresh_X, thresh_X_above, loc0_mean,
                                 lambda_current_cluster, random_generator):
    # 1. Obtain the current parameters in the chosen cluster
    which = Cluster_which[cluster_num]
    current_params = loc0[which]
    current_mean = loc0_mean[which]
    n_current_cluster = len(current_params)
    accept = 0
    
    # 2. Propose parameters
    tmp_parmas_star = np.matmul(np.linalg.inv(inv_loc0_cluster_proposal[cluster_num][0].T) , current_params) + lambda_current_cluster*random_generator.standard_normal(n_current_cluster)
    params_star = np.matmul(inv_loc0_cluster_proposal[cluster_num][0].T , tmp_parmas_star)
    
    # plt.plot(np.arange(n_current_cluster), current_params, np.arange(n_current_cluster),params_star)
    # plt.plot(np.arange(n_current_cluster), np.matmul(np.linalg.inv(inv_loc0_cluster[cluster_num][0].T) , current_params) , np.arange(n_current_cluster),tmp_parmas_star)
    
    # 3. Calculate likelihoods
    # loc0_star = np.empty(loc0.shape[0]); loc0_star[:] = loc0
    # loc0_star[which] = params_star
    log_num = loc0_vec_gev_update_mixture_me_likelihood(Y[which,:], params_star, X_s[which,:], cen[which,:], cen_above[which,:], prob_below, prob_above,
                         delta, tau_sqd, loc1[which], Scale[which,:], Shape[which,:], Time, thresh_X, thresh_X_above) + dmvn(
                         params_star, Cor_loc0_clusters[cluster_num], mean=current_mean, cholesky_inv =inv_loc0_cluster[cluster_num])
    log_denom = loc0_vec_gev_update_mixture_me_likelihood(Y[which,:], current_params, X_s[which,:], cen[which,:], cen_above[which,:], prob_below, prob_above,
                         delta, tau_sqd, loc1[which], Scale[which,:], Shape[which,:], Time, thresh_X, thresh_X_above) + dmvn(
                         current_params, Cor_loc0_clusters[cluster_num], mean=current_mean, cholesky_inv =inv_loc0_cluster[cluster_num])
    
    # 4. Decide whether to update or not
    r = np.exp(log_num - log_denom)
    if ~np.isfinite(r):
        r = 0
    if random_generator.uniform(0,1,1)<r:
        loc0[which] = params_star  # changes argument 'loc0' directly
        accept = 1
    
    #result = (X_s,accept)
    return accept

def update_beta_loc0_GEV_one_cluster_interp(beta_loc0, Cluster_which, cluster_num, inv_loc0_cluster_proposal,
                                 Design_mat, sbeta_loc0, Y, X_s, cen, cen_above, prob_below, prob_above, delta, tau_sqd,
                                 loc1, loc2, loc3, Scale, Shape, WMGHGs, PDSI, ELI_summer_average, xp, den_p, thresh_X, thresh_X_above,
                                 lambda_current_cluster, random_generator):
    # 1. Obtain the current parameters in the chosen cluster
    which = Cluster_which[cluster_num]
    current_params = beta_loc0[which]
    n_current_cluster = len(current_params)
    accept = 0
    
    # 2. Propose parameters
    tmp_parmas_star = np.matmul(np.linalg.inv(inv_loc0_cluster_proposal[cluster_num][0].T) , current_params) + lambda_current_cluster*random_generator.standard_normal(n_current_cluster)
    params_star = np.matmul(inv_loc0_cluster_proposal[cluster_num][0].T , tmp_parmas_star)
    
    # plt.plot(np.arange(n_current_cluster), current_params, np.arange(n_current_cluster),params_star)
    # plt.plot(np.arange(n_current_cluster), np.matmul(np.linalg.inv(inv_loc0_cluster[cluster_num][0].T) , current_params) , np.arange(n_current_cluster),tmp_parmas_star)
    
    # 3. Calculate likelihoods
    # loc0_star = np.empty(loc0.shape[0]); loc0_star[:] = loc0
    # loc0_star[which] = params_star
    log_num = beta_loc0_vec_gev_update_mixture_me_likelihood_interp(Y, params_star, Design_mat, sbeta_loc0, X_s, cen, cen_above, prob_below, prob_above,
                         delta, tau_sqd, loc1, loc2, loc3, Scale, Shape, WMGHGs, PDSI, ELI_summer_average, xp, den_p, thresh_X, thresh_X_above)
    log_denom = beta_loc0_vec_gev_update_mixture_me_likelihood_interp(Y, current_params, Design_mat, sbeta_loc0, X_s, cen, cen_above, prob_below, prob_above,
                         delta, tau_sqd, loc1, loc2, loc3, Scale, Shape, WMGHGs, PDSI, ELI_summer_average, xp, den_p, thresh_X, thresh_X_above)
    
    # 4. Decide whether to update or not
    r = np.exp(log_num - log_denom)
    if ~np.isfinite(r):
        r = 0
    if random_generator.uniform(0,1,1)<r:
        beta_loc0[which] = params_star  # changes argument 'loc0' directly
        accept = 1
    
    #result = (X_s,accept)
    return accept

##
## -------------------------------------------------------------------------- ##



## -------------------------------------------------------------------------- ##
## -----------------------------  loc1 vector ------------------------------- ##
## -------------------------------------------------------------------------- ##
## For the full conditionals of loc1 vector
def loc1_vec_gev_update_mixture_me_likelihood(data, params, X_s, cen, cen_above, prob_below, prob_above,
                     delta, tau_sqd, loc0, Scale, Shape, Time, thresh_X, thresh_X_above):
  
  Y = data
  
  if len(X_s.shape)==1:
      X_s = X_s.reshape((X_s.shape[0],1))
  n_t = X_s.shape[1]
  n_s = X_s.shape[0]
  Loc = np.tile(loc0, n_t) + np.tile(params, n_t)*np.repeat(Time,n_s)
  Loc = Loc.reshape((n_s,n_t),order='F')
  
  max_support = Loc - Scale/Shape
  max_support[Shape>0] = np.inf
  
  tmp=pgev(Y[~cen & ~cen_above], Loc[~cen & ~cen_above], Scale[~cen & ~cen_above], Shape[~cen & ~cen_above])
  
  # If the parameters imply support that is not consistent with the data,
  # then reject the parameters.
  if np.any(Y[~np.isnan(Y)] > max_support[~np.isnan(Y)]) or np.nanmin(tmp)<prob_below-0.001 or np.nanmax(tmp)>prob_above+0.001:
      return -np.inf
  
  X = X_update(Y, cen, cen_above, delta, tau_sqd, Loc, Scale, Shape)
  ll = marg_transform_data_mixture_me_likelihood(Y, X, X_s, cen, cen_above, prob_below, prob_above,
                                    Loc, Scale, Shape, delta, tau_sqd, thresh_X, thresh_X_above)
  return ll


def beta_loc1_vec_gev_update_mixture_me_likelihood_interp(data, params, Design_mat, sbeta_loc1, X_s, cen, cen_above, prob_below, prob_above, delta, tau_sqd, loc0, loc2, loc3, Scale, Shape, WMGHGs, PDSI, ELI_summer_average, xp, den_p, thresh_X, thresh_X_above):
  
  Y = data
  
  if len(X_s.shape)==1:
      X_s = X_s.reshape((X_s.shape[0],1))
  n_t = X_s.shape[1]
  n_s = X_s.shape[0]
  loc1 = Design_mat @params
  Loc = np.tile(loc0, n_t) + np.tile(loc1, n_t)*np.repeat(WMGHGs,n_s) + np.tile(loc2,
        n_t)*PDSI.flatten(order='F') + np.tile(loc3, n_t)*np.repeat(ELI_summer_average,n_s)
  Loc = Loc.reshape((n_s,n_t),order='F')
  
  max_support = Loc - Scale/Shape
  max_support[Shape>0] = np.inf
  
  tmp=pgev(Y[~cen & ~cen_above], Loc[~cen & ~cen_above], Scale[~cen & ~cen_above], Shape[~cen & ~cen_above])
  
  # If the parameters imply support that is not consistent with the data,
  # then reject the parameters.
  if np.any(Y[~np.isnan(Y)] > max_support[~np.isnan(Y)]) or np.nanmin(tmp)<prob_below-0.001 or np.nanmax(tmp)>prob_above+0.001:
      return -np.inf
  
  X = X_update(Y, cen, cen_above, delta, tau_sqd, Loc, Scale, Shape)
  ll = marg_transform_data_mixture_me_likelihood_interp(Y, X, X_s, cen, cen_above, prob_below, prob_above,
                Loc, Scale, Shape, delta, tau_sqd, xp, den_p, thresh_X, thresh_X_above) + dmvn_diag(
                    params, sbeta_loc1)
  return ll


def beta_loc2_vec_gev_update_mixture_me_likelihood_interp(data, params, Design_mat, sbeta_loc2, X_s, cen, cen_above, prob_below, prob_above, delta, tau_sqd, loc0, loc1, loc3, Scale, Shape, WMGHGs, PDSI, ELI_summer_average, xp, den_p, thresh_X, thresh_X_above):
  
  Y = data
  
  if len(X_s.shape)==1:
      X_s = X_s.reshape((X_s.shape[0],1))
  n_t = X_s.shape[1]
  n_s = X_s.shape[0]
  loc2 = Design_mat @params
  Loc = np.tile(loc0, n_t) + np.tile(loc1, n_t)*np.repeat(WMGHGs,n_s) + np.tile(loc2,
        n_t)*PDSI.flatten(order='F') + np.tile(loc3, n_t)*np.repeat(ELI_summer_average,n_s)
  Loc = Loc.reshape((n_s,n_t),order='F')
  
  max_support = Loc - Scale/Shape
  max_support[Shape>0] = np.inf
  
  tmp=pgev(Y[~cen & ~cen_above], Loc[~cen & ~cen_above], Scale[~cen & ~cen_above], Shape[~cen & ~cen_above])
  
  # If the parameters imply support that is not consistent with the data,
  # then reject the parameters.
  if np.any(Y[~np.isnan(Y)] > max_support[~np.isnan(Y)]) or np.nanmin(tmp)<prob_below-0.001 or np.nanmax(tmp)>prob_above+0.001:
      return -np.inf
  
  X = X_update(Y, cen, cen_above, delta, tau_sqd, Loc, Scale, Shape)
  ll = marg_transform_data_mixture_me_likelihood_interp(Y, X, X_s, cen, cen_above, prob_below, prob_above,
                Loc, Scale, Shape, delta, tau_sqd, xp, den_p, thresh_X, thresh_X_above) + dmvn_diag(
                    params, sbeta_loc2)
  return ll


def beta_loc3_vec_gev_update_mixture_me_likelihood_interp(data, params, Design_mat, sbeta_loc3, X_s, cen, cen_above, prob_below, prob_above, delta, tau_sqd, loc0, loc1, loc2, Scale, Shape, WMGHGs, PDSI, ELI_summer_average, xp, den_p, thresh_X, thresh_X_above):
  
  Y = data
  
  if len(X_s.shape)==1:
      X_s = X_s.reshape((X_s.shape[0],1))
  n_t = X_s.shape[1]
  n_s = X_s.shape[0]
  loc3 = Design_mat @params
  Loc = np.tile(loc0, n_t) + np.tile(loc1, n_t)*np.repeat(WMGHGs,n_s) + np.tile(loc2,
        n_t)*PDSI.flatten(order='F') + np.tile(loc3, n_t)*np.repeat(ELI_summer_average,n_s)
  Loc = Loc.reshape((n_s,n_t),order='F')
  
  max_support = Loc - Scale/Shape
  max_support[Shape>0] = np.inf
  
  tmp=pgev(Y[~cen & ~cen_above], Loc[~cen & ~cen_above], Scale[~cen & ~cen_above], Shape[~cen & ~cen_above])
  
  # If the parameters imply support that is not consistent with the data,
  # then reject the parameters.
  if np.any(Y[~np.isnan(Y)] > max_support[~np.isnan(Y)]) or np.nanmin(tmp)<prob_below-0.001 or np.nanmax(tmp)>prob_above+0.001:
      return -np.inf
  
  X = X_update(Y, cen, cen_above, delta, tau_sqd, Loc, Scale, Shape)
  ll = marg_transform_data_mixture_me_likelihood_interp(Y, X, X_s, cen, cen_above, prob_below, prob_above,
                Loc, Scale, Shape, delta, tau_sqd, xp, den_p, thresh_X, thresh_X_above) + dmvn_diag(
                    params, sbeta_loc3)
  return ll

def mu_loc1_vec_gev_update_mixture_me_likelihood_interp(data, params, Design_mat, beta_loc1, sbeta_loc1, X_s, cen, cen_above, prob_below, prob_above,
                     delta, tau_sqd, loc0, Scale, Shape, Time, xp, den_p, thresh_X, thresh_X_above):
  
  Y = data
  
  if len(X_s.shape)==1:
      X_s = X_s.reshape((X_s.shape[0],1))
  n_t = X_s.shape[1]
  n_s = X_s.shape[0]
  loc1 = params +Design_mat @beta_loc1
  Loc = np.tile(loc0, n_t) + np.tile(loc1, n_t)*np.repeat(Time,n_s)
  Loc = Loc.reshape((n_s,n_t),order='F')
  
  max_support = Loc - Scale/Shape
  max_support[Shape>0] = np.inf
  
  tmp=pgev(Y[~cen & ~cen_above], Loc[~cen & ~cen_above], Scale[~cen & ~cen_above], Shape[~cen & ~cen_above])
  
  # If the parameters imply support that is not consistent with the data,
  # then reject the parameters.
  if np.any(Y[~np.isnan(Y)] > max_support[~np.isnan(Y)]) or np.nanmin(tmp)<prob_below-0.001 or np.nanmax(tmp)>prob_above+0.001:
      return -np.inf
  
  X = X_update(Y, cen, cen_above, delta, tau_sqd, Loc, Scale, Shape)
  ll = marg_transform_data_mixture_me_likelihood_interp(Y, X, X_s, cen, cen_above, prob_below, prob_above,
                Loc, Scale, Shape, delta, tau_sqd, xp, den_p, thresh_X, thresh_X_above)
  return ll


## Update loc1 vector by cluster
## --- loc1: vector with length n_s
## --- Cluster_which: bool list for identifying cluster labels
## --- cluster_num: which cluster to update?
## --- inv_loc1_cluster: Cov cholesky matrix list for all clusters
## --- lambda_current_cluster: the random walk variance
def update_loc1_GEV_one_cluster(loc1, Cluster_which, cluster_num, Cor_loc1_clusters, inv_loc1_cluster, inv_loc1_cluster_proposal,
                                 Y, X_s, cen, cen_above, prob_below, prob_above, delta, tau_sqd,
                                 loc0, Scale, Shape, Time, thresh_X, thresh_X_above, loc1_mean,
                                 lambda_current_cluster, random_generator):
    # 1. Obtain the current parameters in the chosen cluster
    which = Cluster_which[cluster_num]
    current_params = loc1[which]
    current_mean = loc1_mean[which]
    n_current_cluster = len(current_params)
    accept = 0
    
    # 2. Propose parameters
    tmp_parmas_star = np.matmul(np.linalg.inv(inv_loc1_cluster_proposal[cluster_num][0].T) , current_params) + lambda_current_cluster*random_generator.standard_normal(n_current_cluster)
    params_star = np.matmul(inv_loc1_cluster_proposal[cluster_num][0].T , tmp_parmas_star)
    
    # plt.plot(np.arange(n_current_cluster), current_params, np.arange(n_current_cluster),params_star)
    # plt.plot(np.arange(n_current_cluster), np.matmul(np.linalg.inv(inv_loc1_cluster[cluster_num][0].T) , current_params) , np.arange(n_current_cluster),tmp_parmas_star)
    
    # 3. Calculate likelihoods
    # loc1_star = np.empty(loc1.shape[0]); loc1_star[:] = loc1
    # loc1_star[which] = params_star
    log_num = loc1_vec_gev_update_mixture_me_likelihood(Y[which,:], params_star, X_s[which,:], cen[which,:], cen_above[which,:], prob_below, prob_above,
                         delta, tau_sqd, loc0[which], Scale[which,:], Shape[which,:], Time, thresh_X, thresh_X_above) + dmvn(
                         params_star, Cor_loc1_clusters[cluster_num], mean=current_mean, cholesky_inv =inv_loc1_cluster[cluster_num])
    log_denom = loc1_vec_gev_update_mixture_me_likelihood(Y[which,:], current_params, X_s[which,:], cen[which,:], cen_above[which,:], prob_below, prob_above,
                         delta, tau_sqd, loc0[which], Scale[which,:], Shape[which,:], Time, thresh_X, thresh_X_above) + dmvn(
                         current_params, Cor_loc1_clusters[cluster_num], mean=current_mean, cholesky_inv =inv_loc1_cluster[cluster_num])

    # 4. Decide whether to update or not
    r = np.exp(log_num - log_denom)
    if ~np.isfinite(r):
        r = 0
    if random_generator.uniform(0,1,1)<r:
        loc1[which] = params_star  # changes argument 'loc1' directly
        accept = 1
    
    #result = (X_s,accept)
    return accept

def update_beta_loc1_GEV_one_cluster_interp(beta_loc1, Cluster_which, cluster_num, inv_loc1_cluster_proposal,
                                 Design_mat, sbeta_loc1, Y, X_s, cen, cen_above, prob_below, prob_above, delta, tau_sqd,
                                 loc0, loc2, loc3, Scale, Shape, WMGHGs, PDSI, ELI_summer_average, xp, den_p, thresh_X, thresh_X_above,
                                 lambda_current_cluster, random_generator):
    # 1. Obtain the current parameters in the chosen cluster
    which = Cluster_which[cluster_num]
    current_params = beta_loc1[which]
    n_current_cluster = len(current_params)
    accept = 0
    
    # 2. Propose parameters
    tmp_parmas_star = np.matmul(np.linalg.inv(inv_loc1_cluster_proposal[cluster_num][0].T) , current_params) + lambda_current_cluster*random_generator.standard_normal(n_current_cluster)
    params_star = np.matmul(inv_loc1_cluster_proposal[cluster_num][0].T , tmp_parmas_star)
    
    # 3. Calculate likelihoods
    log_num = beta_loc1_vec_gev_update_mixture_me_likelihood_interp(Y, params_star, Design_mat, sbeta_loc1, X_s, cen, cen_above, prob_below, prob_above,
                         delta, tau_sqd, loc0, loc2, loc3, Scale, Shape, WMGHGs, PDSI, ELI_summer_average, xp, den_p, thresh_X, thresh_X_above)
    log_denom = beta_loc1_vec_gev_update_mixture_me_likelihood_interp(Y, current_params, Design_mat, sbeta_loc1, X_s, cen, cen_above, prob_below, prob_above,
                         delta, tau_sqd, loc0, loc2, loc3, Scale, Shape, WMGHGs, PDSI, ELI_summer_average, xp, den_p, thresh_X, thresh_X_above)
    
    # 4. Decide whether to update or not
    r = np.exp(log_num - log_denom)
    if ~np.isfinite(r):
        r = 0
    if random_generator.uniform(0,1,1)<r:
        beta_loc1[which] = params_star  # changes argument 'loc1' directly
        accept = 1
    
    #result = (X_s,accept)
    return accept


def update_beta_loc2_GEV_one_cluster_interp(beta_loc2, Cluster_which, cluster_num, inv_loc2_cluster_proposal,
                                 Design_mat, sbeta_loc2, Y, X_s, cen, cen_above, prob_below, prob_above, delta, tau_sqd,
                                 loc0, loc1, loc3, Scale, Shape, WMGHGs, PDSI, ELI_summer_average, xp, den_p, thresh_X, thresh_X_above,
                                 lambda_current_cluster, random_generator):
    # 1. Obtain the current parameters in the chosen cluster
    which = Cluster_which[cluster_num]
    current_params = beta_loc2[which]
    n_current_cluster = len(current_params)
    accept = 0
    
    # 2. Propose parameters
    tmp_parmas_star = np.matmul(np.linalg.inv(inv_loc2_cluster_proposal[cluster_num][0].T) , current_params) + lambda_current_cluster*random_generator.standard_normal(n_current_cluster)
    params_star = np.matmul(inv_loc2_cluster_proposal[cluster_num][0].T , tmp_parmas_star)
    
    # 3. Calculate likelihoods
    log_num = beta_loc2_vec_gev_update_mixture_me_likelihood_interp(Y, params_star, Design_mat, sbeta_loc2, X_s, cen, cen_above, prob_below, prob_above,
                         delta, tau_sqd, loc0, loc1, loc3, Scale, Shape, WMGHGs, PDSI, ELI_summer_average, xp, den_p, thresh_X, thresh_X_above)
    log_denom = beta_loc2_vec_gev_update_mixture_me_likelihood_interp(Y, current_params, Design_mat, sbeta_loc2, X_s, cen, cen_above, prob_below, prob_above,
                         delta, tau_sqd, loc0, loc1, loc3, Scale, Shape, WMGHGs, PDSI, ELI_summer_average, xp, den_p, thresh_X, thresh_X_above)
    
    # 4. Decide whether to update or not
    r = np.exp(log_num - log_denom)
    if ~np.isfinite(r):
        r = 0
    if random_generator.uniform(0,1,1)<r:
        beta_loc2[which] = params_star  # changes argument 'loc2' directly
        accept = 1
    
    #result = (X_s,accept)
    return accept



def update_beta_loc3_GEV_one_cluster_interp(beta_loc3, Cluster_which, cluster_num, inv_loc3_cluster_proposal,
                                 Design_mat, sbeta_loc3, Y, X_s, cen, cen_above, prob_below, prob_above, delta, tau_sqd,
                                 loc0, loc1, loc2, Scale, Shape, WMGHGs, PDSI, ELI_summer_average, xp, den_p, thresh_X, thresh_X_above,
                                 lambda_current_cluster, random_generator):
    # 1. Obtain the current parameters in the chosen cluster
    which = Cluster_which[cluster_num]
    current_params = beta_loc3[which]
    n_current_cluster = len(current_params)
    accept = 0
    
    # 2. Propose parameters
    tmp_parmas_star = np.matmul(np.linalg.inv(inv_loc3_cluster_proposal[cluster_num][0].T) , current_params) + lambda_current_cluster*random_generator.standard_normal(n_current_cluster)
    params_star = np.matmul(inv_loc3_cluster_proposal[cluster_num][0].T , tmp_parmas_star)
    
    # 3. Calculate likelihoods
    log_num = beta_loc3_vec_gev_update_mixture_me_likelihood_interp(Y, params_star, Design_mat, sbeta_loc3, X_s, cen, cen_above, prob_below, prob_above,
                         delta, tau_sqd, loc0, loc1, loc2, Scale, Shape, WMGHGs, PDSI, ELI_summer_average, xp, den_p, thresh_X, thresh_X_above)
    log_denom = beta_loc3_vec_gev_update_mixture_me_likelihood_interp(Y, current_params, Design_mat, sbeta_loc3, X_s, cen, cen_above, prob_below, prob_above,
                         delta, tau_sqd, loc0, loc1, loc2, Scale, Shape, WMGHGs, PDSI, ELI_summer_average, xp, den_p, thresh_X, thresh_X_above)
    
    # 4. Decide whether to update or not
    r = np.exp(log_num - log_denom)
    if ~np.isfinite(r):
        r = 0
    if random_generator.uniform(0,1,1)<r:
        beta_loc3[which] = params_star  # changes argument 'loc3' directly
        accept = 1
    
    #result = (X_s,accept)
    return accept

##
## -------------------------------------------------------------------------- ##




## -------------------------------------------------------------------------- ##
## -----------------------------  scale vector ------------------------------ ##
## -------------------------------------------------------------------------- ##
## For the full conditionals of scale vector
def scale_vec_gev_update_mixture_me_likelihood(data, params, X_s, cen, cen_above, prob_below, prob_above,
                     delta, tau_sqd, Loc, Shape, Time, thresh_X, thresh_X_above):
  
  Y = data
  
  if len(X_s.shape)==1:
      X_s = X_s.reshape((X_s.shape[0],1))
  n_t = X_s.shape[1]
  n_s = X_s.shape[0]
  Scale = np.tile(params, n_t)
  Scale = Scale.reshape((n_s,n_t),order='F')
  
  max_support = Loc - Scale/Shape
  max_support[Shape>0] = np.inf
  
  tmp=pgev(Y[~cen & ~cen_above], Loc[~cen & ~cen_above], Scale[~cen & ~cen_above], Shape[~cen & ~cen_above])
  
  # If the parameters imply support that is not consistent with the data,
  # then reject the parameters.
  if np.any(Y[~np.isnan(Y)] > max_support[~np.isnan(Y)]) or np.nanmin(tmp)<prob_below-0.001 or np.nanmax(tmp)>prob_above+0.001:
      return -np.inf
  
  X = X_update(Y, cen, cen_above, delta, tau_sqd, Loc, Scale, Shape)
  ll = marg_transform_data_mixture_me_likelihood(Y, X, X_s, cen, cen_above, prob_below, prob_above,
                                    Loc, Scale, Shape, delta, tau_sqd, thresh_X, thresh_X_above)
  return ll


def beta_logscale0_vec_gev_update_mixture_me_likelihood_interp(data, params, Design_mat, sbeta_logscale0, X_s, cen, cen_above, prob_below, prob_above,
                     delta, tau_sqd, logscale1, logscale2, Loc, Shape, WMGHGs, ELI_summer_average, xp, den_p, thresh_X, thresh_X_above):
  
  Y = data
  
  if len(X_s.shape)==1:
      X_s = X_s.reshape((X_s.shape[0],1))
  n_t = X_s.shape[1]
  n_s = X_s.shape[0]
  logscale0 = Design_mat @params
  Scale = np.exp(np.tile(logscale0, n_t) + np.tile(logscale1, n_t)*np.repeat(WMGHGs,
                        n_s)  + np.tile(logscale2, n_t)*np.repeat(ELI_summer_average,n_s))
  Scale = Scale.reshape((n_s,n_t),order='F')
  
  max_support = Loc - Scale/Shape
  max_support[Shape>0] = np.inf
  
  tmp=pgev(Y[~cen & ~cen_above], Loc[~cen & ~cen_above], Scale[~cen & ~cen_above], Shape[~cen & ~cen_above])
  
  # If the parameters imply support that is not consistent with the data,
  # then reject the parameters.
  if np.any(Y[~np.isnan(Y)] > max_support[~np.isnan(Y)]) or np.nanmin(tmp)<prob_below-0.001 or np.nanmax(tmp)>prob_above+0.001:
      return -np.inf
  
  X = X_update(Y, cen, cen_above, delta, tau_sqd, Loc, Scale, Shape)
  ll = marg_transform_data_mixture_me_likelihood_interp(Y, X, X_s, cen, cen_above, prob_below, prob_above,
                Loc, Scale, Shape, delta, tau_sqd, xp, den_p, thresh_X, thresh_X_above)+ dmvn_diag(
                    params, sbeta_logscale0)
  return ll



def beta_logscale1_vec_gev_update_mixture_me_likelihood_interp(data, params, Design_mat, sbeta_logscale1, X_s, cen, cen_above, prob_below, prob_above,
                     delta, tau_sqd, logscale0, logscale2, Loc, Shape, WMGHGs, ELI_summer_average, xp, den_p, thresh_X, thresh_X_above):
  
  Y = data
  
  if len(X_s.shape)==1:
      X_s = X_s.reshape((X_s.shape[0],1))
  n_t = X_s.shape[1]
  n_s = X_s.shape[0]
  logscale1 = Design_mat @params
  Scale = np.exp(np.tile(logscale0, n_t) + np.tile(logscale1, n_t)*np.repeat(WMGHGs,
                        n_s)  + np.tile(logscale2, n_t)*np.repeat(ELI_summer_average,n_s))
  Scale = Scale.reshape((n_s,n_t),order='F')
  
  max_support = Loc - Scale/Shape
  max_support[Shape>0] = np.inf
  
  tmp=pgev(Y[~cen & ~cen_above], Loc[~cen & ~cen_above], Scale[~cen & ~cen_above], Shape[~cen & ~cen_above])
  
  # If the parameters imply support that is not consistent with the data,
  # then reject the parameters.
  if np.any(Y[~np.isnan(Y)] > max_support[~np.isnan(Y)]) or np.nanmin(tmp)<prob_below-0.001 or np.nanmax(tmp)>prob_above+0.001:
      return -np.inf
  
  X = X_update(Y, cen, cen_above, delta, tau_sqd, Loc, Scale, Shape)
  ll = marg_transform_data_mixture_me_likelihood_interp(Y, X, X_s, cen, cen_above, prob_below, prob_above,
                Loc, Scale, Shape, delta, tau_sqd, xp, den_p, thresh_X, thresh_X_above)+ dmvn_diag(
                    params, sbeta_logscale1)
  return ll


def beta_logscale2_vec_gev_update_mixture_me_likelihood_interp(data, params, Design_mat, sbeta_logscale2, X_s, cen, cen_above, prob_below, prob_above,
                     delta, tau_sqd, logscale0, logscale1, Loc, Shape, WMGHGs, ELI_summer_average, xp, den_p, thresh_X, thresh_X_above):
  
  Y = data
  
  if len(X_s.shape)==1:
      X_s = X_s.reshape((X_s.shape[0],1))
  n_t = X_s.shape[1]
  n_s = X_s.shape[0]
  logscale2 = Design_mat @params
  Scale = np.exp(np.tile(logscale0, n_t) + np.tile(logscale1, n_t)*np.repeat(WMGHGs,
                        n_s)  + np.tile(logscale2, n_t)*np.repeat(ELI_summer_average,n_s))
  Scale = Scale.reshape((n_s,n_t),order='F')
  
  max_support = Loc - Scale/Shape
  max_support[Shape>0] = np.inf
  
  tmp=pgev(Y[~cen & ~cen_above], Loc[~cen & ~cen_above], Scale[~cen & ~cen_above], Shape[~cen & ~cen_above])
  
  # If the parameters imply support that is not consistent with the data,
  # then reject the parameters.
  if np.any(Y[~np.isnan(Y)] > max_support[~np.isnan(Y)]) or np.nanmin(tmp)<prob_below-0.001 or np.nanmax(tmp)>prob_above+0.001:
      return -np.inf
  
  X = X_update(Y, cen, cen_above, delta, tau_sqd, Loc, Scale, Shape)
  ll = marg_transform_data_mixture_me_likelihood_interp(Y, X, X_s, cen, cen_above, prob_below, prob_above,
                Loc, Scale, Shape, delta, tau_sqd, xp, den_p, thresh_X, thresh_X_above)+ dmvn_diag(
                    params, sbeta_logscale2)
  return ll


def mu_scale_vec_gev_update_mixture_me_likelihood_interp(data, params, Design_mat, beta_scale, sbeta_scale, X_s, cen, cen_above, prob_below, prob_above,
                     delta, tau_sqd, Loc, Shape, Time, xp, den_p, thresh_X, thresh_X_above):
  
  Y = data
  
  if len(X_s.shape)==1:
      X_s = X_s.reshape((X_s.shape[0],1))
  n_t = X_s.shape[1]
  n_s = X_s.shape[0]
  scale = np.exp(params+Design_mat @beta_scale)
  Scale = np.tile(scale, n_t)
  Scale = Scale.reshape((n_s,n_t),order='F')
  
  max_support = Loc - Scale/Shape
  max_support[Shape>0] = np.inf
  
  tmp=pgev(Y[~cen & ~cen_above], Loc[~cen & ~cen_above], Scale[~cen & ~cen_above], Shape[~cen & ~cen_above])
  
  # If the parameters imply support that is not consistent with the data,
  # then reject the parameters.
  if np.any(Y[~np.isnan(Y)] > max_support[~np.isnan(Y)]) or np.nanmin(tmp)<prob_below-0.001 or np.nanmax(tmp)>prob_above+0.001:
      return -np.inf
  
  X = X_update(Y, cen, cen_above, delta, tau_sqd, Loc, Scale, Shape)
  ll = marg_transform_data_mixture_me_likelihood_interp(Y, X, X_s, cen, cen_above, prob_below, prob_above,
                Loc, Scale, Shape, delta, tau_sqd, xp, den_p, thresh_X, thresh_X_above)
  return ll

## Update scale vector by cluster
## --- scale: vector with length n_s
## --- Cluster_which: bool list for identifying cluster labels
## --- cluster_num: which cluster to update?
## --- inv_scale_cluster: Cov cholesky matrix list for all clusters
## --- lambda_current_cluster: the random walk variance
def update_scale_GEV_one_cluster(scale, Cluster_which, cluster_num, Cor_scale_clusters, inv_scale_cluster, inv_scale_cluster_proposal,
                                 Y, X_s, cen, cen_above, prob_below, prob_above, delta, tau_sqd,
                                 Loc, Shape, Time, thresh_X, thresh_X_above, scale_mean,
                                 lambda_current_cluster, random_generator):
    # 1. Obtain the current parameters in the chosen cluster
    which = Cluster_which[cluster_num]
    current_params = scale[which]
    current_mean = scale_mean[which]
    n_current_cluster = len(current_params)
    accept = 0
    
    # 2. Propose parameters
    tmp_parmas_star = np.matmul(np.linalg.inv(inv_scale_cluster_proposal[cluster_num][0].T) , np.log(current_params)) + lambda_current_cluster*random_generator.standard_normal(n_current_cluster)
    log_params_star = np.matmul(inv_scale_cluster_proposal[cluster_num][0].T , tmp_parmas_star)
    
    # plt.plot(np.arange(n_current_cluster), current_params, np.arange(n_current_cluster),params_star)
    # plt.plot(np.arange(n_current_cluster), np.matmul(np.linalg.inv(inv_scale_cluster[cluster_num][0].T) , current_params) , np.arange(n_current_cluster),tmp_parmas_star)
    
    # 3. Calculate likelihoods
    # scale_star = np.empty(scale.shape[0]); scale_star[:] = scale
    # scale_star[which] = np.exp(log_params_star)
    log_num = scale_vec_gev_update_mixture_me_likelihood(Y[which,:], np.exp(log_params_star), X_s[which,:], cen[which,:], cen_above[which,:], prob_below, prob_above,
                         delta, tau_sqd, Loc[which,:], Shape[which,:], Time, thresh_X, thresh_X_above) + dmvn(
                         log_params_star, Cor_scale_clusters[cluster_num], mean=current_mean, cholesky_inv =inv_scale_cluster[cluster_num])
    log_denom = scale_vec_gev_update_mixture_me_likelihood(Y[which,:], current_params, X_s[which,:], cen[which,:], cen_above[which,:], prob_below, prob_above,
                         delta, tau_sqd, Loc[which,:], Shape[which,:], Time, thresh_X, thresh_X_above) + dmvn(
                         np.log(current_params), Cor_scale_clusters[cluster_num], mean=current_mean, cholesky_inv =inv_scale_cluster[cluster_num])
    
    # 4. Decide whether to update or not
    r = np.exp(log_num - log_denom)
    if ~np.isfinite(r):
        r = 0
    if random_generator.uniform(0,1,1)<r:
        scale[which] = np.exp(log_params_star)  # changes argument 'scale' directly
        accept = 1
    
    #result = (X_s,accept)
    return accept


def update_beta_logscale0_GEV_one_cluster_interp(beta_logscale0, Cluster_which, cluster_num,
     inv_logscale0_cluster_proposal, Design_mat, sbeta_logscale0, Y, X_s, cen, cen_above, prob_below, prob_above, delta,
     tau_sqd, logscale1, logscale2, Loc, Shape, WMGHGs, ELI_summer_average, xp, den_p, thresh_X, thresh_X_above, lambda_current_cluster, random_generator):
    # 1. Obtain the current parameters in the chosen cluster
    which = Cluster_which[cluster_num]
    current_params = beta_logscale0[which]
    n_current_cluster = len(current_params)
    accept = 0
    
    # 2. Propose parameters
    tmp_parmas_star = np.matmul(np.linalg.inv(inv_logscale0_cluster_proposal[cluster_num][0].T) , current_params) + lambda_current_cluster*random_generator.standard_normal(n_current_cluster)
    params_star = np.matmul(inv_logscale0_cluster_proposal[cluster_num][0].T , tmp_parmas_star)
    
    
    # 3. Calculate likelihoods
    beta_logscale0_star = np.empty(beta_logscale0.shape[0]); beta_logscale0_star[:] = beta_logscale0
    beta_logscale0_star[which] = params_star
    log_num = beta_logscale0_vec_gev_update_mixture_me_likelihood_interp(Y, beta_logscale0_star, Design_mat,  sbeta_logscale0, X_s, cen, cen_above, prob_below, prob_above,
                         delta, tau_sqd, logscale1, logscale2, Loc, Shape, WMGHGs, ELI_summer_average, xp, den_p, thresh_X, thresh_X_above)
    log_denom = beta_logscale0_vec_gev_update_mixture_me_likelihood_interp(Y, beta_logscale0, Design_mat,
        sbeta_logscale0, X_s, cen, cen_above, prob_below, prob_above,
                         delta, tau_sqd, logscale1, logscale2, Loc, Shape, WMGHGs, ELI_summer_average, xp, den_p, thresh_X, thresh_X_above)
    
    # 4. Decide whether to update or not
    r = np.exp(log_num - log_denom)
    if ~np.isfinite(r):
        r = 0
    if random_generator.uniform(0,1,1)<r:
        beta_logscale0[which] = params_star  # changes argument 'logscale0' directly
        accept = 1
    
    #result = (X_s,accept)
    return accept



def update_beta_logscale1_GEV_one_cluster_interp(beta_logscale1, Cluster_which, cluster_num,
     inv_logscale1_cluster_proposal, Design_mat, sbeta_logscale1, Y, X_s, cen, cen_above, prob_below, prob_above, delta,
     tau_sqd, logscale0, logscale2, Loc, Shape, WMGHGs, ELI_summer_average, xp, den_p, thresh_X, thresh_X_above, lambda_current_cluster, random_generator):
    # 1. Obtain the current parameters in the chosen cluster
    which = Cluster_which[cluster_num]
    current_params = beta_logscale1[which]
    n_current_cluster = len(current_params)
    accept = 0
    
    # 2. Propose parameters
    tmp_parmas_star = np.matmul(np.linalg.inv(inv_logscale1_cluster_proposal[cluster_num][0].T) , current_params) + lambda_current_cluster*random_generator.standard_normal(n_current_cluster)
    params_star = np.matmul(inv_logscale1_cluster_proposal[cluster_num][0].T , tmp_parmas_star)
    
    
    # 3. Calculate likelihoods
    beta_logscale1_star = np.empty(beta_logscale1.shape[0]); beta_logscale1_star[:] = beta_logscale1
    beta_logscale1_star[which] = params_star
    log_num = beta_logscale1_vec_gev_update_mixture_me_likelihood_interp(Y, beta_logscale1_star, Design_mat,  sbeta_logscale1, X_s, cen, cen_above, prob_below, prob_above,
                         delta, tau_sqd, logscale0, logscale2, Loc, Shape, WMGHGs, ELI_summer_average, xp, den_p, thresh_X, thresh_X_above)
    log_denom = beta_logscale1_vec_gev_update_mixture_me_likelihood_interp(Y, beta_logscale1, Design_mat,
        sbeta_logscale1, X_s, cen, cen_above, prob_below, prob_above,
                         delta, tau_sqd, logscale0, logscale2, Loc, Shape, WMGHGs, ELI_summer_average, xp, den_p, thresh_X, thresh_X_above)
    
    # 4. Decide whether to update or not
    r = np.exp(log_num - log_denom)
    if ~np.isfinite(r):
        r = 0
    if random_generator.uniform(0,1,1)<r:
        beta_logscale1[which] = params_star  # changes argument 'logscale1' directly
        accept = 1
    
    #result = (X_s,accept)
    return accept



def update_beta_logscale2_GEV_one_cluster_interp(beta_logscale2, Cluster_which, cluster_num,
     inv_logscale2_cluster_proposal, Design_mat, sbeta_logscale2, Y, X_s, cen, cen_above, prob_below, prob_above, delta,
     tau_sqd, logscale0, logscale1, Loc, Shape, WMGHGs, ELI_summer_average, xp, den_p, thresh_X, thresh_X_above, lambda_current_cluster, random_generator):
    # 1. Obtain the current parameters in the chosen cluster
    which = Cluster_which[cluster_num]
    current_params = beta_logscale2[which]
    n_current_cluster = len(current_params)
    accept = 0
    
    # 2. Propose parameters
    tmp_parmas_star = np.matmul(np.linalg.inv(inv_logscale2_cluster_proposal[cluster_num][0].T) , current_params) + lambda_current_cluster*random_generator.standard_normal(n_current_cluster)
    params_star = np.matmul(inv_logscale2_cluster_proposal[cluster_num][0].T , tmp_parmas_star)
    
    
    # 3. Calculate likelihoods
    beta_logscale2_star = np.empty(beta_logscale2.shape[0]); beta_logscale2_star[:] = beta_logscale2
    beta_logscale2_star[which] = params_star
    log_num = beta_logscale2_vec_gev_update_mixture_me_likelihood_interp(Y, beta_logscale2_star, Design_mat,  sbeta_logscale2, X_s, cen, cen_above, prob_below, prob_above,
                         delta, tau_sqd, logscale0, logscale1, Loc, Shape, WMGHGs, ELI_summer_average, xp, den_p, thresh_X, thresh_X_above)
    log_denom = beta_logscale2_vec_gev_update_mixture_me_likelihood_interp(Y, beta_logscale2, Design_mat,
        sbeta_logscale2, X_s, cen, cen_above, prob_below, prob_above,
                         delta, tau_sqd, logscale0, logscale1, Loc, Shape, WMGHGs, ELI_summer_average, xp, den_p, thresh_X, thresh_X_above)
    
    # 4. Decide whether to update or not
    r = np.exp(log_num - log_denom)
    if ~np.isfinite(r):
        r = 0
    if random_generator.uniform(0,1,1)<r:
        beta_logscale2[which] = params_star  # changes argument 'logscale2' directly
        accept = 1
    
    #result = (X_s,accept)
    return accept
##
## -------------------------------------------------------------------------- ##




## -------------------------------------------------------------------------- ##
## -----------------------------  shape vector ------------------------------ ##
## -------------------------------------------------------------------------- ##
## For the full conditionals of shape vector
def shape_vec_gev_update_mixture_me_likelihood(data, params, X_s, cen, cen_above, prob_below, prob_above,
                     delta, tau_sqd, Loc, Scale, Time, thresh_X, thresh_X_above):
  
  Y = data
  
  if len(X_s.shape)==1:
      X_s = X_s.reshape((X_s.shape[0],1))
  n_t = X_s.shape[1]
  n_s = X_s.shape[0]
  Shape = np.tile(params, n_t)
  Shape = Shape.reshape((n_s,n_t),order='F')
  
  max_support = Loc - Scale/Shape
  max_support[Shape>0] = np.inf
  
  tmp=pgev(Y[~cen & ~cen_above], Loc[~cen & ~cen_above], Scale[~cen & ~cen_above], Shape[~cen & ~cen_above])
  
  # If the parameters imply support that is not consistent with the data,
  # then reject the parameters.
  if np.any(Y[~np.isnan(Y)] > max_support[~np.isnan(Y)]) or np.nanmin(tmp)<prob_below-0.001 or np.nanmax(tmp)>prob_above+0.001:
      return -np.inf
  
  X = X_update(Y, cen, cen_above, delta, tau_sqd, Loc, Scale, Shape)
  ll = marg_transform_data_mixture_me_likelihood(Y, X, X_s, cen, cen_above, prob_below, prob_above,
            Loc, Scale, Shape, delta, tau_sqd, thresh_X, thresh_X_above)
  return ll


def beta_shape_vec_gev_update_mixture_me_likelihood_interp(data, params, Design_mat, sbeta_shape, X_s, cen, cen_above, prob_below, prob_above,
                     delta, tau_sqd, Loc, Scale, xp, den_p, thresh_X, thresh_X_above):
  
  Y = data
  
  if len(X_s.shape)==1:
      X_s = X_s.reshape((X_s.shape[0],1))
  n_t = X_s.shape[1]
  n_s = X_s.shape[0]
  shape = Design_mat @params
  Shape = np.tile(shape, n_t)
  Shape = Shape.reshape((n_s,n_t),order='F')
  
  max_support = Loc - Scale/Shape
  max_support[Shape>0] = np.inf
  
  tmp=pgev(Y[~cen & ~cen_above], Loc[~cen & ~cen_above], Scale[~cen & ~cen_above], Shape[~cen & ~cen_above])
  
  # If the parameters imply support that is not consistent with the data,
  # then reject the parameters.
  if np.any(Y[~np.isnan(Y)] > max_support[~np.isnan(Y)]) or np.nanmin(tmp)<prob_below-0.001 or np.nanmax(tmp)>prob_above+0.001:
      return -np.inf
  
  X = X_update(Y, cen, cen_above, delta, tau_sqd, Loc, Scale, Shape)
  ll = marg_transform_data_mixture_me_likelihood_interp(Y, X, X_s, cen, cen_above, prob_below, prob_above,
            Loc, Scale, Shape, delta, tau_sqd, xp, den_p, thresh_X, thresh_X_above) + dmvn_diag(
                         params, sbeta_shape)
  return ll

def mu_shape_vec_gev_update_mixture_me_likelihood_interp(data, params, Design_mat, beta_shape, sbeta_shape, X_s, cen, cen_above, prob_below, prob_above,
                     delta, tau_sqd, Loc, Scale, Time, xp, den_p, thresh_X, thresh_X_above):
  
  Y = data
  
  if len(X_s.shape)==1:
      X_s = X_s.reshape((X_s.shape[0],1))
  n_t = X_s.shape[1]
  n_s = X_s.shape[0]
  shape = params +Design_mat @beta_shape
  Shape = np.tile(shape, n_t)
  Shape = Shape.reshape((n_s,n_t),order='F')
  
  max_support = Loc - Scale/Shape
  max_support[Shape>0] = np.inf
  
  tmp=pgev(Y[~cen & ~cen_above], Loc[~cen & ~cen_above], Scale[~cen & ~cen_above], Shape[~cen & ~cen_above])
  
  # If the parameters imply support that is not consistent with the data,
  # then reject the parameters.
  if np.any(Y[~np.isnan(Y)] > max_support[~np.isnan(Y)]) or np.nanmin(tmp)<prob_below-0.001 or np.nanmax(tmp)>prob_above+0.001:
      return -np.inf
  
  X = X_update(Y, cen, cen_above, delta, tau_sqd, Loc, Scale, Shape)
  ll = marg_transform_data_mixture_me_likelihood_interp(Y, X, X_s, cen, cen_above, prob_below, prob_above,
            Loc, Scale, Shape, delta, tau_sqd, xp, den_p, thresh_X, thresh_X_above)
  return ll


## Update shape vector by cluster
## --- shape: vector with length n_s
## --- Cluster_which: bool list for identifying cluster labels
## --- cluster_num: which cluster to update?
## --- inv_shape_cluster: Cov cholesky matrix list for all clusters
## --- lambda_current_cluster: the random walk variance
def update_beta_shape_GEV_one_cluster_interp(beta_shape, Cluster_which, cluster_num, inv_shape_cluster_proposal,
                                 Design_mat, sbeta_shape, Y, X_s, cen, cen_above, prob_below, prob_above, delta, tau_sqd,
                                 Loc, Scale, xp, den_p, thresh_X, thresh_X_above,
                                 lambda_current_cluster, random_generator):
    # 1. Obtain the current parameters in the chosen cluster
    which = Cluster_which[cluster_num]
    current_params = beta_shape[which]
    n_current_cluster = len(current_params)
    accept = 0
    
    # 2. Propose parameters
    tmp_parmas_star = np.matmul(np.linalg.inv(inv_shape_cluster_proposal[cluster_num][0].T) , current_params) + lambda_current_cluster*random_generator.standard_normal(n_current_cluster)
    params_star = np.matmul(inv_shape_cluster_proposal[cluster_num][0].T , tmp_parmas_star)
    
    # plt.plot(np.arange(n_current_cluster), current_params, np.arange(n_current_cluster),params_star)
    # plt.plot(np.arange(n_current_cluster), np.matmul(np.linalg.inv(inv_shape_cluster[cluster_num][0].T) , current_params) , np.arange(n_current_cluster),tmp_parmas_star)
    
    # 3. Calculate likelihoods
    beta_shape_star = np.empty(beta_shape.shape[0]); beta_shape_star[:] = beta_shape
    beta_shape_star[which] = params_star
    log_num = beta_shape_vec_gev_update_mixture_me_likelihood_interp(Y, beta_shape_star, Design_mat, sbeta_shape, X_s, cen, cen_above, prob_below, prob_above,
                     delta, tau_sqd, Loc, Scale, xp, den_p, thresh_X, thresh_X_above)
    log_denom = beta_shape_vec_gev_update_mixture_me_likelihood_interp(Y, beta_shape, Design_mat, sbeta_shape, X_s, cen, cen_above, prob_below, prob_above,
                     delta, tau_sqd, Loc, Scale, xp, den_p, thresh_X, thresh_X_above)
    
    # 4. Decide whether to update or not
    r = np.exp(log_num - log_denom)
    if ~np.isfinite(r):
        r = 0
    if random_generator.uniform(0,1,1)<r:
        beta_shape[which] = params_star  # changes argument 'shape' directly
        accept = 1
    
    #result = (X_s,accept)
    return accept

##
## -------------------------------------------------------------------------- ##
