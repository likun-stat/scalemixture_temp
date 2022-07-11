import os
os.chdir("/Users/LikunZhang/Desktop/PyCode/")


import scalemixture_py.integrate as utils
import scalemixture_py.priors as priors
import scalemixture_py.generic_samplers as sampler
import numpy as np
import cProfile
import matplotlib.pyplot as plt
from scipy.stats import uniform
from scipy.stats import norm 



# ------------ 1. Simulation settings -------------
range = 1        # Matern range
nu =  3/2        # Matern smoothness

n_s = 100         # Number of sites
n_t = 64         # Number of time points
tau_sqd = 10     # Nugget SD

delta = 0.55       # For R
prob_below=0.9
prob_above=0.999


# -------------- 2. Generate covariance matrix -----------------
# Calculate distance between rows of 'Y', and return as distance matrix
np.random.seed(seed=1234)
from scipy.spatial import distance
Stations = np.c_[uniform.rvs(0,5,n_s),uniform.rvs(0,5,n_s)]
# plt.scatter(Stations[:,0],Stations[:,1])
S = distance.squareform(distance.pdist(Stations))
Cor = utils.corr_fn(S, np.array([range,nu]))
eig_Cor = np.linalg.eigh(Cor) #For symmetric matrices
V = eig_Cor[1]
d = eig_Cor[0]

R = utils.rPareto(n_t,1,1)
X = np.empty((n_s,n_t))
X[:] = np.nan
X_s = np.empty((n_s,n_t))
X_s[:] = np.nan
Z = np.empty((n_s,n_t))
Z[:] = np.nan
for idx, r in enumerate(R):
  Z_t = utils.eig2inv_times_vector(V, np.sqrt(d), norm.rvs(size=n_s))
  Z_to_W_s = 1/(1-norm.cdf(Z_t))
  tmp = (r**(delta/(1-delta)))*Z_to_W_s
  X_s[: ,idx] = tmp
  X[:,idx] = tmp + np.sqrt(tau_sqd)*norm.rvs(size=n_s)
  Z[:,idx] = Z_t


# ------------ 3. Marginal transformation -----------------
Lon_lat = Stations
Design_mat = np.c_[np.repeat(1,n_s), Lon_lat[:,1]]
n_covariates = Design_mat.shape[1]

beta_loc0 = np.array([0.2,-1])
loc0 = Design_mat @beta_loc0

beta_loc1 = np.array([0.1, -0.1])
loc1 = Design_mat @beta_loc1

Time = np.arange(n_t)
Loc = np.tile(loc0, n_t) + np.tile(loc1, n_t)*np.repeat(Time,n_s)
Loc = Loc.reshape((n_s,n_t),order='F')


beta_scale = np.array([0.1,1])
scale = Design_mat @beta_scale
Scale = np.tile(scale, n_t)
Scale = Scale.reshape((n_s,n_t),order='F')

beta_shape = np.array([-0.02,0.2])
shape = Design_mat @beta_shape
Shape = np.tile(shape, n_t)
Shape = Shape.reshape((n_s,n_t),order='F')

Y = utils.scalemix_me_2_gev(X, delta, tau_sqd, Loc, Scale, Shape)
unifs = utils.pgev(Y, Loc, Scale, Shape)

cen = unifs < prob_below
cen_above = unifs > prob_above
thresh_X =  utils.qmixture_me_interp(prob_below, delta = delta, tau_sqd = tau_sqd)
thresh_X_above =  utils.qmixture_me_interp(prob_above, delta = delta, tau_sqd = tau_sqd)





# ------------ 4. Save initial values -----------------
initial_values = {'delta':delta,
                    'tau_sqd':tau_sqd,
                    'prob_below':prob_below,
                    'prob_above':prob_above,
                    'Dist':S,
                    'theta_c':np.array([range,nu]),
                    'X':X,
                    'Z':Z,
                    'R':R,
                    'Design_mat':Design_mat,
                    'beta_loc0':beta_loc0,
                    'beta_loc1':beta_loc1,
                    'Time':Time,
                    'beta_scale':beta_scale,
                    'beta_shape':beta_shape,
                    }
n_updates = 1001    
sigma_m = {'delta':2.4**2,
             'tau_sqd':2.4**2,
             'theta_c':2.4**2/2,
             'Z_onetime':np.repeat(np.sqrt(tau_sqd),n_s),
             'R_1t':2.4**2,
             'beta_loc0':2.4**2/n_covariates,
             'beta_loc1':2.4**2/n_covariates,
             'beta_scale':2.4**2/n_covariates,
             'beta_shape':2.4**2/n_covariates,
             }
prop_sigma = {'theta_c':np.eye(2),
                'beta_loc0':np.eye(n_covariates),
                'beta_loc1':np.eye(n_covariates),
                'beta_scale':np.eye(n_covariates),
                'beta_shape':np.eye(n_covariates)
                }

from pickle import dump
with open('./test_scalemix.pkl', 'wb') as f:
     dump(Y, f)
     dump(cen, f)
     dump(cen_above,f)
     dump(initial_values, f)
     dump(sigma_m, f)
     dump(prop_sigma, f)







## ---------------------------------------------------------
## ----------------------- For delta -----------------------
## ---------------------------------------------------------
def test(delta):
    return utils.delta_update_mixture_me_likelihood(Y, delta, R, Z, cen, cen_above, prob_below, prob_above,
                                       Loc, Scale, Shape, tau_sqd)
Delta = np.arange(0.54,0.56,step=0.001)
Lik = np.zeros(len(Delta))
for idx, delt in enumerate(Delta):
    Lik[idx] = test(delt)
plt.plot(Delta, Lik, color='gray', linestyle='solid')
plt.axvline(0.55, color='r', linestyle='--');


# cProfile.run('Res = sampler.static_metr(R, 0.55, utils.delta_update_mixture_me_likelihood, priors.interval_unif, np.array([0.1,0.7]),1000, np.nan, 0.005, True, Y, X_s, cen, prob_below, Loc, Scale, Shape, tau_sqd)')

random_generator = np.random.RandomState()
Res = sampler.static_metr(Y, 0.55, utils.delta_update_mixture_me_likelihood, priors.interval_unif, 
                  np.array([0.1,0.7]),1000, 
                  random_generator,
                  np.nan, 5.3690987e-03, True, 
                  R, Z, cen, cen_above, prob_below, prob_above, Loc, Scale, Shape, tau_sqd)
plt.plot(np.arange(1000),Res['trace'][0,:],linestyle='solid')


Res = sampler.adaptive_metr(Y, 0.55, utils.delta_update_mixture_me_likelihood, priors.interval_unif, 
                          np.array([0.1,0.7]),5000,
                          random_generator,
                          np.nan, False, False,
                          .234, 10, .8, 10, 
                          R, Z, cen, cen_above, prob_below, prob_above, Loc, Scale, Shape, tau_sqd)
plt.plot(np.arange(5000),Res['trace'][0,:],linestyle='solid')
plt.hlines(0.55, 0, 5000, colors='r', linestyles='--');




## -------------------------------------------------------
## ----------------------- For tau -----------------------
## -------------------------------------------------------
def test(tau_sqd):
    return utils.tau_update_mixture_me_likelihood(Y, tau_sqd, X_s, cen, cen_above, prob_below, prob_above, Loc, Scale, Shape, delta)

Tau = np.arange(8.5,11.5,step=0.1)
Lik = np.zeros(len(Tau))
for idx, t in enumerate(Tau):
    Lik[idx] = test(t) 
plt.plot(Tau, Lik, linestyle='solid')
plt.axvline(tau_sqd, color='r', linestyle='--');

cProfile.run('Res = sampler.static_metr(R, 4, utils.tau_update_mixture_me_likelihood, priors.invGamma_prior, np.array([0.1,0.1]),1000, np.nan, 1, True, Y, X_s, cen, prob_below, Loc, Scale, Shape, delta)')

Res = sampler.static_metr(Y, 4, utils.tau_update_mixture_me_likelihood, priors.invGamma_prior, 
                          np.array([0.1,0.1]),1000, 
                          random_generator,
                          np.nan, 2.03324631, True, 
                          X_s, cen, cen_above, prob_below, prob_above, Loc, Scale, Shape, delta)
plt.plot(np.arange(1000),Res['trace'][0,:],linestyle='solid')


Res = sampler.adaptive_metr(Y, 4, utils.tau_update_mixture_me_likelihood, priors.invGamma_prior, 
                          np.array([0.1,0.1]),5000, 
                          random_generator,
                          np.nan, False, False,
                          .234, 10, .8, 10, 
                          X_s, cen, cen_above, prob_below, prob_above, Loc, Scale, Shape, delta)
plt.plot(np.arange(5000),Res['trace'][0,:],linestyle='solid')
plt.hlines(tau_sqd, 0, 5000, colors='r', linestyles='--');




## --------------------------------------------------------------
## ----------------------- For GEV params -----------------------
## --------------------------------------------------------------
# (1) loc0: 0.2,-1
def test(x):
    return utils.loc0_gev_update_mixture_me_likelihood(Design_mat, np.array([x,-1]), Y, X_s, cen, cen_above, prob_below, prob_above, 
                     delta, tau_sqd, loc1, Scale, Shape, Time, thresh_X, thresh_X_above)

Coef = np.arange(0.18,0.3,step=0.003)
Lik = np.zeros(len(Coef))
for idx, coef in enumerate(Coef):
    Lik[idx] = test(coef)
plt.plot(Coef, Lik, linestyle='solid')
plt.axvline(0.2, color='r', linestyle='--');

def test(x):
    return utils.loc0_gev_update_mixture_me_likelihood(Design_mat, np.array([0.2,x]), Y, X_s, cen, cen_above, prob_below, prob_above, 
                     delta, tau_sqd, loc1, Scale, Shape, Time, thresh_X, thresh_X_above)

Coef = np.arange(-1.2,-0.8,step=0.03)
Lik = np.zeros(len(Coef))
for idx, coef in enumerate(Coef):
    Lik[idx] = test(coef)
plt.plot(Coef, Lik, linestyle='solid')
plt.axvline(-1, color='r', linestyle='--');


Res = sampler.adaptive_metr(Design_mat, np.array([0.2,-1]), utils.loc0_gev_update_mixture_me_likelihood, 
                            priors.unif_prior, 20, 5000, 
                            random_generator,
                            np.nan, True,
                            False, .234, 10, .8,  10,
                            Y, X_s, cen, cen_above, prob_below, prob_above,
                            delta, tau_sqd, loc1, Scale, Shape, Time, thresh_X, thresh_X_above)

prop_Sigma=np.cov(Res['trace'])
Res = sampler.static_metr(Design_mat, np.array([0.2,-1]), utils.loc0_gev_update_mixture_me_likelihood, 
                            priors.unif_prior, 20, 5000,
                            random_generator,
                            prop_Sigma, 1, True, 
                            Y, X_s, cen, cen_above, prob_below, prob_above,
                            delta, tau_sqd, loc1, Scale, Shape, Time, thresh_X, thresh_X_above)



plt.plot(np.arange(5000),Res['trace'][0,:], color='gray',linestyle='solid')
plt.hlines(0.2, 0, 5000, colors='r', linestyles='--');

plt.plot(np.arange(5000),Res['trace'][1,:], color='gray',linestyle='solid')
plt.hlines(-1, 0, 5000, colors='r', linestyles='--');
plt.plot(*Res['trace'])
plt.show()


def tmpf(x,y):
    return utils.loc0_gev_update_mixture_me_likelihood(Design_mat, np.array([x,y]), Y, X_s, cen, cen_above, prob_below, prob_above,
                     delta, tau_sqd, loc1, Scale, Shape, Time, thresh_X, thresh_X_above)
try_size = 50
x = np.linspace(0.18, 0.22, try_size)
y = np.linspace(-1.1, -0.92, try_size)

Z = np.empty((try_size,try_size))
for idy,yi in enumerate(y):
    for idx,xi in enumerate(x):
         Z[idy,idx] = tmpf(xi,yi)

plt.contourf(x, y, Z, 20, cmap='RdGy')
plt.colorbar();

## Not seem to be better
Res = sampler.adaptive_metr_ratio(Design_mat, np.array([0.2,-1]), utils.loc0_gev_update_mixture_me_likelihood, 
                            priors.unif_prior, 20, 5000, 
                            random_generator,
                            prop_Sigma, -0.2262189, 0.2827393557113686, True,
                            False, .234, 10, .8,  10,
                            Y, X_s, cen, cen_above, prob_below, prob_above, 
                            delta, tau_sqd, loc1, Scale, Shape, Time, thresh_X, thresh_X_above)


# (2) loc1: 0.1, -0.1
def test(x):
    return utils.loc1_gev_update_mixture_me_likelihood(Design_mat, np.array([x,-0.1]), Y, X_s, cen, cen_above, prob_below, prob_above,
                     delta, tau_sqd, loc0, Scale, Shape, Time, thresh_X, thresh_X_above)
Coef = np.arange(0.095,0.12,step=0.0005)
Lik = np.zeros(len(Coef))
for idx, coef in enumerate(Coef):
    Lik[idx] = test(coef)
plt.plot(Coef, Lik, linestyle='solid')


def test(x):
    return utils.loc1_gev_update_mixture_me_likelihood(Design_mat, np.array([0.1,x]), Y, X_s, cen, cen_above, prob_below, prob_above,
                     delta, tau_sqd, loc0, Scale, Shape, Time, thresh_X, thresh_X_above)

Coef = np.arange(-0.11,-0.08,step=0.001)
Lik = np.zeros(len(Coef))
for idx, coef in enumerate(Coef):
    Lik[idx] = test(coef)
plt.plot(Coef, Lik, linestyle='solid')



Res = sampler.adaptive_metr(Design_mat, np.array([0.1,-0.1]), utils.loc1_gev_update_mixture_me_likelihood, 
                            priors.unif_prior, 20, 5000, 
                            random_generator,
                            np.nan, True,
                            False, .234, 10, .8,  10,
                            Y, X_s, cen, cen_above, prob_below, prob_above,
                            delta, tau_sqd, loc0, Scale, Shape, Time, thresh_X, thresh_X_above)



plt.plot(np.arange(5000),Res['trace'][0,:], color='gray',linestyle='solid')
plt.hlines(0.1, 0, 5000, colors='r', linestyles='--');

plt.plot(np.arange(5000),Res['trace'][1,:], color='gray',linestyle='solid')
plt.hlines(-0.1, 0, 5000, colors='r', linestyles='--');
plt.plot(*Res['trace'])
plt.show()


def tmpf(x,y):
    return utils.loc1_gev_update_mixture_me_likelihood(Design_mat, np.array([x,y]), Y, X_s, cen, cen_above, prob_below, prob_above, 
                     delta, tau_sqd, loc0, Scale, Shape, Time, thresh_X, thresh_X_above)
try_size = 50
x = np.linspace(0.098, 0.103, try_size)
y = np.linspace(-0.103, -0.096, try_size)

Z = np.empty((try_size,try_size))
for idy,yi in enumerate(y):
    for idx,xi in enumerate(x):
         Z[idy,idx] = tmpf(xi,yi)

plt.contourf(x, y, Z, 20, cmap='RdGy')
plt.colorbar();



# (3) scale: 0.1,1
def test(x):
    return utils.scale_gev_update_mixture_me_likelihood(Design_mat, np.array([x,1]), Y, X_s, cen, cen_above, prob_below, prob_above,
                     delta, tau_sqd, Loc, Shape, Time, thresh_X, thresh_X_above)
Coef = np.arange(0.095,0.12,step=0.0005)
Lik = np.zeros(len(Coef))
for idx, coef in enumerate(Coef):
    Lik[idx] = test(coef)
plt.plot(Coef, Lik, linestyle='solid')


def test(x):
    return utils.scale_gev_update_mixture_me_likelihood(Design_mat, np.array([0.1,x]), Y, X_s, cen, cen_above, prob_below, prob_above,
                     delta, tau_sqd, Loc, Shape, Time, thresh_X, thresh_X_above)

Coef = np.arange(0.975,1.2,step=0.001)
Lik = np.zeros(len(Coef))
for idx, coef in enumerate(Coef):
    Lik[idx] = test(coef)
plt.plot(Coef, Lik, linestyle='solid')



Res = sampler.adaptive_metr(Design_mat, np.array([0.1,1]), utils.scale_gev_update_mixture_me_likelihood, 
                            priors.unif_prior, 20, 5000, 
                            random_generator,
                            np.nan, True,
                            False, .234, 10, .8,  10,
                            Y, X_s, cen, cen_above, prob_below, prob_above,
                            delta, tau_sqd, Loc, Shape, Time, thresh_X, thresh_X_above)


plt.plot(np.arange(5000),Res['trace'][0,:], color='gray',linestyle='solid')
plt.hlines(0.1, 0, 5000, colors='r', linestyles='--');

plt.plot(np.arange(5000),Res['trace'][1,:], color='gray',linestyle='solid')
plt.hlines(1, 0, 5000, colors='r', linestyles='--');
plt.plot(*Res['trace'])
plt.show()


def tmpf(x,y):
    return utils.scale_gev_update_mixture_me_likelihood(Design_mat, np.array([x,y]), Y, X_s, cen, cen_above, prob_below, prob_above, 
                     delta, tau_sqd, Loc, Shape, Time, thresh_X, thresh_X_above)
try_size = 50
x = np.linspace(0.097, 0.104, try_size)
y = np.linspace(0.996, 1.005, try_size)

Z = np.empty((try_size,try_size))
for idy,yi in enumerate(y):
    for idx,xi in enumerate(x):
         Z[idy,idx] = tmpf(xi,yi)

plt.contourf(x, y, Z, 20, cmap='RdGy')
plt.colorbar();



# (4) shape: -0.02,0.2
def test(x):
    return utils.shape_gev_update_mixture_me_likelihood(Design_mat, np.array([x,0.2]), Y, X_s, cen, cen_above, prob_below, prob_above, 
                     delta, tau_sqd, Loc, Scale, Time, thresh_X, thresh_X_above)
Coef = np.arange(-0.03,0.,step=0.0005)
Lik = np.zeros(len(Coef))
for idx, coef in enumerate(Coef):
    Lik[idx] = test(coef)
plt.plot(Coef, Lik, linestyle='solid')


def test(x):
    return utils.shape_gev_update_mixture_me_likelihood(Design_mat, np.array([-0.02,x]), Y, X_s, cen, cen_above, prob_below, prob_above, 
                     delta, tau_sqd, Loc, Scale, Time, thresh_X, thresh_X_above)

Coef = np.arange(0.18,0.3,step=0.001)
Lik = np.zeros(len(Coef))
for idx, coef in enumerate(Coef):
    Lik[idx] = test(coef)
plt.plot(Coef, Lik, linestyle='solid')



Res = sampler.adaptive_metr(Design_mat, np.array([-0.02,0.2]), utils.shape_gev_update_mixture_me_likelihood, 
                            priors.unif_prior, 20, 5000, 
                            random_generator,
                            np.nan, True,
                            False, .234, 10, .8,  10,
                            Y, X_s, cen, cen_above, prob_below, prob_above,
                            delta, tau_sqd, Loc, Scale, Time, thresh_X, thresh_X_above)


plt.plot(np.arange(5000),Res['trace'][0,:], color='gray',linestyle='solid')
plt.hlines(-0.02, 0, 5000, colors='r', linestyles='--');

plt.plot(np.arange(5000),Res['trace'][1,:], color='gray',linestyle='solid')
plt.hlines(0.2, 0, 5000, colors='r', linestyles='--');
plt.plot(*Res['trace'])
plt.show()


def tmpf(x,y):
    return utils.shape_gev_update_mixture_me_likelihood(Design_mat, np.array([x,y]), Y, X_s, cen, cen_above, prob_below, prob_above,
                     delta, tau_sqd, Loc, Scale, Time, thresh_X, thresh_X_above)
try_size = 50
x = np.linspace(-0.0215, -0.0185, try_size)
y = np.linspace(0.1985, 0.2020, try_size)

Z = np.empty((try_size,try_size))
for idy,yi in enumerate(y):
    for idx,xi in enumerate(x):
         Z[idy,idx] = tmpf(xi,yi)

plt.contourf(x, y, Z, 20, cmap='RdGy')
plt.colorbar();





# (5) GEV altogether: 0.2,-1, 0.1,-0.1, 0.1,1, -0.02,0.2
def tmpf(x,y):
    return utils.gev_update_mixture_me_likelihood(Design_mat, np.array([0.2,x, 0.1,-0.1, 0.1,y, -0.02,0.2]), 
                                                        Y, X_s, cen, cen_above, prob_below, prob_above,
                                                        delta, tau_sqd, Time, thresh_X, thresh_X_above)
try_size = 50
x = np.linspace(-1.6, -0.4, try_size)
y = np.linspace(0.4, 1.6, try_size)

Z = np.empty((try_size,try_size))
for idy,yi in enumerate(y):
    for idx,xi in enumerate(x):
         Z[idy,idx] = tmpf(xi,yi)

plt.contourf(x, y, Z, 20, cmap='RdGy')
plt.colorbar();


Res = sampler.adaptive_metr(Design_mat, np.array([-0.02,0.2]), utils.shape_gev_update_mixture_me_likelihood, 
                            priors.unif_prior, 20, 5000, 
                            random_generator,
                            np.nan, True,
                            False, .234, 10, .8,  10,
                            Y, X_s, cen, cen_above, prob_below, prob_above,
                            delta, tau_sqd, Loc, Scale, Time, thresh_X, thresh_X_above)


plt.plot(np.arange(5000),Res['trace'][0,:], color='gray',linestyle='solid')
plt.hlines(-0.02, 0, 5000, colors='r', linestyles='--');

plt.plot(np.arange(5000),Res['trace'][1,:], color='gray',linestyle='solid')
plt.hlines(0.2, 0, 5000, colors='r', linestyles='--');
plt.plot(*Res['trace'])
plt.show()


def tmpf(x,y):
    return utils.shape_gev_update_mixture_me_likelihood(Design_mat, np.array([x,y]), Y, X_s, cen, cen_above, prob_below, prob_above,
                     delta, tau_sqd, Loc, Scale, Time, thresh_X, thresh_X_above)
try_size = 50
x = np.linspace(-0.0215, -0.0185, try_size)
y = np.linspace(0.1985, 0.2020, try_size)

Z = np.empty((try_size,try_size))
for idy,yi in enumerate(y):
    for idx,xi in enumerate(x):
         Z[idy,idx] = tmpf(xi,yi)

plt.contourf(x, y, Z, 20, cmap='RdGy')
plt.colorbar();




## -------------------------------------------------------
## --------------------- For theta_c ---------------------
## -------------------------------------------------------

# 1, 1.5
def test(x):
    return utils.theta_c_update_mixture_me_likelihood(Z, np.array([x,1.5]), S)

Range = np.arange(0.5,1.3,step=0.01)
Lik = np.zeros(len(Range))
for idx, r in enumerate(Range):
    Lik[idx] = test(r) 
plt.plot(Range, Lik, linestyle='solid')


def test(x):
    return utils.theta_c_update_mixture_me_likelihood(Z, np.array([1,x]), S)

Nu = np.arange(0.9,1.8,step=0.01)
Lik = np.zeros(len(Nu))
for idx, r in enumerate(Nu):
    Lik[idx] = test(r) 
plt.plot(Nu, Lik, linestyle='solid')


Res = sampler.adaptive_metr(Z, np.array([1,1.5]), utils.theta_c_update_mixture_me_likelihood, 
                            priors.unif_prior, 20, 5000, 
                            random_generator,
                            np.nan, True,
                            False, .234, 10, .8,  10,
                            S)

plt.plot(np.arange(5000),Res['trace'][0,:], color='gray',linestyle='solid')
plt.hlines(1, 0, 5000, colors='r', linestyles='--');

plt.plot(np.arange(5000),Res['trace'][1,:], color='gray',linestyle='solid')
plt.hlines(1.5, 0, 5000, colors='r', linestyles='--');
plt.plot(*Res['trace'])
plt.show()


def tmpf(x,y):
    return utils.theta_c_update_mixture_me_likelihood(R, np.array([x,y]), X_s, S)

try_size = 50
x = np.linspace(0.85, 1.15, try_size)
y = np.linspace(1.37, 1.7, try_size)

Z = np.empty((try_size,try_size))
for idy,yi in enumerate(y):
    for idx,xi in enumerate(x):
         Z[idy,idx] = tmpf(xi,yi)

plt.contourf(x, y, Z, 20, cmap='RdGy')
plt.colorbar();




## -------------------------------------------------------
## ------------------------ For Rt -----------------------
## -------------------------------------------------------

t_chosen = 6
def test(x):
    return utils.Rt_update_mixture_me_likelihood(Y[:,t_chosen], x, X[:,t_chosen], Z[:,t_chosen], 
                cen[:,t_chosen], cen_above[:,t_chosen], prob_below, prob_above, 
                Loc[:,t_chosen], Scale[:,t_chosen], Shape[:,t_chosen], delta, tau_sqd,
                thresh_X, thresh_X_above)

Rt = np.arange(R[t_chosen]-1,R[t_chosen]+1,step=0.01)
Lik = np.zeros(len(Rt))
for idx, r in enumerate(Rt):
    Lik[idx] = test(r) 
plt.plot(Rt, Lik, linestyle='solid')


Res = sampler.adaptive_metr(Y[:,t_chosen], R[t_chosen]-.1, utils.Rt_update_mixture_me_likelihood, 
                          priors.R_prior, 1, 5000, 
                          random_generator,
                          np.nan, True,
                          False, .234, 10, .8,  10, 
                          X[:,t_chosen], Z[:,t_chosen], 
                          cen[:,t_chosen], cen_above[:,t_chosen], prob_below, prob_above, 
                          Loc[:,t_chosen], Scale[:,t_chosen], Shape[:,t_chosen], delta, tau_sqd,
                          thresh_X, thresh_X_above)
plt.plot(np.arange(5000),Res['trace'][0,:],linestyle='solid')
plt.hlines(R[t_chosen], 0, 5000, colors='r', linestyles='--');




## -------------------------------------------------------
## ------------------------ For Xs -----------------------
## -------------------------------------------------------

t_chosen = 3; idx = 47
prop_Z= np.empty(n_s)
prop_Z[:] = Z[:,t_chosen]
def test(x):
    prop_Z[idx] =x
    prop_X_s_idx = (R[t_chosen]**(delta/(1-delta)))*utils.norm_to_Pareto(x)
    return utils.marg_transform_data_mixture_me_likelihood_uni(Y[idx, t_chosen], X[idx, t_chosen], prop_X_s_idx, 
                       cen[idx, t_chosen], cen_above[idx, t_chosen], prob_below, prob_above, Loc[idx, t_chosen], Scale[idx, t_chosen], Shape[idx, t_chosen], delta, tau_sqd, 
                       thresh_X, thresh_X_above) + utils.Z_likelihood_conditional(prop_Z, V, d)
Zt = np.arange(Z[idx,t_chosen]-2,Z[idx,t_chosen]+1,step=0.01)
Lik = np.zeros(len(Zt))
for idy, x in enumerate(Zt):
    Lik[idy] = test(x) 
plt.plot(Zt, Lik, linestyle='solid')
plt.axvline(Z[idx, t_chosen], color='r', linestyle='--');


n_updates = 5000
Z_trace = np.empty((3,n_updates))

Z_new = np.empty(n_s)
Z_new[:] = Z[:, t_chosen]
K=10; k=3
r_opt = .234; c_0 = 10; c_1 = .8
accept = np.zeros(n_s)
Sigma_m = np.repeat(np.sqrt(tau_sqd),n_s)

random_generator = np.random.RandomState()
for idx in np.arange(n_updates):
    tmp = utils.Z_update_onetime(Y[:,t_chosen], X[:,t_chosen], R[t_chosen], Z_new, cen[:,t_chosen], cen_above[:, t_chosen], prob_below, prob_above,
                                   delta, tau_sqd, Loc[:,t_chosen], Scale[:,t_chosen], Shape[:,t_chosen], thresh_X, thresh_X_above,
                                   V, d, Sigma_m, random_generator)
    Z_trace[:,idx] = np.array([Z_new[32],Z_new[58],Z_new[47]])
    accept = accept + tmp
    
    if (idx % K) == 0:
        print('Finished ' + str(idx) + ' out of ' + str(n_updates) + ' iterations ')
        gamma2 = 1 / ((idx/K) + k)**(c_1)
        gamma1 = c_0*gamma2
        R_hat = accept/K
        Sigma_m = np.exp(np.log(Sigma_m) + gamma1*(R_hat - r_opt))
        accept[:] = 0


plt.plot(np.arange(n_updates),Z_trace[0,:],linestyle='solid')
plt.hlines(Z[32, t_chosen], 0, n_updates, colors='r', linestyles='--');

plt.plot(np.arange(n_updates),Z_trace[1,:],linestyle='solid')
plt.hlines(Z[58, t_chosen], 0, n_updates, colors='r', linestyles='--');

plt.plot(np.arange(n_updates),Z_trace[2,:],linestyle='solid')
plt.hlines(Z[47, t_chosen], 0, n_updates, colors='r', linestyles='--');



## -------------------------------------------------------
## --------------------- For Sampler ---------------------
## -------------------------------------------------------



