#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 09:34:03 2021

@author: LikunZhang
"""

###################################################################################
## Main sampler

## Must provide data input 'data_input.pkl' to initiate the sampler.
## In 'data_input.pkl', one must include
##      Y ........................................... censored observations on GEV scale
##      cen ........................................................... indicator matrix
##      initial.values .................. a dictionary: delta, tau_sqd, prob_below, Dist,
##                                             theta_c, X, X_s, R, Design_mat, beta_loc0,
##                                             beta_loc1, Time, beta_scale, beta_shape
##      n_updates .................................................... number of updates
##      thinning ......................................... number of runs in each update
##      experiment_name
##      echo_interval ......................... echo process every echo_interval updates
##      sigma_m
##      prop_Sigma
##      true_params ....................... a dictionary: delta, rho, tau_sqd, theta_gpd,
##                                              prob_below, X_s, R
##

 
      

if __name__ == "__main__":
   import scalemixture_spline.integrate as utils
   import scalemixture_spline.priors as priors
   import scalemixture_spline.generic_samplers as sampler
   import os
   import numpy as np
   import matplotlib.pyplot as plt
   from matplotlib.backends.backend_pdf import PdfPages
   from pickle import load
   from pickle import dump
   # from scipy.stats import norm
   # from scipy.stats import invgamma  
   from scipy.linalg import cholesky
   
   # Check whether the 'mpi4py' is installed
   test_mpi = os.system("python -c 'from mpi4py import *' &> /dev/null")
   if test_mpi != 0:
      import sys
      sys.exit("mpi4py import is failing, aborting...")
   
   # get rank and size
   from mpi4py import MPI
  
   comm = MPI.COMM_WORLD
   rank = comm.Get_rank()
   size = comm.Get_size()
   thinning = 10; echo_interval = 50; n_updates = 40001
  
   # Filename for storing the intermediate results
   filename='./scalemix_progress_'+str(rank)+'.pkl'
   
   # Load data input
   if rank==0:
       with open(filename, 'rb') as f:
           Y = load(f)
           cen = load(f)
           cen_above = load(f)
           initial_values = load(f)
           sigma_m = load(f)
           prop_sigma = load(f)
           iter_current = load(f)
           delta_trace = load(f)
           tau_sqd_trace = load(f)
           theta_c_trace = load(f)
           beta_loc0_trace = load(f)
           beta_loc1_trace = load(f)
           beta_scale_trace = load(f)
           beta_shape_trace = load(f)
           sigma_sbeta_loc0_trace = load(f)
           sigma_sbeta_loc1_trace = load(f)
           sigma_sbeta_scale_trace = load(f)
           sigma_sbeta_shape_trace = load(f)
           
           mu_loc0_trace = load(f)
           mu_loc1_trace = load(f)
           mu_scale_trace = load(f)
           mu_shape_trace = load(f)
           
           loc0_trace = load(f)
           loc1_trace = load(f)
           scale_trace = load(f)
           shape_trace = load(f)
           
           Z_1t_trace = load(f)
           R_1t_trace = load(f)
           Y_onetime = load(f)
           X_onetime = load(f)
           X_s_onetime = load(f)
           R_onetime = load(f)
           
           sigma_m_Z_cluster = load(f)
           sigma_m_beta_loc0_cluster = load(f)
           sigma_m_beta_loc1_cluster = load(f)
           sigma_m_beta_scale_cluster = load(f)
           sigma_m_beta_shape_cluster = load(f)
           
           sigma_beta_loc0_cluster_proposal = load(f)
           sigma_beta_loc1_cluster_proposal = load(f)
           sigma_beta_scale_cluster_proposal = load(f)
           sigma_beta_shape_cluster_proposal = load(f)
           sigma_Z_cluster_proposal = load(f)
           f.close()
   else:
       with open(filename, 'rb') as f:
           Y = load(f)
           cen = load(f)
           cen_above = load(f)
           initial_values = load(f)
           sigma_m = load(f)
           sigma_m_Z_cluster = load(f)
           iter_current = load(f)
           Z_1t_trace = load(f)
           R_1t_trace = load(f)
           Y_onetime = load(f)
           X_onetime = load(f)
           X_s_onetime = load(f)
           R_onetime = load(f)
           sigma_Z_cluster_proposal = load(f)
           f.close()
     
   # Bookkeeping
   n_s = Y.shape[0]
   n_t = Y.shape[1]
   if n_t != size:
      import sys
      sys.exit("Make sure the number of cpus (N) = number of time replicates (n_t), i.e.\n     srun -N python scalemix_sampler.py")
   n_updates_thinned = np.int(np.ceil(n_updates/thinning))
   wh_to_plot_Xs = n_s*np.array([0.25,0.5,0.75])
   wh_to_plot_Xs = wh_to_plot_Xs.astype(int)
   
   # Generate multiple independent random streams
   random_generator = np.random.RandomState()
  
   # Constants to control adaptation of the Metropolis sampler
   c_0 = 10
   c_1 = 0.8
   offset = 3  # the iteration offset
   r_opt_1d = .41
   r_opt_2d = .35
   r_opt_md = .23
   eps = 1e-6 # a small number
  
   # Hyper parameters for the prior of the mixing distribution parameters and
   hyper_params_delta = np.array([0.1,0.7])
   hyper_params_tau_sqd = np.array([0.1,0.1])
   hyper_params_theta_c = np.array([0, 20])
   hyper_params_theta_c_loc0 = np.array([0, 20])
   hyper_params_theta_c_loc1 = np.array([0, 20])
   hyper_params_theta_c_scale = np.array([0, 20])
   hyper_params_theta_c_shape = np.array([0, 20])
   
   hyper_params_mu_loc0 = np.array([-100,100])
   hyper_params_mu_loc1 = np.array([-100,100])
   hyper_params_mu_scale = np.array([-100,100])
   hyper_params_mu_shape = np.array([-100,100]) 
   
   hyper_params_sbeta_loc0 = 1
   hyper_params_sbeta_loc1 = 1
   hyper_params_sbeta_scale = 1
   hyper_params_sbeta_shape = 1
   
   # Load latest values
   initial_values = comm.bcast(initial_values,root=0) # Latest values are mostly in initial_values
   delta = initial_values['delta']
   tau_sqd = initial_values['tau_sqd']
   grid = utils.density_interp_grid(delta, tau_sqd)
   xp = grid[0]; den_p = grid[1]
   prob_below = initial_values['prob_below']
   prob_above = initial_values['prob_above']
   
   X = initial_values['X']
   Z = initial_values['Z']
   R = initial_values['R']
   # X_s = (R**(delta/(1-delta)))*utils.norm_to_Pareto(Z)
   Y_onetime = Y[:,rank]
   X_onetime = X[:,rank]
   Z_onetime = Z[:,rank]
   R_onetime = R[rank]
   
   loc0 = initial_values['loc0']
   loc1 = initial_values['loc1']
   scale = initial_values['scale']
   shape = initial_values['shape']
   Design_mat = initial_values['Design_mat']
   beta_loc0 = initial_values['beta_loc0']
   beta_loc1 = initial_values['beta_loc1']
   Time = initial_values['Time']
   beta_scale = initial_values['beta_scale']
   beta_shape = initial_values['beta_shape']
   mu_loc0 = initial_values['mu_loc0']
   mu_loc1 = initial_values['mu_loc1']
   mu_scale = initial_values['mu_scale']
   mu_shape = initial_values['mu_shape']
   
   
   theta_c = initial_values['theta_c']
   sbeta_loc0 = initial_values['sbeta_loc0']
   sbeta_loc1 = initial_values['sbeta_loc1']
   sbeta_scale = initial_values['sbeta_scale']
   sbeta_shape = initial_values['sbeta_shape']
   
   Cluster_which = initial_values['Cluster_which']
   S_clusters = initial_values['S_clusters']
   betaCluster_which = initial_values['betaCluster_which']
   
   n_covariates = len(beta_loc0)
   Dist = initial_values['Dist']
   
   if prob_below==0:
        thresh_X = -np.inf
   else:
        thresh_X = utils.qmixture_me_interp(prob_below, delta = delta, tau_sqd = tau_sqd)
   if prob_above==1:
        thresh_X_above = np.inf
   else:
        thresh_X_above = utils.qmixture_me_interp(prob_above, delta = delta, tau_sqd = tau_sqd)
   
   # Cholesky decomposition of the correlation matrix
   # tmp_vec = np.ones(n_s)
   # Cor = utils.corr_fn(Dist, theta_c)
   # # eig_Cor = np.linalg.eigh(Cor) #For symmetric matrices
   # # V = eig_Cor[1]
   # # d = eig_Cor[0]
   # cholesky_inv_all = lapack.dposv(Cor,tmp_vec)
   n_clusters = len(S_clusters)
   n_beta_clusters = len(betaCluster_which)
   
   Cor_Z_clusters=list()
   inv_Z_cluster=list()
   for i in np.arange(n_clusters):
        Cor_tmp = utils.corr_fn(S_clusters[i], theta_c)
        cholesky_inv = (cholesky(Cor_tmp,lower=False),np.repeat(1,Cor_tmp.shape[0]))
        Cor_Z_clusters.append(Cor_tmp)
        inv_Z_cluster.append(cholesky_inv)

   inv_Z_cluster_proposal=list()
   for i in np.arange(n_clusters):
        which_tmp = Cluster_which[i]
        inv_Z_cluster_proposal.append((cholesky(sigma_Z_cluster_proposal[i],lower=False),np.repeat(1,np.sum(which_tmp))))

   Z_within_thinning = np.empty((n_s,thinning)); Z_within_thinning[:] = np.nan
   
   # Marginal GEV parameters: per location x time
   loc0 = mu_loc0+Design_mat @beta_loc0
   loc1 = mu_loc1+Design_mat @beta_loc1
   scale = np.exp(mu_scale+Design_mat @beta_scale)
   shape = mu_shape+Design_mat @beta_shape
   
   Loc = np.tile(loc0, n_t) + np.tile(loc1, n_t)*np.repeat(Time,n_s)
   Loc = Loc.reshape((n_s,n_t),order='F')

   Scale = np.tile(scale, n_t)
   Scale = Scale.reshape((n_s,n_t),order='F')

   Shape = np.tile(shape, n_t)
   Shape = Shape.reshape((n_s,n_t),order='F')
    
   # Initial trace objects
   Z_1t_accept = np.repeat(0,n_clusters)
   R_accept = 0
   if rank == 0:
     print("Number of time replicates = %d"%size)
     X_s = np.empty((n_s,n_t))
     theta_c_trace_within_thinning = np.empty((2,thinning)); theta_c_trace_within_thinning[:] = np.nan
         
     delta_accept = 0
     tau_sqd_accept = 0
     theta_c_accept = 0
          
     beta_loc0_within_thinning = np.empty((n_covariates,thinning)); beta_loc0_within_thinning[:] = np.nan
     beta_loc1_within_thinning = np.empty((n_covariates,thinning)); beta_loc1_within_thinning[:] = np.nan
     beta_scale_within_thinning = np.empty((n_covariates,thinning)); beta_scale_within_thinning[:] = np.nan
     beta_shape_within_thinning = np.empty((n_covariates,thinning)); beta_shape_within_thinning[:] = np.nan
     
     beta_loc0_accept = np.repeat(0,n_beta_clusters)
     beta_loc1_accept = np.repeat(0,n_beta_clusters)
     beta_scale_accept = np.repeat(0,n_beta_clusters)
     beta_shape_accept = np.repeat(0,n_beta_clusters)
     
     mu_loc0_accept = 0
     mu_loc1_accept = 0
     mu_scale_accept = 0
     mu_shape_accept = 0
     
     sbeta_loc0_accept = 0
     sbeta_loc1_accept = 0
     sbeta_scale_accept = 0
     sbeta_shape_accept = 0
     
     inv_beta_loc0_cluster_proposal=list()
     for i in np.arange(n_beta_clusters):
        which_tmp = betaCluster_which[i]
        inv_beta_loc0_cluster_proposal.append((cholesky(sigma_beta_loc0_cluster_proposal[i],lower=False),np.repeat(1,np.sum(which_tmp))))
        
    
     inv_beta_loc1_cluster_proposal=list()
     for i in np.arange(n_beta_clusters):
        which_tmp = betaCluster_which[i]
        inv_beta_loc1_cluster_proposal.append((cholesky(sigma_beta_loc1_cluster_proposal[i],lower=False),np.repeat(1,np.sum(which_tmp))))           
  
     inv_beta_scale_cluster_proposal=list()
     for i in np.arange(n_beta_clusters):
        which_tmp = betaCluster_which[i]
        inv_beta_scale_cluster_proposal.append((cholesky(sigma_beta_scale_cluster_proposal[i],lower=False),np.repeat(1,np.sum(which_tmp))))
 
    
     inv_beta_shape_cluster_proposal=list()
     for i in np.arange(n_beta_clusters):
        which_tmp = betaCluster_which[i]
        inv_beta_shape_cluster_proposal.append((cholesky(sigma_beta_shape_cluster_proposal[i],lower=False),np.repeat(1,np.sum(which_tmp))))
     
     # XtX = np.matmul(Design_mat.T, Design_mat)
     # D_sigma_loc0_inv = np.concatenate((np.repeat(1/sbeta_loc0,97), np.repeat(0.0025,2)))
     # D_sigma_loc1_inv = np.concatenate((np.repeat(1/sbeta_loc1,97), np.repeat(0.0025,2)))
     # D_sigma_scale_inv = np.concatenate((np.repeat(1/sbeta_scale,97), np.repeat(0.0025,2)))
     # D_sigma_shape_inv = np.concatenate((np.repeat(1/sbeta_shape,97), np.repeat(0.0025,2)))
  
    
    
    
   # -----------------------------------------------------------------------------------
   # -----------------------------------------------------------------------------------
   # --------------------------- Start Metropolis Updates ------------------------------
   # -----------------------------------------------------------------------------------
   # -----------------------------------------------------------------------------------
   for iter in np.arange(iter_current+1,n_updates):
       index_within = (iter-1)%thinning
       # Update X
       # print(str(rank)+" "+str(iter)+" Gathered? "+str(np.where(~cen)))
       X_onetime = utils.X_update(Y_onetime, cen[:,rank], cen_above[:,rank], delta, tau_sqd, Loc[:,rank], Scale[:,rank], Shape[:,rank])
      
       # Update Z
       for cluster_num in np.arange(n_clusters):
             Z_1t_accept[cluster_num] += utils.update_Z_1t_one_cluster_interp(Z_onetime, Cluster_which, cluster_num, Cor_Z_clusters, inv_Z_cluster, inv_Z_cluster_proposal,
                                 Y_onetime, X_onetime, R_onetime, cen[:,rank], cen_above[:,rank], prob_below, prob_above, delta, tau_sqd,
                                 Loc[:,rank], Scale[:,rank], Shape[:,rank], xp, den_p, thresh_X, thresh_X_above,
                                 sigma_m_Z_cluster[cluster_num], random_generator)
       Z_within_thinning[:, index_within] = Z_onetime
       
       # Update R
       Metr_R = sampler.static_metr(Y_onetime, R_onetime, utils.Rt_update_mixture_me_likelihood_interp,
                           priors.R_prior, 1, 2,
                           random_generator,
                           np.nan, sigma_m['R_1t'], False,
                           X_onetime, Z_onetime,
                           cen[:,rank], cen_above[:,rank], prob_below, prob_above,
                           Loc[:,rank], Scale[:,rank], Shape[:,rank], delta, tau_sqd,
                           xp, den_p, thresh_X, thresh_X_above)
       R_accept = R_accept + Metr_R['acc_prob']
       R_onetime = Metr_R['trace'][0,1]
       
       X_s_onetime = (R_onetime**(delta/(1-delta)))*utils.norm_to_Pareto(Z_onetime)

       # *** Gather items ***
       X_s_recv = comm.gather(X_s_onetime,root=0)
       X_recv = comm.gather(X_onetime, root=0)
       Z_recv = comm.gather(Z_onetime, root=0)
       R_recv = comm.gather(R_onetime, root=0)

       if rank==0:
           X_s[:] = np.vstack(X_s_recv).T
           X[:] = np.vstack(X_recv).T
           Z[:] = np.vstack(Z_recv).T
           R[:] = R_recv
           
           # print('beta_shape_accept=',beta_shape_accept, ', iter=', iter)

           # Update delta
           Metr_delta = sampler.static_metr(Y, delta, utils.delta_update_mixture_me_likelihood_interp, priors.interval_unif,
                   hyper_params_delta, 2,
                   random_generator,
                   np.nan, sigma_m['delta'], False,
                   R, Z, cen, cen_above, prob_below, prob_above, Loc, Scale, Shape, tau_sqd)
           delta_accept = delta_accept + Metr_delta['acc_prob']
           delta = Metr_delta['trace'][0,1]
           X_s[:] = (R**(delta/(1-delta)))*utils.norm_to_Pareto(Z)
           
           # Update tau_sqd
           Metr_tau_sqd = sampler.static_metr(Y, tau_sqd, utils.tau_update_mixture_me_likelihood_interp, priors.invGamma_prior,
                           hyper_params_tau_sqd, 2,
                           random_generator,
                           np.nan, sigma_m['tau_sqd'], False,
                           X_s, cen, cen_above, prob_below, prob_above, Loc, Scale, Shape, delta)
           tau_sqd_accept = tau_sqd_accept + Metr_tau_sqd['acc_prob']
           tau_sqd = Metr_tau_sqd['trace'][0,1]
           
           grid = utils.density_interp_grid(delta, tau_sqd)
           xp = grid[0]; den_p = grid[1]
           
           if prob_below==0:
               thresh_X = -np.inf
           else:
               thresh_X = utils.qmixture_me_interp(prob_below, delta = delta, tau_sqd = tau_sqd)
           if prob_above==1:
               thresh_X_above = np.inf
           else:
               thresh_X_above = utils.qmixture_me_interp(prob_above, delta = delta, tau_sqd = tau_sqd)
           
           # Update theta_c
           Metr_theta_c = sampler.static_metr(Z, theta_c, utils.theta_c_update_mixture_me_likelihood,
                             priors.interval_unif_multi, hyper_params_theta_c, 2,
                             random_generator,
                             prop_sigma['theta_c'], sigma_m['theta_c'], False,
                             1, Cluster_which, S_clusters)
           theta_c_accept = theta_c_accept + Metr_theta_c['acc_prob']
           theta_c = Metr_theta_c['trace'][:,1]
           theta_c_trace_within_thinning[:,index_within] = theta_c
          
           if Metr_theta_c['acc_prob']>0:
               Cor_Z_clusters=list()
               inv_Z_cluster=list()
               for i in np.arange(n_clusters):
                   Cor_tmp = utils.corr_fn(S_clusters[i], theta_c)
                   cholesky_inv = (cholesky(Cor_tmp,lower=False),np.repeat(1,Cor_tmp.shape[0]))
                   Cor_Z_clusters.append(Cor_tmp)
                   inv_Z_cluster.append(cholesky_inv)
           
           # Update mu_loc0 and beta_loc0
           Metr_mu_loc0 = sampler.static_metr(Y, mu_loc0, utils.mu_loc0_vec_gev_update_mixture_me_likelihood_interp, priors.interval_unif,
                   hyper_params_mu_loc0, 2,
                   random_generator,
                   np.nan, sigma_m['mu_loc0'], False,
                   Design_mat, beta_loc0, sbeta_loc0, X_s, cen, cen_above, prob_below, prob_above,
                   delta, tau_sqd, loc1, Scale, Shape, Time, xp, den_p, thresh_X, thresh_X_above)
           mu_loc0_accept = mu_loc0_accept + Metr_mu_loc0['acc_prob']
           mu_loc0 = Metr_mu_loc0['trace'][0,1]

           for cluster_num in np.arange(n_beta_clusters):
               beta_loc0_accept[cluster_num] += utils.update_beta_loc0_GEV_one_cluster_interp(beta_loc0, betaCluster_which, cluster_num, inv_beta_loc0_cluster_proposal,
                                                                            Design_mat, mu_loc0, sbeta_loc0, Y, X_s, cen, cen_above, prob_below, prob_above, delta, tau_sqd,
                                                                            loc1, Scale, Shape, Time, xp, den_p, thresh_X, thresh_X_above, 
                                                                            sigma_m_beta_loc0_cluster[cluster_num], random_generator)
           beta_loc0_within_thinning[:, index_within] = beta_loc0
           loc0 = mu_loc0+Design_mat @beta_loc0
           
           
           # Update mu_loc1 and beta_loc1
           Metr_mu_loc1 = sampler.static_metr(Y, mu_loc1, utils.mu_loc1_vec_gev_update_mixture_me_likelihood_interp, priors.interval_unif,
                   hyper_params_mu_loc1, 2,
                   random_generator,
                   np.nan, sigma_m['mu_loc1'], False,
                   Design_mat, beta_loc1, sbeta_loc1, X_s, cen, cen_above, prob_below, prob_above,
                   delta, tau_sqd, loc0, Scale, Shape, Time, xp, den_p, thresh_X, thresh_X_above)
           mu_loc1_accept = mu_loc1_accept + Metr_mu_loc1['acc_prob']
           mu_loc1 = Metr_mu_loc1['trace'][0,1]
           
           for cluster_num in np.arange(n_beta_clusters):
               beta_loc1_accept[cluster_num] += utils.update_beta_loc1_GEV_one_cluster_interp(beta_loc1, betaCluster_which, cluster_num, inv_beta_loc1_cluster_proposal,
                                                                            Design_mat, mu_loc1, sbeta_loc1, Y, X_s, cen, cen_above, prob_below, prob_above, delta, tau_sqd,
                                                                            loc0, Scale, Shape, Time, xp, den_p, thresh_X, thresh_X_above,
                                                                            sigma_m_beta_loc1_cluster[cluster_num], random_generator)
           beta_loc1_within_thinning[:, index_within] = beta_loc1
           loc1 = mu_loc1+Design_mat @beta_loc1
           Loc = np.tile(loc0, n_t) + np.tile(loc1, n_t)*np.repeat(Time,n_s)
           Loc = Loc.reshape((n_s,n_t),order='F')
           
           # Update mu_scale and beta_scale
           Metr_mu_scale = sampler.static_metr(Y, mu_scale, utils.mu_scale_vec_gev_update_mixture_me_likelihood_interp, priors.interval_unif,
                   hyper_params_mu_scale, 2,
                   random_generator,
                   np.nan, sigma_m['mu_scale'], False,
                   Design_mat, beta_scale, sbeta_scale, X_s, cen, cen_above, prob_below, prob_above,
                   delta, tau_sqd, Loc, Shape, Time, xp, den_p, thresh_X, thresh_X_above)
           mu_scale_accept = mu_scale_accept + Metr_mu_scale['acc_prob']
           mu_scale = Metr_mu_scale['trace'][0,1]
           
           for cluster_num in np.arange(n_beta_clusters):
               beta_scale_accept[cluster_num] += utils.update_beta_scale_GEV_one_cluster_interp(beta_scale, betaCluster_which, cluster_num, inv_beta_scale_cluster_proposal,
                                                                            Design_mat, mu_scale, sbeta_scale, Y, X_s, cen, cen_above, prob_below, prob_above, delta, tau_sqd,
                                                                            Loc, Shape, Time, xp, den_p, thresh_X, thresh_X_above, 
                                                                            sigma_m_beta_scale_cluster[cluster_num], random_generator)
           beta_scale_within_thinning[:, index_within] = beta_scale
           scale = np.exp(mu_scale+Design_mat @beta_scale)
           Scale = np.tile(scale, n_t)
           Scale = Scale.reshape((n_s,n_t),order='F')
            
            
           # Update beta_scale and beta_shape
           Metr_mu_shape = sampler.static_metr(Y, mu_shape, utils.mu_shape_vec_gev_update_mixture_me_likelihood_interp, priors.interval_unif,
                   hyper_params_mu_shape, 2,
                   random_generator,
                   np.nan, sigma_m['mu_shape'], False,
                   Design_mat, beta_shape, sbeta_shape, X_s, cen, cen_above, prob_below, prob_above,
                   delta, tau_sqd, Loc, Scale, Time, xp, den_p, thresh_X, thresh_X_above)
           mu_shape_accept = mu_shape_accept + Metr_mu_shape['acc_prob']
           mu_shape = Metr_mu_shape['trace'][0,1]
           
           for cluster_num in np.arange(n_beta_clusters):
               beta_shape_accept[cluster_num] += utils.update_beta_shape_GEV_one_cluster_interp(beta_shape, betaCluster_which, cluster_num, inv_beta_shape_cluster_proposal,
                                                                            Design_mat, mu_shape, sbeta_shape, Y, X_s, cen, cen_above, prob_below, prob_above, delta, tau_sqd,
                                                                            Loc, Scale, Time, xp, den_p, thresh_X, thresh_X_above,
                                                                            sigma_m_beta_shape_cluster[cluster_num], random_generator)
           beta_shape_within_thinning[:, index_within] = beta_shape
           shape = mu_shape+Design_mat @beta_shape
           Shape = np.tile(shape, n_t)
           Shape = Shape.reshape((n_s,n_t),order='F')
           
           # Update sbeta_loc0
           Metr_sbeta_loc0 = sampler.static_metr(beta_loc0, sbeta_loc0, utils.dmvn_diag, priors.half_cauchy,
                   hyper_params_sbeta_loc0, 2,
                   random_generator,
                   np.nan, sigma_m['sbeta_loc0'], False,
                   0)
           sbeta_loc0_accept = sbeta_loc0_accept + Metr_sbeta_loc0['acc_prob']
           sbeta_loc0 = Metr_sbeta_loc0['trace'][0,1]
           
           # Update sbeta_loc1
           Metr_sbeta_loc1 = sampler.static_metr(beta_loc1, sbeta_loc1, utils.dmvn_diag, priors.half_cauchy,
                   hyper_params_sbeta_loc1, 2,
                   random_generator,
                   np.nan, sigma_m['sbeta_loc1'], False,
                   0)
           sbeta_loc1_accept = sbeta_loc1_accept + Metr_sbeta_loc1['acc_prob']
           sbeta_loc1 = Metr_sbeta_loc1['trace'][0,1]
           
           # Update sbeta_scale
           Metr_sbeta_scale = sampler.static_metr(beta_scale, sbeta_scale, utils.dmvn_diag, priors.half_cauchy,
                   hyper_params_sbeta_scale, 2,
                   random_generator,
                   np.nan, sigma_m['sbeta_scale'], False,
                   0)
           sbeta_scale_accept = sbeta_scale_accept + Metr_sbeta_scale['acc_prob']
           sbeta_scale = Metr_sbeta_scale['trace'][0,1]
           
           # Update sbeta_shape
           Metr_sbeta_shape = sampler.static_metr(beta_shape, sbeta_shape, utils.dmvn_diag, priors.half_cauchy,
                   hyper_params_sbeta_shape, 2,
                   random_generator,
                   np.nan, sigma_m['sbeta_shape'], False,
                   0)
           sbeta_shape_accept = sbeta_shape_accept + Metr_sbeta_shape['acc_prob']
           sbeta_shape = Metr_sbeta_shape['trace'][0,1]
           # cen[:] = utils.which_censored(Y, Loc, Scale, Shape, prob_below)
           # cen_above[:] = utils.which_censored(Y, Loc, Scale, Shape, prob_above)
           
           # print(str(iter)+" Freshly updated: "+str(np.where(~cen)))
       # *** Broadcast items ***
       delta = comm.bcast(delta,root=0)
       tau_sqd = comm.bcast(tau_sqd,root=0)
       xp = comm.bcast(xp,root=0)
       den_p = comm.bcast(den_p,root=0)
       thresh_X = comm.bcast(thresh_X,root=0)
       thresh_X_above = comm.bcast(thresh_X_above,root=0)
       theta_c = comm.bcast(theta_c,root=0)
       # V = comm.bcast(V,root=0)
       # d = comm.bcast(d,root=0)
       Cor_Z_clusters = comm.bcast(Cor_Z_clusters,root=0)
       inv_Z_cluster = comm.bcast(inv_Z_cluster,root=0)
       Loc = comm.bcast(Loc,root=0)
       Scale = comm.bcast(Scale,root=0)
       Shape = comm.bcast(Shape,root=0)
       # cen = comm.bcast(cen,root=0)
       # cen_above = comm.bcast(cen_above,root=0)
      
      
       # ----------------------------------------------------------------------------------------
       # --------------------------- Summarize every 'thinning' steps ---------------------------
       # ----------------------------------------------------------------------------------------
       if (iter % thinning) == 0:
           index = np.int(iter/thinning)
           
           # Fill in trace objects
           Z_1t_trace[:,index] = Z_onetime
           R_1t_trace[index] = R_onetime
           if rank == 0:
               delta_trace[index] = delta
               tau_sqd_trace[index] = tau_sqd
               theta_c_trace[:,index] = theta_c
               beta_loc0_trace[:,index] = beta_loc0
               beta_loc1_trace[:,index] = beta_loc1
               beta_scale_trace[:,index] = beta_scale
               beta_shape_trace[:,index] = beta_shape
               sigma_sbeta_loc0_trace[index] = sbeta_loc0
               sigma_sbeta_loc1_trace[index] = sbeta_loc1
               sigma_sbeta_scale_trace[index] = sbeta_scale
               sigma_sbeta_shape_trace[index] = sbeta_shape
               
               mu_loc0_trace[index] = mu_loc0
               mu_loc1_trace[index] = mu_loc1
               mu_scale_trace[index] = mu_scale
               mu_shape_trace[index] = mu_shape
               
               loc0_trace[index,:] = loc0
               loc1_trace[index,:] = loc1
               scale_trace[index,:] = scale
               shape_trace[index,:] = shape
          
            
           # Adapt via Shaby and Wells (2010)
           gamma2 = 1 / (index + offset)**(c_1)
           gamma1 = c_0*gamma2
           sigma_m_Z_cluster[:] = np.exp(np.log(sigma_m_Z_cluster) + gamma1*(Z_1t_accept/thinning - r_opt_md))
           Z_1t_accept[:] = np.repeat(0,n_clusters)
           inv_Z_cluster_proposal=list()
           for i in np.arange(n_clusters):
               which_tmp = Cluster_which[i]
               sigma_Z_cluster_proposal[i] = sigma_Z_cluster_proposal[i] + gamma2*(np.cov(Z_within_thinning[which_tmp,:]) - sigma_Z_cluster_proposal[i])
               inv_Z_cluster_proposal.append((cholesky(sigma_Z_cluster_proposal[i],lower=False),np.repeat(1,np.sum(which_tmp))))           

           sigma_m['R_1t'] = np.exp(np.log(sigma_m['R_1t']) + gamma1*(R_accept/thinning - r_opt_1d))
           R_accept = 0
          
           if rank == 0:
               sigma_m['delta'] = np.exp(np.log(sigma_m['delta']) + gamma1*(delta_accept/thinning - r_opt_1d))
               delta_accept = 0
               sigma_m['tau_sqd'] = np.exp(np.log(sigma_m['tau_sqd']) + gamma1*(tau_sqd_accept/thinning - r_opt_1d))
               tau_sqd_accept = 0
          
               sigma_m['theta_c'] = np.exp(np.log(sigma_m['theta_c']) + gamma1*(theta_c_accept/thinning - r_opt_2d))
               theta_c_accept = 0
               prop_sigma['theta_c'] = prop_sigma['theta_c'] + gamma2*(np.cov(theta_c_trace_within_thinning) - prop_sigma['theta_c'])
               check_chol_cont = True
               while check_chol_cont:
                   try:
                       # Initialize prop_C
                       np.linalg.cholesky(prop_sigma['theta_c'])
                       check_chol_cont = False
                   except  np.linalg.LinAlgError:
                       prop_sigma['theta_c'] = prop_sigma['theta_c'] + eps*np.eye(2)
                       print("Oops. Proposal covariance matrix is now:\n")
                       print(prop_sigma['theta_c'])
              
                
               sigma_m_beta_loc0_cluster[:] = np.exp(np.log(sigma_m_beta_loc0_cluster) + gamma1*(beta_loc0_accept/thinning - r_opt_md))
               beta_loc0_accept[:] = np.repeat(0,n_beta_clusters)
               inv_beta_loc0_cluster_proposal=list()
               for i in np.arange(n_beta_clusters):
                   which_tmp = betaCluster_which[i]
                   sigma_beta_loc0_cluster_proposal[i] = sigma_beta_loc0_cluster_proposal[i] + gamma2*(np.cov(beta_loc0_within_thinning[which_tmp,:]) - sigma_beta_loc0_cluster_proposal[i])
                   inv_beta_loc0_cluster_proposal.append((cholesky(sigma_beta_loc0_cluster_proposal[i],lower=False),np.repeat(1,np.sum(which_tmp))))
                   
               # print(' Done with '+str(index)+", mean accept="+str(np.mean(loc0_accept))+", mean sigma_m_loc0="+str(np.mean(sigma_m_beta_loc0_cluster))+",\n")
               sigma_m['mu_loc0'] = np.exp(np.log(sigma_m['mu_loc0']) + gamma1*(mu_loc0_accept/thinning - r_opt_1d))
               mu_loc0_accept = 0
               
               sigma_m_beta_loc1_cluster[:] = np.exp(np.log(sigma_m_beta_loc1_cluster) + gamma1*(beta_loc1_accept/thinning - r_opt_md))
               beta_loc1_accept[:] = np.repeat(0,n_beta_clusters)
               inv_beta_loc1_cluster_proposal=list()
               for i in np.arange(n_beta_clusters):
                   which_tmp = betaCluster_which[i]
                   sigma_beta_loc1_cluster_proposal[i] = sigma_beta_loc1_cluster_proposal[i] + gamma2*(np.cov(beta_loc1_within_thinning[which_tmp,:]) - sigma_beta_loc1_cluster_proposal[i])
                   inv_beta_loc1_cluster_proposal.append((cholesky(sigma_beta_loc1_cluster_proposal[i],lower=False),np.repeat(1,np.sum(which_tmp))))           
             
               sigma_m['mu_loc1'] = np.exp(np.log(sigma_m['mu_loc1']) + gamma1*(mu_loc1_accept/thinning - r_opt_1d))
               mu_loc1_accept = 0
               
               sigma_m_beta_scale_cluster[:] = np.exp(np.log(sigma_m_beta_scale_cluster) + gamma1*(beta_scale_accept/thinning - r_opt_md))
               beta_scale_accept[:] = np.repeat(0,n_beta_clusters)
               inv_beta_scale_cluster_proposal=list()
               for i in np.arange(n_beta_clusters):
                   which_tmp = betaCluster_which[i]
                   sigma_beta_scale_cluster_proposal[i] = sigma_beta_scale_cluster_proposal[i] + gamma2*(np.cov(beta_scale_within_thinning[which_tmp,:]) - sigma_beta_scale_cluster_proposal[i])
                   inv_beta_scale_cluster_proposal.append((cholesky(sigma_beta_scale_cluster_proposal[i],lower=False),np.repeat(1,np.sum(which_tmp))))
 
               sigma_m['mu_scale'] = np.exp(np.log(sigma_m['mu_scale']) + gamma1*(mu_scale_accept/thinning - r_opt_1d))
               mu_scale_accept = 0
               
               sigma_m_beta_shape_cluster[:] = np.exp(np.log(sigma_m_beta_shape_cluster) + gamma1*(beta_shape_accept/thinning - r_opt_md))
               beta_shape_accept[:] = np.repeat(0,n_beta_clusters)
               inv_beta_shape_cluster_proposal=list()
               for i in np.arange(n_beta_clusters):
                   which_tmp = betaCluster_which[i]
                   sigma_beta_shape_cluster_proposal[i] = sigma_beta_shape_cluster_proposal[i] + gamma2*(np.cov(beta_shape_within_thinning[which_tmp,:]) - sigma_beta_shape_cluster_proposal[i])
                   inv_beta_shape_cluster_proposal.append((cholesky(sigma_beta_shape_cluster_proposal[i],lower=False),np.repeat(1,np.sum(which_tmp))))    
                   
               sigma_m['mu_shape'] = np.exp(np.log(sigma_m['mu_shape']) + gamma1*(mu_shape_accept/thinning - r_opt_1d))
               mu_shape_accept = 0

               sigma_m['sbeta_loc0'] = np.exp(np.log(sigma_m['sbeta_loc0']) + gamma1*(sbeta_loc0_accept/thinning - r_opt_1d))
               sbeta_loc0_accept = 0
               
               sigma_m['sbeta_loc1'] = np.exp(np.log(sigma_m['sbeta_loc1']) + gamma1*(sbeta_loc1_accept/thinning - r_opt_1d))
               sbeta_loc1_accept = 0
               
               sigma_m['sbeta_scale'] = np.exp(np.log(sigma_m['sbeta_scale']) + gamma1*(sbeta_scale_accept/thinning - r_opt_1d))
               sbeta_scale_accept = 0
               
               sigma_m['sbeta_scale'] = np.exp(np.log(sigma_m['sbeta_scale']) + gamma1*(sbeta_scale_accept/thinning - r_opt_1d))
               sbeta_scale_accept = 0                   
                   
          
       # ----------------------------------------------------------------------------------------
       # -------------------------- Echo & save every 'thinning' steps --------------------------
       # ----------------------------------------------------------------------------------------
       if (iter / thinning) % echo_interval == 0:
           # print(rank, iter)
           if rank == 0:
               print('Done with '+str(index)+" updates while thinned by "+str(thinning)+" steps,\n")
               
               # Save the intermediate results to filename
               initial_values = {'delta':delta,
                    'tau_sqd':tau_sqd,
                    'prob_below':prob_below,
                    'prob_above':prob_above,
                    'Dist':Dist,
                    'theta_c':theta_c,
                    'X':X,
                    'Z':Z,
                    'R':R,
                    'loc0':loc0,
                    'loc1':loc1,
                    'scale':scale,
                    'shape':shape,
                    'Design_mat':Design_mat,
                    'beta_loc0':beta_loc0,
                    'beta_loc1':beta_loc1,
                    'Time':Time,
                    'beta_scale':beta_scale,
                    'beta_shape':beta_shape,
                    'mu_loc0':mu_loc0,
                    'mu_loc1':mu_loc1,
                    'mu_scale':mu_scale,
                    'mu_shape':mu_shape,
                    'sbeta_loc0':sbeta_loc0,
                    'sbeta_loc1':sbeta_loc1,
                    'sbeta_scale':sbeta_scale,
                    'sbeta_shape':sbeta_shape,
                    'Cluster_which':Cluster_which,
                    'S_clusters':S_clusters,
                    'betaCluster_which':betaCluster_which
                    }
               with open(filename, 'wb') as f:
                   dump(Y, f)
                   dump(cen, f)
                   dump(cen_above,f)
                   dump(initial_values, f)
                   dump(sigma_m, f)
                   dump(prop_sigma, f)
                   dump(iter, f)
                   dump(delta_trace, f)
                   dump(tau_sqd_trace, f)
                   dump(theta_c_trace, f)
                   dump(beta_loc0_trace, f)
                   dump(beta_loc1_trace, f)
                   dump(beta_scale_trace, f)
                   dump(beta_shape_trace, f)
                   dump(sigma_sbeta_loc0_trace,f)
                   dump(sigma_sbeta_loc1_trace,f)
                   dump(sigma_sbeta_scale_trace,f)
                   dump(sigma_sbeta_shape_trace,f)
                   
                   dump(mu_loc0_trace,f)
                   dump(mu_loc1_trace,f)
                   dump(mu_scale_trace,f)
                   dump(mu_shape_trace,f)
                   
                   dump(loc0_trace,f)
                   dump(loc1_trace,f)
                   dump(scale_trace,f)
                   dump(shape_trace,f)
                   
                   dump(Z_1t_trace, f)
                   dump(R_1t_trace, f)
                   dump(Y_onetime, f)
                   dump(X_onetime, f)
                   dump(X_s_onetime, f)
                   dump(R_onetime, f)
                   
                   dump(sigma_m_Z_cluster, f)
                   dump(sigma_m_beta_loc0_cluster, f)
                   dump(sigma_m_beta_loc1_cluster, f)
                   dump(sigma_m_beta_scale_cluster, f)
                   dump(sigma_m_beta_shape_cluster, f)
                   
                   dump(sigma_beta_loc0_cluster_proposal, f)
                   dump(sigma_beta_loc1_cluster_proposal, f)
                   dump(sigma_beta_scale_cluster_proposal, f)
                   dump(sigma_beta_shape_cluster_proposal, f)
                   dump(sigma_Z_cluster_proposal, f)
                   f.close()
                   
               # Echo trace plots
               pdf_pages = PdfPages('./progress.pdf')
               grid_size = (4,2)
               #-page-1
               fig = plt.figure(figsize = (8.75, 11.75))
               plt.subplot2grid(grid_size, (0,0)) # delta
               plt.plot(delta_trace, color='gray', linestyle='solid')
               plt.ylabel(r'$\delta$')
               plt.subplot2grid(grid_size, (0,1)) # tau_sqd
               plt.plot(tau_sqd_trace, color='gray', linestyle='solid')
               plt.ylabel(r'$\tau^2$')
               plt.subplot2grid(grid_size, (1,0)) # rho
               plt.plot(theta_c_trace[0,:], color='gray', linestyle='solid')
               plt.ylabel(r'Matern $\rho$')
               plt.subplot2grid(grid_size, (1,1)) # nu
               plt.plot(theta_c_trace[1,:], color='gray', linestyle='solid')
               plt.ylabel(r'Matern $\nu$')
               plt.subplot2grid(grid_size, (2,0)) # mu0: beta_0
               plt.plot(beta_loc0_trace[0,:], color='gray', linestyle='solid')
               plt.ylabel(r'Location $\mu_0$: $\beta_0$')
               plt.subplot2grid(grid_size, (2,1)) # mu0: beta_1
               plt.plot(beta_loc0_trace[50,:], color='gray', linestyle='solid')
               plt.ylabel(r'Location $\mu_0$: $\beta_1$')
               plt.subplot2grid(grid_size, (3,0)) # mu1: beta_0
               plt.plot(beta_loc1_trace[0,:], color='gray', linestyle='solid')
               plt.ylabel(r'Location $\mu_1$: $\beta_0$')
               plt.subplot2grid(grid_size, (3,1)) # mu1: beta_1
               plt.plot(beta_loc1_trace[50,:], color='gray', linestyle='solid')
               plt.ylabel(r'Location $\mu_1$: $\beta_1$')
               plt.tight_layout()
               pdf_pages.savefig(fig)
               plt.close()
                   
               #-page-2
               fig = plt.figure(figsize = (8.75, 11.75))
               plt.subplot2grid(grid_size, (0,0)) # scale: beta_0
               plt.plot(beta_scale_trace[0,:], color='gray', linestyle='solid')
               plt.ylabel(r'Scale $\sigma$: $\beta_0$')
               plt.subplot2grid(grid_size, (0,1)) # scale: beta_1
               plt.plot(beta_scale_trace[1,:], color='gray', linestyle='solid')
               plt.ylabel(r'Scale $\sigma$: $\beta_1$')
               plt.subplot2grid(grid_size, (1,0))  # shape: beta_0
               plt.plot(beta_shape_trace[0,:], color='gray', linestyle='solid')
               plt.ylabel(r'Shape $\xi$: $\beta_0$')
               plt.subplot2grid(grid_size, (1,1))  # shape: beta_1
               plt.plot(beta_shape_trace[1,:], color='gray', linestyle='solid')
               plt.ylabel(r'Shape $\xi$: $\beta_1$')
               plt.subplot2grid(grid_size, (2,0))   # X^*
               plt.plot(Z_1t_trace[1,:], color='gray', linestyle='solid')
               plt.ylabel(r'$Z$'+'['+str(1)+","+str(rank)+']')
               where = [(2,1),(3,0),(3,1)]
               for wh_sub,i in enumerate(wh_to_plot_Xs):
                   plt.subplot2grid(grid_size, where[wh_sub]) # X^*
                   plt.plot(Z_1t_trace[i,:], color='gray', linestyle='solid')
                   plt.ylabel(r'$Z$'+'['+str(i)+","+str(rank)+']')
               plt.tight_layout()
               pdf_pages.savefig(fig)
               plt.close()
               

               #-page-3
               fig = plt.figure(figsize = (8.75, 11.75))
               plt.subplot2grid(grid_size, (0,0))
               plt.plot(sigma_sbeta_loc0_trace[:], color='gray', linestyle='solid')
               plt.ylabel(r'$\sigma^2_{\beta}(loc0)$')
               plt.subplot2grid(grid_size, (0,1))
               plt.plot(sigma_sbeta_loc1_trace[:], color='gray', linestyle='solid')
               plt.ylabel(r'$\sigma^2_{\beta}(loc1)$')
               plt.subplot2grid(grid_size, (1,0))
               plt.plot(sigma_sbeta_scale_trace[:], color='gray', linestyle='solid')
               plt.ylabel(r'$\sigma^2_{\beta}(scale)$')
               plt.subplot2grid(grid_size, (1,1))
               plt.plot(sigma_sbeta_shape_trace[:], color='gray', linestyle='solid')
               plt.ylabel(r'$\sigma^2_{\beta}(shape)$')
               plt.subplot2grid(grid_size, (2,0)) # loc0
               plt.plot(loc0_trace[:,wh_to_plot_Xs[0]], color='gray', linestyle='solid')
               plt.ylabel(r'loc0'+'['+str(wh_to_plot_Xs[0])+']')
               plt.subplot2grid(grid_size, (2,1)) # loc1
               plt.plot(loc1_trace[:,wh_to_plot_Xs[0]], color='gray', linestyle='solid')
               plt.ylabel(r'loc1'+'['+str(wh_to_plot_Xs[0])+']')
               plt.subplot2grid(grid_size, (3,0)) # scale
               plt.plot(scale_trace[:,wh_to_plot_Xs[0]], color='gray', linestyle='solid')
               plt.ylabel(r'scale'+'['+str(wh_to_plot_Xs[0])+']')
               plt.subplot2grid(grid_size, (3,1)) # shape
               plt.plot(shape_trace[:,wh_to_plot_Xs[0]], color='gray', linestyle='solid')
               plt.ylabel(r'shape'+'['+str(wh_to_plot_Xs[0])+']')
               plt.tight_layout()
               pdf_pages.savefig(fig)
               plt.close()
               
               #-page-4
               fig = plt.figure(figsize = (8.75, 11.75))
               plt.subplot2grid(grid_size, (0,0)) # loc0
               plt.plot(loc0_trace[:,wh_to_plot_Xs[1]], color='gray', linestyle='solid')
               plt.ylabel(r'loc0'+'['+str(wh_to_plot_Xs[1])+']')
               plt.subplot2grid(grid_size, (0,1)) # loc1
               plt.plot(loc1_trace[:,wh_to_plot_Xs[1]], color='gray', linestyle='solid')
               plt.ylabel(r'loc1'+'['+str(wh_to_plot_Xs[1])+']')
               plt.subplot2grid(grid_size, (1,0)) # scale
               plt.plot(scale_trace[:,wh_to_plot_Xs[1]], color='gray', linestyle='solid')
               plt.ylabel(r'scale'+'['+str(wh_to_plot_Xs[1])+']')
               plt.subplot2grid(grid_size, (1,1)) # shape
               plt.plot(shape_trace[:,wh_to_plot_Xs[1]], color='gray', linestyle='solid')
               plt.ylabel(r'shape'+'['+str(wh_to_plot_Xs[1])+']')
               plt.subplot2grid(grid_size, (2,0)) # loc0
               plt.plot(loc0_trace[:,wh_to_plot_Xs[2]], color='gray', linestyle='solid')
               plt.ylabel(r'loc0'+'['+str(wh_to_plot_Xs[2])+']')
               plt.subplot2grid(grid_size, (2,1)) # loc1
               plt.plot(loc1_trace[:,wh_to_plot_Xs[2]], color='gray', linestyle='solid')
               plt.ylabel(r'loc1'+'['+str(wh_to_plot_Xs[2])+']')
               plt.subplot2grid(grid_size, (3,0)) # scale
               plt.plot(scale_trace[:,wh_to_plot_Xs[2]], color='gray', linestyle='solid')
               plt.ylabel(r'scale'+'['+str(wh_to_plot_Xs[2])+']')
               plt.subplot2grid(grid_size, (3,1)) # shape
               plt.plot(shape_trace[:,wh_to_plot_Xs[2]], color='gray', linestyle='solid')
               plt.ylabel(r'shape'+'['+str(wh_to_plot_Xs[2])+']')
               plt.tight_layout()
               pdf_pages.savefig(fig)
               plt.close()
               pdf_pages.close()
           else:
               with open(filename, 'wb') as f:
                   dump(Y, f)
                   dump(cen, f)
                   dump(cen_above,f)
                   dump(initial_values, f)
                   dump(sigma_m, f)
                   dump(sigma_m_Z_cluster, f)
                   dump(iter, f)
                   dump(Z_1t_trace, f)
                   dump(R_1t_trace, f)
                   dump(Y_onetime, f)
                   dump(X_onetime, f)
                   dump(X_s_onetime, f)
                   dump(R_onetime, f)
                   dump(sigma_Z_cluster_proposal, f)
                   f.close()
               
