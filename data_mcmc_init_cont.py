#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 12:08:19 2021

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
   import scalemixture_temp.integrate as utils
   import scalemixture_temp.priors as priors
   import scalemixture_temp.generic_samplers as sampler
   import os
   import numpy as np
   import matplotlib.pyplot as plt
   from matplotlib.backends.backend_pdf import PdfPages
   from pickle import load
   from pickle import dump
   # from scipy.stats import norm
   # from scipy.stats import invgamma  
   from scipy.linalg import cholesky
   np.seterr(invalid='ignore', over = 'ignore')
   
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
   # rank=0; size=68
   thinning = 10; echo_interval = 50; n_updates = 40001
     
   # Filename for storing the intermediate results
   filename='./scalemix_progress_'+str(rank)+'.pkl'
   
   n_updates_thinned = np.int(np.ceil(n_updates/thinning))
   # Load data input
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
           beta_loc2_trace = load(f)
           beta_loc3_trace = load(f)
           beta_logscale0_trace = load(f)
           beta_logscale1_trace = load(f)
           beta_logscale2_trace = load(f)
           beta_shape_trace = load(f)
           sigma_sbeta_loc0_trace = load(f)
           sigma_sbeta_loc1_trace = load(f)
           sigma_sbeta_loc2_trace = load(f)
           sigma_sbeta_loc3_trace = load(f)
           sigma_sbeta_logscale0_trace = load(f)
           sigma_sbeta_logscale1_trace = load(f)
           sigma_sbeta_logscale2_trace = load(f)
           sigma_sbeta_shape_trace = load(f)
           
           loc0_trace = load(f)
           loc1_trace = load(f)
           loc2_trace = load(f)
           loc3_trace = load(f)
           logscale0_trace = load(f)
           logscale1_trace = load(f)           
           logscale2_trace = load(f)
           shape_trace = load(f)
           
           Z_1t_trace = load(f)
           R_1t_trace = load(f)
           Y_onetime = load(f)
           X_onetime =load(f)
           X_s_onetime = load(f)
           R_onetime = load(f)
           
           sigma_m_Z_cluster = load(f)
           sigma_m_beta_loc0_cluster = load(f)
           sigma_m_beta_loc1_cluster = load(f)
           sigma_m_beta_loc2_cluster = load(f)
           sigma_m_beta_loc3_cluster = load(f)
           sigma_m_beta_logscale0_cluster = load(f)
           sigma_m_beta_logscale1_cluster = load(f)
           sigma_m_beta_logscale2_cluster = load(f)
           sigma_m_beta_shape_cluster = load(f)
        
           sigma_beta_loc0_cluster_proposal = load(f)
           sigma_beta_loc1_cluster_proposal = load(f)
           sigma_beta_loc2_cluster_proposal = load(f)
           sigma_beta_loc3_cluster_proposal = load(f)
           sigma_beta_logscale0_cluster_proposal = load(f)
           sigma_beta_logscale1_cluster_proposal = load(f)
           sigma_beta_logscale2_cluster_proposal = load(f)
           sigma_beta_shape_cluster_proposal = load(f)
           sigma_Z_cluster_proposal_nonMissing = load(f)
           f.close()
           
           Z_onetime = Z_1t_trace[:,len(delta_trace)]
           if(len(delta_trace)<n_updates_thinned):
               add_length = n_updates_thinned - len(delta_trace)
                
               delta_trace = np.pad(delta_trace, (0, add_length), 'constant', constant_values=np.nan)
               tau_sqd_trace = np.pad(tau_sqd_trace, (0, add_length), 'constant', constant_values=np.nan)
               theta_c_trace = np.pad(theta_c_trace, ((0,0),(0,add_length)),'constant', constant_values=np.nan)
               beta_loc0_trace = np.pad(beta_loc0_trace, ((0,0),(0,add_length)),'constant', constant_values=np.nan)
               beta_loc1_trace = np.pad(beta_loc1_trace, ((0,0),(0,add_length)),'constant', constant_values=np.nan)
               beta_loc2_trace = np.pad(beta_loc2_trace, ((0,0),(0,add_length)),'constant', constant_values=np.nan)
               beta_loc3_trace = np.pad(beta_loc3_trace, ((0,0),(0,add_length)),'constant', constant_values=np.nan)
               beta_logscale0_trace = np.pad(beta_logscale0_trace, ((0,0),(0,add_length)),'constant', constant_values=np.nan)
               beta_logscale1_trace = np.pad(beta_logscale1_trace, ((0,0),(0,add_length)),'constant', constant_values=np.nan)
               beta_logscale2_trace = np.pad(beta_logscale2_trace, ((0,0),(0,add_length)),'constant', constant_values=np.nan)
               beta_shape_trace = np.pad(beta_shape_trace, ((0,0),(0,add_length)),'constant', constant_values=np.nan)
               sigma_sbeta_loc0_trace = np.pad(sigma_sbeta_loc0_trace, (0, add_length),'constant', constant_values=np.nan)
               sigma_sbeta_loc1_trace = np.pad(sigma_sbeta_loc1_trace, (0, add_length),'constant', constant_values=np.nan)
               sigma_sbeta_loc2_trace = np.pad(sigma_sbeta_loc2_trace, (0, add_length),'constant', constant_values=np.nan)
               sigma_sbeta_loc3_trace = np.pad(sigma_sbeta_loc3_trace, (0, add_length),'constant', constant_values=np.nan)
               sigma_sbeta_logscale0_trace = np.pad(sigma_sbeta_logscale0_trace, (0, add_length),'constant', constant_values=np.nan)
               sigma_sbeta_logscale1_trace = np.pad(sigma_sbeta_logscale1_trace, (0, add_length),'constant', constant_values=np.nan)
               sigma_sbeta_logscale2_trace = np.pad(sigma_sbeta_logscale2_trace, (0, add_length),'constant', constant_values=np.nan)
               sigma_sbeta_shape_trace = np.pad(sigma_sbeta_shape_trace, (0, add_length),'constant', constant_values=np.nan)
               
               loc0_trace = np.pad(loc0_trace, ((0,add_length),(0,0)),'constant', constant_values=np.nan)
               loc1_trace = np.pad(loc1_trace, ((0,add_length),(0,0)),'constant', constant_values=np.nan)
               loc2_trace = np.pad(loc2_trace, ((0,add_length),(0,0)),'constant', constant_values=np.nan)
               loc3_trace = np.pad(loc3_trace, ((0,add_length),(0,0)),'constant', constant_values=np.nan)
               logscale0_trace = np.pad(logscale0_trace, ((0,add_length),(0,0)),'constant', constant_values=np.nan)
               logscale1_trace = np.pad(logscale1_trace, ((0,add_length),(0,0)),'constant', constant_values=np.nan)
               logscale2_trace = np.pad(logscale2_trace, ((0,add_length),(0,0)),'constant', constant_values=np.nan)
               shape_trace = np.pad(shape_trace, ((0,add_length),(0,0)),'constant', constant_values=np.nan)
               
               Z_1t_trace = np.pad(Z_1t_trace, ((0,0),(0,add_length)),'constant', constant_values=np.nan)
               R_1t_trace = np.pad(R_1t_trace, (0, add_length), 'constant', constant_values=np.nan)
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
           sigma_Z_cluster_proposal_nonMissing = load(f)
           f.close()
           
           Z_onetime = Z_1t_trace[:,len(delta_trace)]
           if(len(delta_trace)<n_updates_thinned):
               add_length = n_updates_thinned - len(delta_trace)
               Z_1t_trace = np.pad(Z_1t_trace, ((0,0),(0,add_length)),'constant', constant_values=np.nan)
               R_1t_trace = np.pad(R_1t_trace, (0, add_length), 'constant', constant_values=np.nan)

     
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
   
   # hyper_params_mu_loc0 = np.array([-100,100])
   # hyper_params_mu_loc1 = np.array([-100,100])
   # hyper_params_mu_scale = np.array([-100,100])
   # hyper_params_mu_shape = np.array([-100,100])
   
   hyper_params_sbeta_loc0 = 1
   hyper_params_sbeta_loc1 = 1
   hyper_params_sbeta_loc2 = 1
   hyper_params_sbeta_loc3 = 1
   hyper_params_sbeta_logscale0 = 1
   hyper_params_sbeta_logscale1 = 1
   hyper_params_sbeta_logscale2 = 1
   hyper_params_sbeta_shape = 1
    
   ## -------------------------------------------------------
   ##                  Set initial values
   ## -------------------------------------------------------
   delta = initial_values['delta']
   tau_sqd = initial_values['tau_sqd']
   prob_below = initial_values['prob_below']
   prob_above = initial_values['prob_above']
   grid = utils.density_interp_grid(delta, tau_sqd)
   xp = grid[0]; den_p = grid[1]
      
   X = initial_values['X']
   Z = initial_values['Z']
   R = initial_values['R']

   # Y_onetime = Y[:,rank]
   # X_onetime = X[:,rank]
   # Z_onetime = Z[:,rank]
   # R_onetime = R[rank]
   
   loc0 = initial_values['loc0']
   loc1 = initial_values['loc1']
   loc2 = initial_values['loc2']
   loc3 = initial_values['loc3']
   logscale0 = initial_values['logscale0']
   logscale1 = initial_values['logscale1']
   logscale2 = initial_values['logscale2']
   shape = initial_values['shape']
   Design_mat = initial_values['Design_mat']
   WMGHGs = initial_values['WMGHGs']
   PDSI = initial_values['PDSI']
   ELI_summer_average = initial_values['ELI_summer_average']
   
   beta_loc0 = initial_values['beta_loc0']
   beta_loc1 = initial_values['beta_loc1']
   beta_loc2 = initial_values['beta_loc2']
   beta_loc3 = initial_values['beta_loc3']
   beta_logscale0 = initial_values['beta_logscale0']
   beta_logscale1 = initial_values['beta_logscale1']
   beta_logscale2 = initial_values['beta_logscale2']
   beta_shape = initial_values['beta_shape']
   # mu_loc0 = initial_values['mu_loc0']
   # mu_loc1 = initial_values['mu_loc1']
   # mu_scale = initial_values['mu_scale']
   # mu_shape = initial_values['mu_shape']
   
   
   theta_c = initial_values['theta_c']
   sbeta_loc0 = initial_values['sbeta_loc0']
   sbeta_loc1 = initial_values['sbeta_loc1']
   sbeta_loc2 = initial_values['sbeta_loc2']
   sbeta_loc3 = initial_values['sbeta_loc3']
   sbeta_logscale0 = initial_values['sbeta_logscale0']
   sbeta_logscale1 = initial_values['sbeta_logscale1']
   sbeta_logscale2 = initial_values['sbeta_logscale2']
   sbeta_shape = initial_values['sbeta_shape']
   
   Cluster_which = initial_values['Cluster_which']
   S_clusters = initial_values['S_clusters']
   betaCluster_which = initial_values['betaCluster_which']
   
   # Bookkeeping
   n_s = Y.shape[0]
   n_t = Y.shape[1]
   if n_t != size:
      import sys
      sys.exit("Make sure the number of cpus (N) = number of time replicates (n_t), i.e.\n     srun -N python scalemix_sampler.py")
   n_covariates = len(beta_loc0)
   Dist = initial_values['Dist']
   
   wh_to_plot_Xs = n_s*np.array([0.25,0.5,0.75])
   wh_to_plot_Xs = wh_to_plot_Xs.astype(int)

   delta = comm.bcast(delta,root=0)
   tau_sqd = comm.bcast(tau_sqd,root=0)
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
   
   inv_beta_loc0_cluster_proposal=list()
   for i in np.arange(n_beta_clusters):
       which_tmp = betaCluster_which[i]
       inv_beta_loc0_cluster_proposal.append((np.diag(np.repeat(1, sum(which_tmp))),np.repeat(1,sum(which_tmp))))       
   
   inv_beta_loc1_cluster_proposal=list()
   for i in np.arange(n_beta_clusters):
       which_tmp = betaCluster_which[i]
       inv_beta_loc1_cluster_proposal.append((np.diag(np.repeat(1, sum(which_tmp))),np.repeat(1,sum(which_tmp))))       

   inv_beta_loc2_cluster_proposal=list()
   for i in np.arange(n_beta_clusters):
       which_tmp = betaCluster_which[i]
       inv_beta_loc2_cluster_proposal.append((np.diag(np.repeat(1, sum(which_tmp))),np.repeat(1,sum(which_tmp))))       
   
   inv_beta_loc3_cluster_proposal=list()
   for i in np.arange(n_beta_clusters):
       which_tmp = betaCluster_which[i]
       inv_beta_loc3_cluster_proposal.append((np.diag(np.repeat(1, sum(which_tmp))),np.repeat(1,sum(which_tmp))))       
    
   inv_beta_logscale0_cluster_proposal=list()
   for i in np.arange(n_beta_clusters):
       which_tmp = betaCluster_which[i]
       inv_beta_logscale0_cluster_proposal.append((np.diag(np.repeat(1, sum(which_tmp))),np.repeat(1,sum(which_tmp))))       
     
   inv_beta_logscale1_cluster_proposal=list()
   for i in np.arange(n_beta_clusters):
       which_tmp = betaCluster_which[i]
       inv_beta_logscale1_cluster_proposal.append((np.diag(np.repeat(1, sum(which_tmp))),np.repeat(1,sum(which_tmp))))       
   
   inv_beta_logscale2_cluster_proposal=list()
   for i in np.arange(n_beta_clusters):
       which_tmp = betaCluster_which[i]
       inv_beta_logscale2_cluster_proposal.append((np.diag(np.repeat(1, sum(which_tmp))),np.repeat(1,sum(which_tmp))))       

   inv_beta_shape_cluster_proposal=list()
   for i in np.arange(n_beta_clusters):
       which_tmp = betaCluster_which[i]
       inv_beta_shape_cluster_proposal.append((np.diag(np.repeat(1, sum(which_tmp))),np.repeat(1,sum(which_tmp))))       
 
   Cor_Z_clusters_nonMissing=list()
   inv_Z_cluster_nonMissing=list()
   nonMissing_1t_cluster =list()
   S_clusters_nonMissing = list()
   sigma_Z_cluster_proposal_nonMissing=list()
   inv_Z_cluster_proposal_nonMissing = list()
   
   theta_c = comm.bcast(theta_c,root=0)
   for i in np.arange(n_clusters):
        tmp_nonMissing = ~np.isnan(Y[Cluster_which[i],rank])
        nonMissing_1t_cluster.append(tmp_nonMissing)
        
        tmp_S_clusters = S_clusters[i][:, tmp_nonMissing][tmp_nonMissing, :]
        S_clusters_nonMissing.append(tmp_S_clusters)
        Cor_tmp = utils.corr_fn(tmp_S_clusters, theta_c)
        cholesky_inv = (cholesky(Cor_tmp,lower=False),np.repeat(1,Cor_tmp.shape[0]))
        Cor_Z_clusters_nonMissing.append(Cor_tmp)
        inv_Z_cluster_nonMissing.append(cholesky_inv)
        sigma_Z_cluster_proposal_nonMissing.append(np.diag(np.repeat(1, Cor_tmp.shape[0])))
        inv_Z_cluster_proposal_nonMissing.append((np.diag(np.repeat(1, Cor_tmp.shape[0])),np.repeat(1,Cor_tmp.shape[0])))

   current_lik = utils.theta_c_update_mixture_me_likelihood_1t(Z_onetime, theta_c, 1, Cluster_which, 
                                    S_clusters_nonMissing, nonMissing_1t_cluster)
   current_lik_recv = comm.gather(current_lik, root=0)

     
   Z_within_thinning = np.empty((n_s,thinning)); Z_within_thinning[:] = np.nan

   # Marginal GEV parameters: per location x time
   beta_loc0 = comm.bcast(beta_loc0,root=0)
   beta_loc1 = comm.bcast(beta_loc1,root=0)
   beta_loc2 = comm.bcast(beta_loc2,root=0)
   beta_loc3 = comm.bcast(beta_loc3,root=0)
   beta_logscale0 = comm.bcast(beta_logscale0,root=0)
   beta_logscale1 = comm.bcast(beta_logscale1,root=0)
   beta_logscale2 = comm.bcast(beta_logscale2,root=0)
   beta_shape = comm.bcast(beta_shape,root=0)
   
   loc0 = Design_mat @beta_loc0
   loc1 = Design_mat @beta_loc1
   loc2 = Design_mat @beta_loc2
   loc3 = Design_mat @beta_loc3
   logscale0 = Design_mat @beta_logscale0
   logscale1 = Design_mat @beta_logscale1
   logscale2 = Design_mat @beta_logscale2
   shape = Design_mat @beta_shape
   
   Loc = np.tile(loc0, n_t) + np.tile(loc1, n_t)*np.repeat(WMGHGs,n_s) + np.tile(loc2, 
        n_t)*PDSI.flatten(order='F') + np.tile(loc3, n_t)*np.repeat(ELI_summer_average,n_s)
   Loc = Loc.reshape((n_s,n_t),order='F')

   Scale = np.exp(np.tile(logscale0, n_t) + np.tile(logscale1, n_t)*np.repeat(WMGHGs,
         n_s)  + np.tile(logscale2, n_t)*np.repeat(ELI_summer_average,n_s))
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
         
     beta_loc0_within_thinning = np.empty((n_covariates,thinning)); beta_loc0_within_thinning[:] = np.nan
     beta_loc1_within_thinning = np.empty((n_covariates,thinning)); beta_loc1_within_thinning[:] = np.nan
     beta_loc2_within_thinning = np.empty((n_covariates,thinning)); beta_loc2_within_thinning[:] = np.nan
     beta_loc3_within_thinning = np.empty((n_covariates,thinning)); beta_loc3_within_thinning[:] = np.nan

     beta_logscale0_within_thinning = np.empty((n_covariates,thinning)); beta_logscale0_within_thinning[:] = np.nan
     beta_logscale1_within_thinning = np.empty((n_covariates,thinning)); beta_logscale1_within_thinning[:] = np.nan
     beta_logscale2_within_thinning = np.empty((n_covariates,thinning)); beta_logscale2_within_thinning[:] = np.nan
     beta_shape_within_thinning = np.empty((n_covariates,thinning)); beta_shape_within_thinning[:] = np.nan
   
     delta_accept = 0
     tau_sqd_accept = 0
     theta_c_accept = 0
     
     beta_loc0_accept = np.repeat(0,n_beta_clusters)
     beta_loc1_accept = np.repeat(0,n_beta_clusters)
     beta_loc2_accept = np.repeat(0,n_beta_clusters)
     beta_loc3_accept = np.repeat(0,n_beta_clusters)
     beta_logscale0_accept = np.repeat(0,n_beta_clusters)
     beta_logscale1_accept = np.repeat(0,n_beta_clusters)
     beta_logscale2_accept = np.repeat(0,n_beta_clusters)
     beta_shape_accept = np.repeat(0,n_beta_clusters)
     
     # mu_loc0_accept = 0
     # mu_loc1_accept = 0
     # mu_scale_accept = 0
     # mu_shape_accept = 0
     
     sbeta_loc0_accept = 0
     sbeta_loc1_accept = 0
     sbeta_loc2_accept = 0
     sbeta_loc3_accept = 0
     sbeta_logscale0_accept = 0
     sbeta_logscale1_accept = 0
     sbeta_logscale2_accept = 0
     sbeta_shape_accept = 0
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
   for iter in np.arange(1,n_updates):
       index_within = (iter-1)%thinning
       # Update X
       # print(str(rank)+" "+str(iter)+" Gathered? "+str(np.where(~cen)))
       X_onetime = utils.X_update(Y_onetime, cen[:,rank], cen_above[:,rank], delta, tau_sqd, Loc[:,rank], Scale[:,rank], Shape[:,rank])
      
       # Update Z
       for cluster_num in np.arange(n_clusters):
             Z_1t_accept[cluster_num] += utils.update_Z_1t_one_cluster_interp(Z_onetime, Cluster_which, cluster_num, nonMissing_1t_cluster,
                                 Cor_Z_clusters_nonMissing, inv_Z_cluster_nonMissing, inv_Z_cluster_proposal_nonMissing, 
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

       # Update theta_c
       accept = 0
       theta_c_star = np.empty(2)
       if rank==0:
           tmp_upper = cholesky(prop_sigma['theta_c'],lower=False)
           tmp_params_star = sigma_m['theta_c']*random_generator.standard_normal(2)
           theta_c_star = theta_c + np.matmul(tmp_upper.T , tmp_params_star)
       theta_c_star = comm.bcast(theta_c_star, root=0)
       
       current_lik = utils.theta_c_update_mixture_me_likelihood_1t(Z_onetime, theta_c, 1, Cluster_which, 
                                    S_clusters_nonMissing, nonMissing_1t_cluster)
       if np.all(np.logical_and(theta_c_star>hyper_params_theta_c[0],
                                theta_c_star<hyper_params_theta_c[1])):
           star_lik = utils.theta_c_update_mixture_me_likelihood_1t(Z_onetime, theta_c_star, 1, Cluster_which, 
                                    S_clusters_nonMissing, nonMissing_1t_cluster)
       else:
           star_lik= -np.inf
       star_lik_recv = comm.gather(star_lik, root=0)  
       current_lik_recv = comm.gather(current_lik, root=0)
       
       if rank==0:
           log_num = np.sum(star_lik_recv)
           log_denom = np.sum(current_lik_recv)
           if log_num>log_denom:
               r=1
           else:
               r = np.exp(log_num - log_denom)
               if ~np.isfinite(r):
                    r = 0
           if random_generator.uniform(0,1,1)<r:
               theta_c[:] = theta_c_star
               # current_lik_recv[:] = star_lik_recv
               accept = 1
               theta_c_accept = theta_c_accept + 1
           theta_c_trace_within_thinning[:, index_within] = theta_c
       
       # Broadcast according to accept
       accept = comm.bcast(accept,root=0)
       if accept==1:
           # current_lik = star_lik
           theta_c[:] = theta_c_star
           Cor_Z_clusters_nonMissing=list()
           inv_Z_cluster_nonMissing=list()
           for i in np.arange(n_clusters):
               tmp_nonMissing = nonMissing_1t_cluster[i]
               Cor_tmp = utils.corr_fn(S_clusters_nonMissing[i], theta_c)
               cholesky_inv = (cholesky(Cor_tmp,lower=False),np.repeat(1,Cor_tmp.shape[0]))
               Cor_Z_clusters_nonMissing.append(Cor_tmp)
               inv_Z_cluster_nonMissing.append(cholesky_inv)

                   
                   
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
           
           
           # Update beta_loc0
           for cluster_num in np.arange(n_beta_clusters):
               beta_loc0_accept[cluster_num] += utils.update_beta_loc0_GEV_one_cluster_interp(beta_loc0, betaCluster_which, cluster_num, inv_beta_loc0_cluster_proposal,
                                                                            Design_mat, sbeta_loc0, Y, X_s, cen, cen_above, prob_below, prob_above, delta, tau_sqd,
                                                                            loc1, loc2, loc3, Scale, Shape, WMGHGs, PDSI, ELI_summer_average, xp, den_p, thresh_X, thresh_X_above, 
                                                                            sigma_m_beta_loc0_cluster[cluster_num], random_generator)
           beta_loc0_within_thinning[:, index_within] = beta_loc0
           loc0 = Design_mat @beta_loc0
           
           
           # Update beta_loc1
           for cluster_num in np.arange(n_beta_clusters):
               beta_loc1_accept[cluster_num] += utils.update_beta_loc1_GEV_one_cluster_interp(beta_loc1, betaCluster_which, cluster_num, inv_beta_loc1_cluster_proposal,
                                                                            Design_mat, sbeta_loc1, Y, X_s, cen, cen_above, prob_below, prob_above, delta, tau_sqd,
                                                                            loc0, loc2, loc3, Scale, Shape, WMGHGs, PDSI, ELI_summer_average, xp, den_p, thresh_X, thresh_X_above,
                                                                            sigma_m_beta_loc1_cluster[cluster_num], random_generator)
           beta_loc1_within_thinning[:, index_within] = beta_loc1
           loc1 = Design_mat @beta_loc1
           
           
           # Update beta_loc2
           for cluster_num in np.arange(n_beta_clusters):
               beta_loc2_accept[cluster_num] += utils.update_beta_loc2_GEV_one_cluster_interp(beta_loc2, betaCluster_which, cluster_num, inv_beta_loc2_cluster_proposal,
                                                                            Design_mat, sbeta_loc2, Y, X_s, cen, cen_above, prob_below, prob_above, delta, tau_sqd,
                                                                            loc0, loc1, loc3, Scale, Shape, WMGHGs, PDSI, ELI_summer_average, xp, den_p, thresh_X, thresh_X_above,
                                                                            sigma_m_beta_loc2_cluster[cluster_num], random_generator)
           beta_loc2_within_thinning[:, index_within] = beta_loc2
           loc2 = Design_mat @beta_loc2
           
           # Update beta_loc3
           for cluster_num in np.arange(n_beta_clusters):
               beta_loc3_accept[cluster_num] += utils.update_beta_loc3_GEV_one_cluster_interp(beta_loc3, betaCluster_which, cluster_num, inv_beta_loc3_cluster_proposal,
                                                                            Design_mat, sbeta_loc3, Y, X_s, cen, cen_above, prob_below, prob_above, delta, tau_sqd,
                                                                            loc0, loc1, loc2, Scale, Shape, WMGHGs, PDSI, ELI_summer_average, xp, den_p, thresh_X, thresh_X_above,
                                                                            sigma_m_beta_loc3_cluster[cluster_num], random_generator)
           beta_loc3_within_thinning[:, index_within] = beta_loc3
           loc3 = Design_mat @beta_loc3
           
           Loc = np.tile(loc0, n_t) + np.tile(loc1, n_t)*np.repeat(WMGHGs,n_s) + np.tile(loc2,
                    n_t)*PDSI.flatten(order='F') + np.tile(loc3, n_t)*np.repeat(ELI_summer_average,n_s)
           Loc = Loc.reshape((n_s,n_t),order='F')
           
           # Update beta_logscale0  
           for cluster_num in np.arange(n_beta_clusters):
               beta_logscale0_accept[cluster_num] += utils.update_beta_logscale0_GEV_one_cluster_interp(beta_logscale0, betaCluster_which, cluster_num, inv_beta_logscale0_cluster_proposal,
                                                                            Design_mat,sbeta_logscale0, Y, X_s, cen, cen_above, prob_below, prob_above, delta, tau_sqd,
                                                                            logscale1, logscale2, Loc, Shape, WMGHGs, ELI_summer_average, xp, den_p, thresh_X, thresh_X_above, 
                                                                            sigma_m_beta_logscale0_cluster[cluster_num], random_generator)
           beta_logscale0_within_thinning[:, index_within] = beta_logscale0
           logscale0 = Design_mat @beta_logscale0
           
           # Update beta_logscale1 
           for cluster_num in np.arange(n_beta_clusters):
               beta_logscale1_accept[cluster_num] += utils.update_beta_logscale1_GEV_one_cluster_interp(beta_logscale1, betaCluster_which, cluster_num, inv_beta_logscale1_cluster_proposal,
                                                                            Design_mat,sbeta_logscale1, Y, X_s, cen, cen_above, prob_below, prob_above, delta, tau_sqd,
                                                                            logscale0, logscale2, Loc, Shape, WMGHGs, ELI_summer_average, xp, den_p, thresh_X, thresh_X_above, 
                                                                            sigma_m_beta_logscale1_cluster[cluster_num], random_generator)
           beta_logscale1_within_thinning[:, index_within] = beta_logscale1
           logscale1 = Design_mat @beta_logscale1
           
           # Update beta_logscale2  
           for cluster_num in np.arange(n_beta_clusters):
               beta_logscale2_accept[cluster_num] += utils.update_beta_logscale2_GEV_one_cluster_interp(beta_logscale2, betaCluster_which, cluster_num, inv_beta_logscale2_cluster_proposal,
                                                                            Design_mat,sbeta_logscale2, Y, X_s, cen, cen_above, prob_below, prob_above, delta, tau_sqd,
                                                                            logscale0, logscale1, Loc, Shape, WMGHGs, ELI_summer_average, xp, den_p, thresh_X, thresh_X_above, 
                                                                            sigma_m_beta_logscale2_cluster[cluster_num], random_generator)
           beta_logscale2_within_thinning[:, index_within] = beta_logscale2
           logscale2 = Design_mat @beta_logscale2
           
           Scale = np.exp(np.tile(logscale0, n_t) + np.tile(logscale1, n_t)*np.repeat(WMGHGs,
                        n_s)  + np.tile(logscale2, n_t)*np.repeat(ELI_summer_average,n_s))
           Scale = Scale.reshape((n_s,n_t),order='F')
            
            
           # Update beta_shape
           for cluster_num in np.arange(n_beta_clusters):
               beta_shape_accept[cluster_num] += utils.update_beta_shape_GEV_one_cluster_interp(beta_shape, betaCluster_which, cluster_num, inv_beta_shape_cluster_proposal,
                                                                            Design_mat, sbeta_shape, Y, X_s, cen, cen_above, prob_below, prob_above, delta, tau_sqd,
                                                                            Loc, Scale, xp, den_p, thresh_X, thresh_X_above,
                                                                            sigma_m_beta_shape_cluster[cluster_num], random_generator)
           beta_shape_within_thinning[:, index_within] = beta_shape
           shape = Design_mat @beta_shape
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
           
           # Update sbeta_loc2
           Metr_sbeta_loc2 = sampler.static_metr(beta_loc2, sbeta_loc2, utils.dmvn_diag, priors.half_cauchy,
                   hyper_params_sbeta_loc2, 2,
                   random_generator,
                   np.nan, sigma_m['sbeta_loc2'], False,
                   0)
           sbeta_loc2_accept = sbeta_loc2_accept + Metr_sbeta_loc2['acc_prob']
           sbeta_loc2 = Metr_sbeta_loc2['trace'][0,1]
           
           # Update sbeta_loc3
           Metr_sbeta_loc3 = sampler.static_metr(beta_loc3, sbeta_loc3, utils.dmvn_diag, priors.half_cauchy,
                   hyper_params_sbeta_loc3, 2,
                   random_generator,
                   np.nan, sigma_m['sbeta_loc3'], False,
                   0)
           sbeta_loc3_accept = sbeta_loc3_accept + Metr_sbeta_loc3['acc_prob']
           sbeta_loc3 = Metr_sbeta_loc3['trace'][0,1]
           
           # Update sbeta_logscale0
           Metr_sbeta_logscale0 = sampler.static_metr(beta_logscale0, sbeta_logscale0, utils.dmvn_diag, priors.half_cauchy,
                   hyper_params_sbeta_logscale0, 2,
                   random_generator,
                   np.nan, sigma_m['sbeta_logscale0'], False,
                   0)
           sbeta_logscale0_accept = sbeta_logscale0_accept + Metr_sbeta_logscale0['acc_prob']
           sbeta_logscale0 = Metr_sbeta_logscale0['trace'][0,1]
           
           # Update sbeta_logscale1
           Metr_sbeta_logscale1 = sampler.static_metr(beta_logscale1, sbeta_logscale1, utils.dmvn_diag, priors.half_cauchy,
                   hyper_params_sbeta_logscale1, 2,
                   random_generator,
                   np.nan, sigma_m['sbeta_logscale1'], False,
                   0)
           sbeta_logscale1_accept = sbeta_logscale1_accept + Metr_sbeta_logscale1['acc_prob']
           sbeta_logscale1 = Metr_sbeta_logscale1['trace'][0,1]
           
           # Update sbeta_logscale2
           Metr_sbeta_logscale2 = sampler.static_metr(beta_logscale2, sbeta_logscale2, utils.dmvn_diag, priors.half_cauchy,
                   hyper_params_sbeta_logscale2, 2,
                   random_generator,
                   np.nan, sigma_m['sbeta_logscale2'], False,
                   0)
           sbeta_logscale2_accept = sbeta_logscale2_accept + Metr_sbeta_logscale2['acc_prob']
           sbeta_logscale2 = Metr_sbeta_logscale2['trace'][0,1]
           
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
       # theta_c = comm.bcast(theta_c,root=0)
       # V = comm.bcast(V,root=0)
       # d = comm.bcast(d,root=0)
       # Cor_Z_clusters = comm.bcast(Cor_Z_clusters,root=0)
       # inv_Z_cluster = comm.bcast(inv_Z_cluster,root=0)
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
               beta_loc2_trace[:,index] = beta_loc2
               beta_loc3_trace[:,index] = beta_loc3
               beta_logscale0_trace[:,index] = beta_logscale0
               beta_logscale1_trace[:,index] = beta_logscale1
               beta_logscale2_trace[:,index] = beta_logscale2
               beta_shape_trace[:,index] = beta_shape
               sigma_sbeta_loc0_trace[index] = sbeta_loc0
               sigma_sbeta_loc1_trace[index] = sbeta_loc1
               sigma_sbeta_loc2_trace[index] = sbeta_loc2
               sigma_sbeta_loc3_trace[index] = sbeta_loc3
               sigma_sbeta_logscale0_trace[index] = sbeta_logscale0
               sigma_sbeta_logscale1_trace[index] = sbeta_logscale1
               sigma_sbeta_logscale2_trace[index] = sbeta_logscale2
               sigma_sbeta_shape_trace[index] = sbeta_shape
               
               # mu_loc0_trace[index] = mu_loc0
               # mu_loc1_trace[index] = mu_loc1
               # mu_scale_trace[index] = mu_scale
               # mu_shape_trace[index] = mu_shape
               
               loc0_trace[index,:] = loc0
               loc1_trace[index,:] = loc1
               loc2_trace[index,:] = loc2
               loc3_trace[index,:] = loc3
               logscale0_trace[index,:] = logscale0
               logscale1_trace[index,:] = logscale1
               logscale2_trace[index,:] = logscale2
               shape_trace[index,:] = shape
          
            
           # Adapt via Shaby and Wells (2010)
           gamma2 = 1 / (index + offset)**(c_1)
           gamma1 = c_0*gamma2
           sigma_m_Z_cluster[:] = np.exp(np.log(sigma_m_Z_cluster) + gamma1*(Z_1t_accept/thinning - r_opt_md))
           Z_1t_accept[:] = np.repeat(0,n_clusters)
           inv_Z_cluster_proposal=list()
           for i in np.arange(n_clusters):
               which_tmp = np.where(Cluster_which[i])[0]
               sigma_Z_cluster_proposal_nonMissing[i] = sigma_Z_cluster_proposal_nonMissing[i] + gamma2*(np.cov(Z_within_thinning[which_tmp[nonMissing_1t_cluster[i]],:]) - sigma_Z_cluster_proposal_nonMissing[i])
               inv_Z_cluster_proposal_nonMissing.append((cholesky(sigma_Z_cluster_proposal_nonMissing[i],lower=False),np.repeat(1,np.sum(nonMissing_1t_cluster[i]))))           

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
                   
               # sigma_m['mu_loc0'] = np.exp(np.log(sigma_m['mu_loc0']) + gamma1*(mu_loc0_accept/thinning - r_opt_1d))
               # mu_loc0_accept = 0
               
               sigma_m_beta_loc1_cluster[:] = np.exp(np.log(sigma_m_beta_loc1_cluster) + gamma1*(beta_loc1_accept/thinning - r_opt_md))
               beta_loc1_accept[:] = np.repeat(0,n_beta_clusters)
               inv_beta_loc1_cluster_proposal=list()
               for i in np.arange(n_beta_clusters):
                   which_tmp = betaCluster_which[i]
                   sigma_beta_loc1_cluster_proposal[i] = sigma_beta_loc1_cluster_proposal[i] + gamma2*(np.cov(beta_loc1_within_thinning[which_tmp,:]) - sigma_beta_loc1_cluster_proposal[i])
                   inv_beta_loc1_cluster_proposal.append((cholesky(sigma_beta_loc1_cluster_proposal[i],lower=False),np.repeat(1,np.sum(which_tmp))))           
             
               # sigma_m['mu_loc1'] = np.exp(np.log(sigma_m['mu_loc1']) + gamma1*(mu_loc1_accept/thinning - r_opt_1d))
               # mu_loc1_accept = 0
               
               sigma_m_beta_loc2_cluster[:] = np.exp(np.log(sigma_m_beta_loc2_cluster) + gamma1*(beta_loc2_accept/thinning - r_opt_md))
               beta_loc2_accept[:] = np.repeat(0,n_beta_clusters)
               inv_beta_loc2_cluster_proposal=list()
               for i in np.arange(n_beta_clusters):
                   which_tmp = betaCluster_which[i]
                   sigma_beta_loc2_cluster_proposal[i] = sigma_beta_loc2_cluster_proposal[i] + gamma2*(np.cov(beta_loc2_within_thinning[which_tmp,:]) - sigma_beta_loc2_cluster_proposal[i])
                   inv_beta_loc2_cluster_proposal.append((cholesky(sigma_beta_loc2_cluster_proposal[i],lower=False),np.repeat(1,np.sum(which_tmp))))           
          
               sigma_m_beta_loc3_cluster[:] = np.exp(np.log(sigma_m_beta_loc3_cluster) + gamma1*(beta_loc3_accept/thinning - r_opt_md))
               beta_loc3_accept[:] = np.repeat(0,n_beta_clusters)
               inv_beta_loc3_cluster_proposal=list()
               for i in np.arange(n_beta_clusters):
                   which_tmp = betaCluster_which[i]
                   sigma_beta_loc3_cluster_proposal[i] = sigma_beta_loc3_cluster_proposal[i] + gamma2*(np.cov(beta_loc3_within_thinning[which_tmp,:]) - sigma_beta_loc3_cluster_proposal[i])
                   inv_beta_loc3_cluster_proposal.append((cholesky(sigma_beta_loc3_cluster_proposal[i],lower=False),np.repeat(1,np.sum(which_tmp))))           
          
               sigma_m_beta_logscale0_cluster[:] = np.exp(np.log(sigma_m_beta_logscale0_cluster) + gamma1*(beta_logscale0_accept/thinning - r_opt_md))
               beta_logscale0_accept[:] = np.repeat(0,n_beta_clusters)
               inv_beta_logscale0_cluster_proposal=list()
               for i in np.arange(n_beta_clusters):
                   which_tmp = betaCluster_which[i]
                   sigma_beta_logscale0_cluster_proposal[i] = sigma_beta_logscale0_cluster_proposal[i] + gamma2*(np.cov(beta_logscale0_within_thinning[which_tmp,:]) - sigma_beta_logscale0_cluster_proposal[i])
                   inv_beta_logscale0_cluster_proposal.append((cholesky(sigma_beta_logscale0_cluster_proposal[i],lower=False),np.repeat(1,np.sum(which_tmp))))
 
               # sigma_m['mu_scale'] = np.exp(np.log(sigma_m['mu_scale']) + gamma1*(mu_scale_accept/thinning - r_opt_1d))
               # mu_scale_accept = 0
               
               sigma_m_beta_logscale1_cluster[:] = np.exp(np.log(sigma_m_beta_logscale1_cluster) + gamma1*(beta_logscale1_accept/thinning - r_opt_md))
               beta_logscale1_accept[:] = np.repeat(0,n_beta_clusters)
               inv_beta_logscale1_cluster_proposal=list()
               for i in np.arange(n_beta_clusters):
                   which_tmp = betaCluster_which[i]
                   sigma_beta_logscale1_cluster_proposal[i] = sigma_beta_logscale1_cluster_proposal[i] + gamma2*(np.cov(beta_logscale1_within_thinning[which_tmp,:]) - sigma_beta_logscale1_cluster_proposal[i])
                   inv_beta_logscale1_cluster_proposal.append((cholesky(sigma_beta_logscale1_cluster_proposal[i],lower=False),np.repeat(1,np.sum(which_tmp))))
               
               sigma_m_beta_logscale2_cluster[:] = np.exp(np.log(sigma_m_beta_logscale2_cluster) + gamma1*(beta_logscale2_accept/thinning - r_opt_md))
               beta_logscale2_accept[:] = np.repeat(0,n_beta_clusters)
               inv_beta_logscale2_cluster_proposal=list()
               for i in np.arange(n_beta_clusters):
                   which_tmp = betaCluster_which[i]
                   sigma_beta_logscale2_cluster_proposal[i] = sigma_beta_logscale2_cluster_proposal[i] + gamma2*(np.cov(beta_logscale2_within_thinning[which_tmp,:]) - sigma_beta_logscale2_cluster_proposal[i])
                   inv_beta_logscale2_cluster_proposal.append((cholesky(sigma_beta_logscale2_cluster_proposal[i],lower=False),np.repeat(1,np.sum(which_tmp))))
 
               sigma_m_beta_shape_cluster[:] = np.exp(np.log(sigma_m_beta_shape_cluster) + gamma1*(beta_shape_accept/thinning - r_opt_md))
               beta_shape_accept[:] = np.repeat(0,n_beta_clusters)
               inv_beta_shape_cluster_proposal=list()
               for i in np.arange(n_beta_clusters):
                   which_tmp = betaCluster_which[i]
                   sigma_beta_shape_cluster_proposal[i] = sigma_beta_shape_cluster_proposal[i] + gamma2*(np.cov(beta_shape_within_thinning[which_tmp,:]) - sigma_beta_shape_cluster_proposal[i])
                   inv_beta_shape_cluster_proposal.append((cholesky(sigma_beta_shape_cluster_proposal[i],lower=False),np.repeat(1,np.sum(which_tmp))))    
                   
               # sigma_m['mu_shape'] = np.exp(np.log(sigma_m['mu_shape']) + gamma1*(mu_shape_accept/thinning - r_opt_1d))
               # mu_shape_accept = 0
               
               sigma_m['sbeta_loc0'] = np.exp(np.log(sigma_m['sbeta_loc0']) + gamma1*(sbeta_loc0_accept/thinning - r_opt_1d))
               sbeta_loc0_accept = 0
               
               sigma_m['sbeta_loc1'] = np.exp(np.log(sigma_m['sbeta_loc1']) + gamma1*(sbeta_loc1_accept/thinning - r_opt_1d))
               sbeta_loc1_accept = 0
               
               sigma_m['sbeta_loc2'] = np.exp(np.log(sigma_m['sbeta_loc2']) + gamma1*(sbeta_loc2_accept/thinning - r_opt_1d))
               sbeta_loc2_accept = 0
               
               sigma_m['sbeta_loc3'] = np.exp(np.log(sigma_m['sbeta_loc3']) + gamma1*(sbeta_loc3_accept/thinning - r_opt_1d))
               sbeta_loc3_accept = 0
               
               sigma_m['sbeta_logscale0'] = np.exp(np.log(sigma_m['sbeta_logscale0']) + gamma1*(sbeta_logscale0_accept/thinning - r_opt_1d))
               sbeta_logscale0_accept = 0
               
               sigma_m['sbeta_logscale1'] = np.exp(np.log(sigma_m['sbeta_logscale1']) + gamma1*(sbeta_logscale1_accept/thinning - r_opt_1d))
               sbeta_logscale1_accept = 0
               
               sigma_m['sbeta_logscale2'] = np.exp(np.log(sigma_m['sbeta_logscale2']) + gamma1*(sbeta_logscale2_accept/thinning - r_opt_1d))
               sbeta_logscale2_accept = 0
               
               sigma_m['sbeta_shape'] = np.exp(np.log(sigma_m['sbeta_shape']) + gamma1*(sbeta_shape_accept/thinning - r_opt_1d))
               sbeta_shape_accept = 0
          
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
                    'loc2':loc2,
                    'loc3':loc3,
                    'logscale0':logscale0,
                    'logscale1':logscale1,
                    'logscale2':logscale2,
                    'shape':shape,
                    'Design_mat':Design_mat,
                    'WMGHGs':WMGHGs,
                    'PDSI':PDSI,
                    'ELI_summer_average':ELI_summer_average,
                    'beta_loc0':beta_loc0,
                    'beta_loc1':beta_loc1,
                    'beta_loc2':beta_loc2,
                    'beta_loc3':beta_loc3,
                    'beta_logscale0':beta_logscale0,
                    'beta_logscale1':beta_logscale0,
                    'beta_logscale2':beta_logscale0,
                    'beta_shape':beta_shape,
                    # 'mu_loc0':mu_loc0,
                    # 'mu_loc1':mu_loc1,
                    # 'mu_scale':mu_scale,
                    # 'mu_shape':mu_shape,
                    'sbeta_loc0':sbeta_loc0,
                    'sbeta_loc1':sbeta_loc1,
                    'sbeta_loc2':sbeta_loc2,
                    'sbeta_loc3':sbeta_loc3,
                    'sbeta_logscale0':sbeta_logscale0,
                    'sbeta_logscale1':sbeta_logscale1,
                    'sbeta_logscale2':sbeta_logscale2,
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
                   dump(beta_loc2_trace, f)
                   dump(beta_loc3_trace, f)
                   dump(beta_logscale0_trace, f)
                   dump(beta_logscale1_trace, f)
                   dump(beta_logscale2_trace, f)
                   dump(beta_shape_trace, f)
                   dump(sigma_sbeta_loc0_trace,f)
                   dump(sigma_sbeta_loc1_trace,f)
                   dump(sigma_sbeta_loc2_trace,f)
                   dump(sigma_sbeta_loc3_trace,f)
                   dump(sigma_sbeta_logscale0_trace,f)
                   dump(sigma_sbeta_logscale1_trace,f)
                   dump(sigma_sbeta_logscale2_trace,f)
                   dump(sigma_sbeta_shape_trace,f)
                   
                   # dump(mu_loc0_trace,f)
                   # dump(mu_loc1_trace,f)
                   # dump(mu_scale_trace,f)
                   # dump(mu_shape_trace,f)
                   
                   dump(loc0_trace,f)
                   dump(loc1_trace,f)
                   dump(loc2_trace,f)
                   dump(loc3_trace,f)
                   dump(logscale0_trace,f)
                   dump(logscale1_trace,f)
                   dump(logscale2_trace,f)
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
                   dump(sigma_m_beta_loc2_cluster, f)
                   dump(sigma_m_beta_loc3_cluster, f)
                   dump(sigma_m_beta_logscale0_cluster, f)
                   dump(sigma_m_beta_logscale1_cluster, f)
                   dump(sigma_m_beta_logscale2_cluster, f)
                   dump(sigma_m_beta_shape_cluster, f)
                   
                   dump(sigma_beta_loc0_cluster_proposal, f)
                   dump(sigma_beta_loc1_cluster_proposal, f)
                   dump(sigma_beta_loc2_cluster_proposal, f)
                   dump(sigma_beta_loc3_cluster_proposal, f)
                   dump(sigma_beta_logscale0_cluster_proposal, f)
                   dump(sigma_beta_logscale1_cluster_proposal, f)
                   dump(sigma_beta_logscale2_cluster_proposal, f)
                   dump(sigma_beta_shape_cluster_proposal, f)
                   dump(sigma_Z_cluster_proposal_nonMissing, f)
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
               plt.plot(beta_logscale0_trace[0,:], color='gray', linestyle='solid')
               plt.ylabel(r'Scale $\sigma$: $\beta_0$')
               plt.subplot2grid(grid_size, (0,1)) # scale: beta_1
               plt.plot(beta_logscale0_trace[1,:], color='gray', linestyle='solid')
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
               plt.plot(sigma_sbeta_logscale0_trace[:], color='gray', linestyle='solid')
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
               plt.plot(logscale0_trace[:,wh_to_plot_Xs[0]], color='gray', linestyle='solid')
               plt.ylabel(r'logscale0'+'['+str(wh_to_plot_Xs[0])+']')
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
               plt.plot(logscale0_trace[:,wh_to_plot_Xs[1]], color='gray', linestyle='solid')
               plt.ylabel(r'logscale0'+'['+str(wh_to_plot_Xs[1])+']')
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
               plt.plot(logscale0_trace[:,wh_to_plot_Xs[2]], color='gray', linestyle='solid')
               plt.ylabel(r'logscale0'+'['+str(wh_to_plot_Xs[2])+']')
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
                   dump(sigma_Z_cluster_proposal_nonMissing, f)
                   f.close()
               
