import numpy as np
import matplotlib
# matplotlib.use("pgf")
matplotlib.rcParams.update({
    # 'font.family': 'serif',
    'font.size':20,
})
from matplotlib import pyplot as plt
import os
import sys
import json
import seaborn as sns
from vti_writer import *


##############################################################################################33

def plot_contours(args, nu_batch, u_gen_batch, u_num_batch):
    for i in range(args.batch_size):
        u = u_gen_batch[i,0,:,:]
        u_exact = u_num_batch[i,0,:,:] 
        print("u_exact.shape = ", u_exact.shape, "u.shape ", u.shape)
        diff = u - u_exact
        k = (nu_batch[i]).squeeze(0)
        
        np.set_printoptions(precision=3)
        print(np.around(np.linalg.norm(diff.flatten())/args.output_size, 4))
    
        plt.figure(figsize=(24,4))
        
        plt.subplot(1,4,1)
        plt.imshow(k,cmap='jet')
        plt.colorbar()
        
        plt.subplot(1,4,2)
        plt.imshow(u,cmap='jet')
        plt.colorbar()

        # plt.figure()
        plt.subplot(1,4,3)
        plt.imshow(u_exact,cmap='jet')
        plt.colorbar()

    #     # plt.figure()
        plt.subplot(1,4,4)
        plt.imshow(diff,cmap='jet')
        plt.colorbar()

        figure_save_dir = os.path.join(args.model_dir, 'res_'+str(args.output_size), 'version_'+str(args.version))
        # if not os.path.exists(figure_save_dir):
        #     os.makedirs(figure_save_dir)
        plt.savefig(os.path.join(figure_save_dir, 'query_contour_' + str(i) + '.png'))
    plt.close('all')

def plot_line_cuts(args, nu_batch, u_gen_batch, u_num_batch):
    if args.fem_basis_deg == 1:
        n_elms = args.output_size + 1
        n_nodes = n_elms + 1
    elif args.fem_basis_deg == 2:
        n_elms = (args.output_size + 2) / 2
        n_nodes = 2 * n_elms + 1
    elif args.fem_basis_deg == 3:
        n_elms = (args.output_size + 2) / 3
        n_nodes = u_gen_batch.shape[3] #3 * n_elms + 1
    x = np.linspace(0,1,n_nodes)
    y = np.linspace(0,1,n_nodes)
    for i in range(args.batch_size):
        u = u_gen_batch[i,0,:,:]
        u_exact = u_num_batch[i,0,:,:]            
        diff = u - u_exact
        k = (nu_batch[i]).squeeze(0)        

        plt.figure(figsize=(12,4))
        n_subplots = 3
        u_maxlim = 1.1

        figure_save_dir = os.path.join(args.model_dir, 'res_'+str(args.output_size), 'version_'+str(args.version))
    #     print((u[:,65]))
        
        # x-constant slices
        plt.subplot(1,n_subplots,1)
        idx = int(0.2 * n_nodes)
        plt.plot(y,u[:,idx],label='u_gen')
        plt.plot(y,u_exact[:,idx],label='u_num')
        plt.ylim(0,u_maxlim)
        plt.xlabel('y')
        plt.legend(loc="lower right")
        
        plt.subplot(1,n_subplots,2)
        idx = int(0.5 * n_nodes)
        plt.plot(y,u[:,idx],label='u_gen')
        plt.plot(y,u_exact[:,idx],label='u_num')
        plt.ylim(0,u_maxlim)
        plt.xlabel('y')
        plt.legend(loc="lower right")
        
        plt.subplot(1,n_subplots,3)
        idx = int(0.8 * n_nodes)
        plt.plot(y,u[:,idx],label='u_gen')
        plt.plot(y,u_exact[:,idx],label='u_num')      
        plt.ylim(0,u_maxlim)
        plt.xlabel('y')
        plt.legend(loc="lower right")

        figure_save_dir = os.path.join(args.model_dir, 'res_'+str(args.output_size), 'version_'+str(args.version))
        
        plt.figure(figsize=(12,4))
        n_subplots = 3
        u_maxlim = 1.1
        # y-constant slices
        plt.subplot(1,n_subplots,1)
        idx = int(0.2 * n_nodes)
        plt.plot(x,u[idx,:],label='u_gen')
        plt.plot(x,u_exact[idx,:],label='u_num')
        # plt.ylim(0,u_maxlim)
        plt.xlabel('x')
        plt.legend(loc="lower right")
        
        plt.subplot(1,n_subplots,2)
        idx = int(0.5 * n_nodes)
        plt.plot(x,u[idx,:],label='u_gen')
        plt.plot(x,u_exact[idx,:],label='u_num')
        plt.ylim(0,u_maxlim)
        plt.xlabel('x')
        plt.legend(loc="lower right")

        plt.subplot(1,n_subplots,3)
        idx = int(0.8 * n_nodes)
        plt.plot(x,u[idx,:],label='u_gen')
        plt.plot(x,u_exact[idx,:],label='u_num')
        plt.ylim(0,u_maxlim)
        plt.xlabel('x')
        plt.legend(loc="lower right")
        plt.tight_layout()
        # if not os.path.exists(figure_save_dir):
            # os.makedirs(figure_save_dir)
        figure_save_dir = os.path.join(args.model_dir, 'res_'+str(args.output_size), 'version_'+str(args.version))
        plt.savefig(os.path.join(figure_save_dir, 'line_cut_' + str(i) + '.pdf'))
    plt.close('all')

class plotter_3d(object):
    """docstring for plotter_3d"""
    def __init__(self, nu, u_gen, u_num):
        self.nu = nu
        self.u_gen = u_gen
        self.u_num = u_num

    def plot_slices_z(self):
        return   

