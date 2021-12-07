import os
import sys
import json
import math
import torch
import numpy as np

import matplotlib
# matplotlib.use("pgf")
matplotlib.rcParams.update({
    # 'font.family': 'serif',
    'font.size':12,
})
from matplotlib import pyplot as plt

import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
seed_everything(42)

from networks.wgan import GoodNetwork
from networks.wgan_old import GoodGenerator
from networks.autoencoders import AE
from DiffNetFEM import DiffNet2DFEM
from datasets.rectangles import RectangleManufacturedNonZeroBC

from pytorch_lightning.callbacks.base import Callback

class OptimSwitchLBFGS(Callback):
    def __init__(self, epochs=10):
        self.switch_epoch = epochs

    def on_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch == self.switch_epoch:
            opts = [torch.optim.LBFGS(pl_module.network.parameters(), lr=1.0, max_iter=10)]
            trainer.optimizers = opts


class Poisson(DiffNet2DFEM):
    """docstring for Poisson"""
    def __init__(self, network, dataset, **kwargs):
        super(Poisson, self).__init__(network, dataset, **kwargs)
        self.om = dataset.om
        x = np.linspace(0,1,self.domain_size)
        y = np.linspace(0,1,self.domain_size)
        xx, yy = np.meshgrid(x,y)
        self.u_exact = torch.tensor(self.exact_solution(xx,yy))

    def exact_solution(self, x,y):
        return np.exp(-self.om*x)*np.sin(self.om*y)

    def loss(self, u, inputs_tensor, forcing_tensor):

        f = forcing_tensor # renaming variable
        
        # extract diffusivity and boundary conditions here
        nu = inputs_tensor[:,0:1,:,:]
        bc1 = inputs_tensor[:,1:2,:,:]
        bc2 = inputs_tensor[:,2:3,:,:]

        # print("size(u_exact) = ", self.u_exact.shape); exit()
        u_exact = self.u_exact.unsqueeze(0).unsqueeze(0).type_as(nu)        
        # apply boundary conditions
        u = torch.where(bc1>0.5,u_exact,u)
        u = torch.where(bc2>0.5,u*0.0,u)

        # print("size(bc1) = ", bc1.shape)
        # print("size(bc2) = ", bc2.shape)

        # gg = u.squeeze().detach().cpu().numpy()
        # hh = u_exact.squeeze().detach().cpu().numpy()
        # bc11 = bc1.squeeze().detach().cpu().numpy()
        # bc22 = bc2.squeeze().detach().cpu().numpy()
        # plt.figure(figsize=(20,10))
        # plt.subplot(2,2,1)
        # plt.imshow(bc11,cmap='jet')
        # plt.colorbar()
        # plt.subplot(2,2,2)
        # plt.imshow(bc22,cmap='jet')
        # plt.colorbar()
        # plt.subplot(2,2,2)
        # plt.plot(gg[:,0])
        # plt.subplot(2,2,1)
        # plt.plot(gg[0,:])
        # plt.subplot(2,2,2)
        # plt.plot(gg[-1,:])
        # plt.subplot(2,2,3)
        # plt.plot(gg[:,0])
        # plt.subplot(2,2,4)
        # plt.plot(gg[:,-1])
        # plt.savefig('test.png')
        # exit()


        nu_gp = self.gauss_pt_evaluation(nu)
        f_gp = self.gauss_pt_evaluation(f)
        u_gp = self.gauss_pt_evaluation(u)
        u_x_gp = self.gauss_pt_evaluation_der_x(u)
        u_y_gp = self.gauss_pt_evaluation_der_y(u)

        transformation_jacobian = self.gpw.unsqueeze(-1).unsqueeze(-1).unsqueeze(0).type_as(nu_gp)
        res_elmwise = (transformation_jacobian * (0.5 * nu_gp * (u_x_gp**2 + u_y_gp**2) - (u_gp * f_gp)))
        res_elmwise = torch.sum(res_elmwise, 1) 

        loss = torch.mean(res_elmwise)
        return loss

    def forward(self, batch):
        inputs_tensor, forcing_tensor = batch
        # print("size of inputs_tensor = ", inputs_tensor.shape)
        # print("size of forcing_tensor = ", forcing_tensor.shape)
        # exit()
        return self.network(inputs_tensor), inputs_tensor, forcing_tensor

    # def configure_optimizers(self):
    #     """
    #     Configure optimizer for network parameters
    #     """
    #     lr = self.learning_rate
    #     opts = [torch.optim.Adam(self.network, lr=lr)]
    #     return opts, []

    # def training_step(self, batch, batch_idx, optimizer_idx):
    # # def training_step(self, batch, batch_idx):
    #     if self.current_epoch % 2:
    #         opt = 1
    #     else:
    #         opt = 0
    #     if optimizer_idx==opt:
    #         u, inputs_tensor, forcing_tensor = self.forward(batch)
    #         loss_val = self.loss(u, inputs_tensor, forcing_tensor).mean()
    #         # self.log('PDE_loss', loss_val.item())
    #         # self.log('loss', loss_val.item())
    #     else:
    #         loss_val = torch.zeros((1), requires_grad=True)
    #     return {"loss": loss_val}

    def training_step(self, batch, batch_idx):
    # def training_step(self, batch, batch_idx):
        u, inputs_tensor, forcing_tensor = self.forward(batch)
        loss_val = self.loss(u, inputs_tensor, forcing_tensor).mean()
        # self.log('PDE_loss', loss_val.item())
        # self.log('loss', loss_val.item())
        return {"loss": loss_val}

    def training_step_end(self, training_step_outputs):
        loss = training_step_outputs["loss"]
        return training_step_outputs

    def configure_optimizers(self):
        lr = self.learning_rate

        opts = [torch.optim.Adam(self.network.parameters(), lr=lr)]

        # opts = [torch.optim.Adam(self.network, lr=lr), torch.optim.LBFGS(self.network, lr=1.0, max_iter=5)]
        schd = [torch.optim.lr_scheduler.ExponentialLR(opts[0], gamma=0.95)]
        return opts, schd

    def on_epoch_end(self):
        fig, axs = plt.subplots(1, 4, figsize=(2*4,1.2),
                            subplot_kw={'aspect': 'auto'}, sharex=True, sharey=True, squeeze=True)
        for ax in axs:
            ax.set_xticks([])
            ax.set_yticks([])
        self.network.eval()
        inputs, forcing = self.dataset[0]

        u, inputs_tensor, forcing_tensor = self.forward((inputs.unsqueeze(0).type_as(next(self.network.parameters())), forcing.unsqueeze(0).type_as(next(self.network.parameters()))))

        f = forcing_tensor.squeeze().detach().cpu() # renaming variable
        
        # extract diffusivity and boundary conditions here
        nu = inputs_tensor[:,0:1,:,:]
        bc1 = inputs_tensor[:,1:2,:,:]
        bc2 = inputs_tensor[:,2:3,:,:]

        ubc = self.u_exact.unsqueeze(0).unsqueeze(0).type_as(nu)
        # apply boundary conditions
        u = torch.where(bc1>0.5,ubc,u)
        u = torch.where(bc2>0.5,u*0.0,u)


        k = nu.squeeze().detach().cpu()
        u = u.squeeze().detach().cpu()
        self.u_curr = u

        u_exact = self.u_exact.squeeze().detach().cpu()
        diff = u - u_exact
        print(np.linalg.norm(diff.flatten())/self.domain_size)
        im0 = axs[0].imshow(f,cmap='jet')
        fig.colorbar(im0, ax=axs[0], ticks=[0.0, 4.0, 8.0, 12.0, 16.0, 20.0])
        im1 = axs[1].imshow(u,cmap='jet', vmin=0.0, vmax=1.0)
        fig.colorbar(im1, ax=axs[1])
        im2 = axs[2].imshow(u_exact,cmap='jet', vmin=0.0, vmax=1.0)
        fig.colorbar(im2, ax=axs[2])
        im3 = axs[3].imshow(diff,cmap='jet')
        fig.colorbar(im3, ax=axs[3])
        plt.savefig(os.path.join(self.logger[0].log_dir, 'contour_' + str(self.current_epoch) + '.png'))
        self.logger[0].experiment.add_figure('Contour Plots', fig, self.current_epoch)
        plt.close('all')

        # npz_filepath = os.path.join(self.logger[0].log_dir, 'outdata_' + str(self.current_epoch) + '.npz')
        # np.savez(npz_filepath, u=u, u_exact=u_exact, f=f)

    def calc_l2_err(self):
        cn = lambda j,n: [j,j+1,j+n,(j+1)+n]
        N = lambda x,y: (1./4.)*np.array([(1-x)*(1-y), (1+x)*(1-y), (1-x)*(1+y), (1+x)*(1+y)])
        transform = lambda a,b,x: ((a+b)/2. + (b-a)/2.*x)

        ngp = 4
        gpx = np.array([-0.577350269189626, 0.577350269189626, -0.577350269189626, 0.577350269189626])
        gpy = np.array([-0.577350269189626, -0.577350269189626, 0.577350269189626, 0.577350269189626])
        gpw = np.ones(4)

        nnodex = self.domain_size
        nnodey = self.domain_size
        nelmx = self.domain_size - 1
        nelmy = self.domain_size - 1
        hx = (1. / nelmx)
        hy = (1. / nelmy)
        J = (hx/2.)*(hy/2.)
        print("J = ", J)

        x = np.linspace(0,1,self.domain_size)
        y = np.linspace(0,1,self.domain_size)
        u_sol = self.u_curr.numpy()

        usolnorm = 0.
        uexnorm = 0.
        l2_err = 0.
        for j in range(nelmy):
            for i in range(nelmx):
                local_u = (u_sol[j:j+2,i:i+2].reshape(4,1)).squeeze()
                # print("local_u = ", local_u)
                for igp in range(ngp):
                    basis = N(gpx[igp], gpy[igp])
                    # print("basis = ", basis)
                    xp = transform(x[i],x[i+1],gpx[igp])
                    yp = transform(y[j],y[j+1],gpy[igp])

                    u1 = np.dot(local_u, basis)
                    u2 = self.exact_solution(xp,yp)

                    # print("(xp,yp,uex,usol) = ", xp,yp,u2,u1)
                    # print("u1 = ", u1)
                    # print("u2 = ", u2)
                    # print("gpw(igp) = ", gpw[igp])
                    # print("J = ", J)
                    l2_err += (u1 - u2)**2 * J
                    usolnorm += u1**2*J
                    uexnorm += u2**2*J

        l2_err = np.sqrt(l2_err)
        usolnorm = np.sqrt(usolnorm)
        uexnorm = np.sqrt(uexnorm)

        u_ex = self.u_exact.squeeze().detach().cpu().numpy()


        print("usol.shape =", u_sol.shape)
        print("uex.shape =", u_ex.shape)
        print("||u_sol||, ||uex|| = ", usolnorm, uexnorm)
        print("||e||_{{L2}} = ", l2_err)
        # by taking vector norm
        print("||e|| (vector-norm) = ", np.linalg.norm(u_ex - u_sol, 'fro')/nnodex)


def main():
    # u_tensor = np.random.randn(1,1,256,256)

    domain_size = 32

    u_tensor = np.ones((1,1,domain_size,domain_size))
    network = AE(in_channels=3, out_channels=1, dims=8, n_downsample=3)
    # network = GoodNetwork(in_channels=3, out_channels=1, in_dim=domain_size, out_dim=domain_size, lowest_dim=4, filters=32)
    # network = GoodGenerator()
    dataset = RectangleManufacturedNonZeroBC(domain_size=domain_size)
    basecase = Poisson(network, dataset, batch_size=1, domain_size=domain_size, learning_rate=3e-4)

    # ------------------------
    # 1 INIT TRAINER
    # ------------------------
    logger = pl.loggers.TensorBoardLogger('.', name="mms-network")
    csv_logger = pl.loggers.CSVLogger(logger.save_dir, name=logger.name, version=logger.version)

    early_stopping = pl.callbacks.early_stopping.EarlyStopping('loss',
        min_delta=1e-8, patience=10, verbose=False, mode='max', strict=True)
    checkpoint = pl.callbacks.model_checkpoint.ModelCheckpoint(monitor='loss',
        dirpath=logger.log_dir, filename='{epoch}-{step}',
        mode='min', save_last=True)

    lbfgs_switch = OptimSwitchLBFGS(epochs=1)

    trainer = Trainer(gpus=[0],callbacks=[early_stopping,lbfgs_switch],
        checkpoint_callback=checkpoint, precision=64, logger=[logger,csv_logger],
        max_epochs=150, deterministic=True, profiler="simple")

    # ------------------------
    # 4 Training
    # ------------------------

    trainer.fit(basecase)

    # ------------------------
    # 5 SAVE NETWORK
    # ------------------------
    torch.save(basecase.network, os.path.join(logger.log_dir, 'network.pt'))

    basecase.calc_l2_err()


if __name__ == '__main__':
    main()