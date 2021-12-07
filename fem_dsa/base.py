import torch
from torch.utils.data import DataLoader
from pytorch_lightning.core import LightningModule


class PDE(LightningModule):
    """
    PDE Base Class
    """

    def __init__(self, network, dataset, **kwargs):
        super().__init__()
        # self.save_hyperparameters(kwargs)
        self.kwargs = kwargs
        self.dataset = dataset
        self.network = network
        self.nsd = kwargs.get('nsd', 2)
        self.domain_size = kwargs.get('domain_size', 64)
        self.batch_size =  kwargs.get('batch_size', 64)
        self.n_workers = kwargs.get('n_workers', 1)
        self.learning_rate = kwargs.get('learning_rate', 3e-4)

    def loss(self, u, inputs_tensor, forcing_tensor):
        raise NotImplementedError

    def forward(self, batch):
        inputs_tensor, forcing_tensor = batch
        u = self.network(inputs_tensor)
        return u, inputs_tensor, forcing_tensor

    def training_step(self, batch, batch_idx):
        u, inputs_tensor, forcing_tensor = self.forward(batch)
        loss_val = self.loss(u, inputs_tensor, forcing_tensor).mean()
        self.log('PDE_loss', loss_val.item())
        self.log('loss', loss_val.item())
        return loss_val

    def train_dataloader(self):
        """
        The data returned by DataLoader is on the same device that PL is using for network parameter
        """
        return DataLoader(self.dataset, batch_size=self.batch_size)

    def configure_optimizers(self):
        """
        Configure optimizer for network parameters
        """
        lr = self.learning_rate
        opts = [torch.optim.Adam(self.network.parameters(), lr=lr)]
        return opts, []