import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
import torch
import numpy as np
from torchvision.ops import MLP

from pytorch_forecasting.models.temporal_fusion_transformer.sub_modules import GatedResidualNetwork

def get_mlp_module(input_size: int, output_size: int, mlp_hidden_size: int, mlp_hidden_layers: int,
                   activation_function=nn.ReLU, sigmoid_out: bool = False):
    if mlp_hidden_layers > 1:
        encoder_input = [nn.Linear(input_size, mlp_hidden_size), activation_function()]
        encoder_hidden = (mlp_hidden_layers - 1) * [nn.Linear(mlp_hidden_size, mlp_hidden_size), activation_function()]
        encoder_output = [nn.Linear(mlp_hidden_size, output_size), activation_function()]
        encoder_layers = encoder_input + encoder_hidden + encoder_output
    else:
        encoder_layers = [nn.Linear(input_size, output_size), activation_function()]

    if sigmoid_out:
        encoder_layers += [nn.Sigmoid()]

    mlp = nn.Sequential(*encoder_layers)
    return mlp


def get_conic_network(input_size: int, output_size: int, hidden_layers: int,
                   activation_function=nn.ReLU, normalization=None):
    
    if hidden_layers > 1:
        layer_sizes = np.linspace(input_size, stop=output_size, num=hidden_layers, dtype=int)
    else:
        layer_sizes = [output_size]
        
    conic_network = MLP(in_channels=input_size, hidden_channels=layer_sizes, activation_layer=activation_function)

    if normalization is not None:
        conic_network = nn.Sequential(
            conic_network,
            normalization()
        )
    
    return conic_network


def get_bottleneck_conic_network(input_size: int, output_size: int, bottleneck_size, hidden_layers: int,
                   activation_function=nn.ReLU, normalization=None):
    
    layer_sizes_pre = np.linspace(input_size, stop=bottleneck_size, num=2, dtype=int)
    layer_sizes_post = np.linspace(bottleneck_size, stop=output_size, num=3, dtype=int)
    
    layer_sizes = list(layer_sizes_pre) + list(layer_sizes_post)
    conic_network = MLP(in_channels=input_size, hidden_channels=layer_sizes, activation_layer=activation_function)

    if normalization is not None:
        conic_network = nn.Sequential(
            conic_network,
            normalization()
        )
    
    return conic_network


class SimpleVariableSelectionNetwork(nn.Module):
    
    def __init__(self, input_size, hidden_size, context_size=None) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.context_size = context_size
        
        self.weight_network = GatedResidualNetwork(self.input_size, self.hidden_size, self.hidden_size, context_size=self.context_size)
        
        self.encoder = nn.Sequential(
            GatedResidualNetwork(self.input_size, self.hidden_size, self.hidden_size),
            nn.ELU()
        )
        
    def forward(self, x, context=None):
        x_encoded = self.encoder(x)
        weights = self.weight_network(x, context)
        weights = F.softmax(weights)
        return x_encoded * weights
        
    

class NonSequenceMLP(pl.LightningModule):

    def __init__(self, input_size, output_size, mlp_hidden_size, mlp_hidden_layers, activation_function=nn.ReLU):
        super().__init__()
        self.mlp = get_mlp_module(input_size, output_size, mlp_hidden_size, mlp_hidden_layers, activation_function)

    def forward(self, x, future=1):
        x_static = x["static"]
        const = x_static[:, :1, :]
        out = self.mlp(const)

        out = out.squeeze(1)

        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        prediction = self(x)

        loss = F.mse_loss(prediction, y)
        self.log("train_loss", loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        prediction = self(x)

        loss = F.mse_loss(prediction, y)
        self.log("val_loss", loss, sync_dist=True)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        prediction = self(x)

        loss = F.mse_loss(prediction, y)
        self.log("test_loss", loss, sync_dist=True)
        return loss

    def configure_optimizers(self):
        # Zero-input: lr = 2e-3
        # Const_input lr = 3e-5
        # EncodedHiddenState lr= 2e-3
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

        return optimizer

    def __str__(self):
        # TODO Might be moved to BaseModel
        assert self.name
        out = self.name
        for key, param in self.hparams.items():
            concat_key = key.replace("_", "")
            out += f"_{concat_key}{param}"
        return out