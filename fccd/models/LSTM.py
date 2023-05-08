import torch.nn as nn
import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from pytorch_forecasting.models.temporal_fusion_transformer.sub_modules import GatedResidualNetwork, InterpretableMultiHeadAttention, GateAddNorm
from .MLP import SimpleVariableSelectionNetwork, get_conic_network

from torch.optim.lr_scheduler import StepLR

EXCLUDE_FROM_NAME = ["lr", "learning_rate", "name"]


class BaseModel(pl.LightningModule):
    def __init__(self, learning_rate=1e3, name=None, scheduler=None):
        super().__init__()
        self.learning_rate = learning_rate
        self.lr = learning_rate
        self.name = name
        self.scheduler = scheduler

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=(self.lr or self.learning_rate))
        
        if self.scheduler == True:
            self.scheduler = StepLR(optimizer, step_size=50, gamma=0.7)

        if self.scheduler is not None:
            scheduler_data = [{"scheduler": self.scheduler, "interval": "epoch"}]
            optimizer_config = [optimizer], scheduler_data
        else:
            optimizer_config = optimizer
            
        return optimizer_config
    
    def __str__(self):
        assert self.name
        out = self.name
        for key, param in self.hparams.items():
            if key in EXCLUDE_FROM_NAME:
                continue
            concat_key = key.replace("_", "")
            out += f"_{concat_key}{param}"
        return out


class BaseLSTM(BaseModel):

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y["y"]

        prediction_length = y.size(1)
        prediction = self(x, future=prediction_length)

        loss = F.mse_loss(prediction, y)
        self.log("train_loss", loss, sync_dist=True, batch_size=64)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y["y"]

        prediction_length = y.size(1)
        prediction = self(x, future=prediction_length)

        loss = F.mse_loss(prediction, y)
        self.log("val_loss", loss, sync_dist=True, batch_size=64)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y = y["y"]

        prediction_length = y.size(1)
        prediction = self(x, future=prediction_length)

        loss = F.mse_loss(prediction, y)
        self.log("test_loss", loss, sync_dist=True, batch_size=64)
        return loss


class PostAggregationLSTM(BaseLSTM):
    def __init__(self, input_size, output_size, n_hidden, mlp_hidden_size, mlp_hidden_layers, n_layers=2, learning_rate=2e-3,
                 name="PostAgg"):
        super().__init__(learning_rate, name)
        self.mlp_hidden_layers = mlp_hidden_layers
        self.mlp_hidden_size = mlp_hidden_size
        self.lstm_hidden_size = n_hidden
        self.lstm_layers = n_layers
        self.save_hyperparameters()

        self.output_size = output_size
        self.input_size = input_size
        

        self.lstm: nn.LSTM = nn.LSTM(output_size, self.lstm_hidden_size, n_layers, batch_first=True)
        self.lstm_decoder = nn.Linear(self.lstm_hidden_size, out_features=self.output_size)

        self.output_mlp = get_conic_network(self.output_size + self.input_size, self.output_size, mlp_hidden_layers)

    def forward(self, x, future=1):
        x_static = x["static"]
        n_samples = x_static.size(0)
        const = x_static[:, 0, :] # use any timestep

        h_t = torch.zeros(self.lstm_layers, n_samples, self.lstm_hidden_size, dtype=torch.float32, device=self.device)
        c_t = torch.zeros(self.lstm_layers, n_samples, self.lstm_hidden_size, dtype=torch.float32, device=self.device)

        lstm_input = torch.zeros(n_samples, 1, self.output_size, device=self.device)
        output_sequence = []

        for t in range(future):
            encoded_prediction, (h_t, c_t) = self.lstm(lstm_input, (h_t, c_t))
            lstm_prediction = self.lstm_decoder(encoded_prediction)

            mlp_input = torch.cat([lstm_prediction, const.unsqueeze(1)], dim=2)
            prediction = self.output_mlp(mlp_input)

            lstm_input = prediction
            output_sequence += [prediction]

        output_sequence = torch.cat(output_sequence, dim=1)
        output_sequence = output_sequence.squeeze(2)

        return output_sequence


class RepeatedInputLSTM(BaseLSTM):
    def __init__(self, input_size, n_hidden, output_size=1, n_layers=2, learning_rate=3e-5, name="RepIn"):
        super().__init__(learning_rate, name)
        self.n_hidden = n_hidden
        self.n_layers = n_layers

        self.output_size = output_size
        self.input_size = input_size
        self.save_hyperparameters()

        self.lstm_input_size = self.input_size + self.output_size
        self.lstm: nn.LSTM = nn.LSTM(self.lstm_input_size, self.n_hidden, n_layers, batch_first=True)
        self.lstm_decoder = nn.Linear(self.n_hidden, out_features=self.output_size)

    def forward(self, x, future=1):
        x_static = x["static"]
        n_samples = x_static.size(0)
        const = x_static[:, :1, :]

        h_t = torch.zeros(self.n_layers, n_samples, self.n_hidden, dtype=torch.float32, device=self.device)
        c_t = torch.zeros(self.n_layers, n_samples, self.n_hidden, dtype=torch.float32, device=self.device)

        initial_input = torch.zeros(n_samples, 1, self.output_size, device=self.device)
        initial_input = torch.cat([const, initial_input], dim=2)
        lstm_input = initial_input
        output_sequence = []

        for t in range(future):
            encoded_prediction, (h_t, c_t) = self.lstm(lstm_input, (h_t, c_t))
            prediction = self.lstm_decoder(encoded_prediction)

            lstm_input = torch.cat([const, prediction], dim=2)
            output_sequence += [prediction]

        output_sequence = torch.cat(output_sequence, dim=1)
        output_sequence = output_sequence.squeeze(2)

        return output_sequence


class RawHiddenStateLSTM(BaseLSTM):
    def __init__(self, input_size, output_size=1, lstm_layers=2, learning_rate=2e-3, name="RawIn"):
        super().__init__(learning_rate, name)
        self.lstm_layers = lstm_layers

        self.output_size = output_size
        self.input_size = input_size

        self.lstm_hidden_size = self.input_size

        self.save_hyperparameters()
        self.lstm: nn.LSTM = nn.LSTM(self.output_size, self.lstm_hidden_size, lstm_layers, batch_first=True)
        self.lstm_decoder = nn.Linear(self.lstm_hidden_size, out_features=self.output_size)

    def forward(self, x, future=1):
        x_static = x["static"]
        n_samples = x_static.size(0)
        static_parameters = x_static[:, :1, :]

        h_t = static_parameters
        h_t = torch.swapdims(h_t, 0, 1)
        h_t = h_t.repeat(self.lstm_layers, 1, 1)
        c_t = torch.ones_like(h_t, device=self.device)

        lstm_input = torch.zeros(n_samples, 1, self.output_size, device=self.device)
        output_sequence = []

        for t in range(future):
            encoded_prediction, (h_t, c_t) = self.lstm(lstm_input, (h_t, c_t))
            prediction = self.lstm_decoder(encoded_prediction)
            lstm_input = prediction
            output_sequence += [prediction]

        output_sequence = torch.cat(output_sequence, dim=1)
        output_sequence = output_sequence.squeeze(2)

        return output_sequence


class EncodedHiddenStateLSTM(BaseLSTM):

    def __init__(self, input_size, lstm_hidden_size, mlp_hidden_layers, output_size=1, lstm_layers=2,
                 learning_rate=2e-3, name="EncHidLSTM"):
        super().__init__(learning_rate, name)
        self.output_size = output_size
        self.mlp_hidden_layers = mlp_hidden_layers
        self.lstm_hidden_size = lstm_hidden_size
        self.save_hyperparameters()
        self.param_encoder_c = get_conic_network(input_size, lstm_hidden_size, mlp_hidden_layers)
        self.param_encoder_h = get_conic_network(input_size, lstm_hidden_size, mlp_hidden_layers)
        self.lstm_layers = lstm_layers

        self.rnn = nn.LSTM(self.output_size, lstm_hidden_size, lstm_layers, batch_first=True)

        self.lstm_decoder = nn.Linear(self.lstm_hidden_size, out_features=self.output_size)

    def forward(self, x, future=1):
        x_static = x["static"]
        n_samples = x_static.size(0)
        static_parameters = x_static[:, 0, :]

        # Compute plastic part
        h_t = encode_state(static_parameters, self.param_encoder_h, self.lstm_layers)
        c_t = encode_state(static_parameters, self.param_encoder_c, self.lstm_layers)

        lstm_input = torch.zeros(n_samples, 1, self.output_size, device=self.device)
        output_sequence = []

        for t in range(future):
            encoded_prediction, (h_t, c_t) = self.rnn(lstm_input, (h_t, c_t))
            prediction = self.lstm_decoder(encoded_prediction)
            output_sequence += [prediction]
            lstm_input = prediction

        output_sequence = torch.cat(output_sequence, dim=1)
        output_sequence = output_sequence.squeeze(2)

        return output_sequence
    
    
class PostAggEncodedHiddenStateLSTM(BaseLSTM):

    def __init__(self, input_size, lstm_hidden_size, mlp_hidden_layers, output_size=1, lstm_layers=2,
                 learning_rate=2e-3, name="EncHidLSTM"):
        super().__init__(learning_rate, name)
        self.output_size = output_size
        self.mlp_hidden_layers = mlp_hidden_layers
        self.lstm_hidden_size = lstm_hidden_size
        self.save_hyperparameters()
        self.param_encoder_c = get_conic_network(input_size, lstm_hidden_size, mlp_hidden_layers)
        self.param_encoder_h = get_conic_network(input_size, lstm_hidden_size, mlp_hidden_layers)
        self.param_encoder_post = get_conic_network(input_size, lstm_hidden_size, mlp_hidden_layers)
        self.lstm_layers = lstm_layers

        self.rnn = nn.LSTM(self.output_size, lstm_hidden_size, lstm_layers, batch_first=True)

        self.lstm_decoder = nn.Linear(2 * self.lstm_hidden_size, out_features=self.output_size)

    def forward(self, x, future=1):
        x_static = x["static"]
        n_samples = x_static.size(0)
        static_parameters = x_static[:, 0, :]

        # Compute plastic part
        h_t = encode_state(static_parameters, self.param_encoder_h, self.lstm_layers)
        c_t = encode_state(static_parameters, self.param_encoder_c, self.lstm_layers)
        post_agg = encode_state(static_parameters, self.param_encoder_post, self.lstm_layers)

        lstm_input = torch.zeros(n_samples, 1, self.output_size, device=self.device)
        output_sequence = []

        for t in range(future):
            encoded_prediction, (h_t, c_t) = self.rnn(lstm_input, (h_t, c_t))
            aggregated_encoded_prediction = torch.cat([encoded_prediction, post_agg], dim=2)
            prediction = self.lstm_decoder(aggregated_encoded_prediction)
            output_sequence += [prediction]
            lstm_input = prediction

        output_sequence = torch.cat(output_sequence, dim=1)
        output_sequence = output_sequence.squeeze(2)

        return output_sequence


class EncodedHiddenStateGRU(BaseLSTM):

    def __init__(self, input_size, lstm_hidden_size, mlp_hidden_layers, output_size=1, lstm_layers=2,
                 learning_rate=2e-3, name="EncHidGRU"):
        super().__init__(learning_rate, name)
        self.output_size = output_size
        self.mlp_hidden_layers = mlp_hidden_layers
        self.lstm_hidden_size = lstm_hidden_size
        self.save_hyperparameters()
        self.param_encoder_c = get_conic_network(input_size, lstm_hidden_size, mlp_hidden_layers)
        self.param_encoder_h = get_conic_network(input_size, lstm_hidden_size, mlp_hidden_layers)
        self.lstm_layers = lstm_layers

        self.rnn = nn.GRU(self.output_size, lstm_hidden_size, lstm_layers, batch_first=True)

        self.lstm_decoder = nn.Linear(self.lstm_hidden_size, out_features=self.output_size)

    def forward(self, x, future=1):
        x_static = x["static"]
        n_samples = x_static.size(0)
        static_parameters = x_static[:, 0, :]

        # Compute plastic part
        h_t = encode_state(static_parameters, self.param_encoder_h, self.lstm_layers)

        lstm_input = torch.zeros(n_samples, 1, self.output_size, device=self.device)
        output_sequence = []

        for t in range(future):
            encoded_prediction, h_t = self.rnn(lstm_input, h_t)
            prediction = self.lstm_decoder(encoded_prediction)
            output_sequence += [prediction]
            lstm_input = prediction

        output_sequence = torch.cat(output_sequence, dim=1)
        output_sequence = output_sequence.squeeze(2)

        return output_sequence


class ChunkingEncodedHiddenStateLSTM(BaseLSTM):

    def __init__(self, input_size, lstm_hidden_size, mlp_hidden_layers, lstm_layers=2, output_size=1,
                 learning_rate=2e-3, name="ChuEncHid", padding_value=-10, target_stress_idx=None):
        super().__init__(learning_rate, name)
        
        assert not (output_size > 1 and target_stress_idx is None)

        self.padding_value = padding_value
        self.estimate_psd_idx = None

        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_layers = lstm_layers

        self.output_size = output_size
        self.input_size = input_size
        self.save_hyperparameters()
        self.target_stress_idx = target_stress_idx

        self.param_encoder_c = get_conic_network(input_size, lstm_hidden_size, mlp_hidden_layers)
        self.param_encoder_h = get_conic_network(input_size, lstm_hidden_size, mlp_hidden_layers)

        self.rnn = nn.LSTM(self.output_size, self.lstm_hidden_size, lstm_layers, batch_first=True)
        self.lstm_decoder = get_conic_network(self.lstm_hidden_size, self.output_size, mlp_hidden_layers)

    def with_estimator(self, estimator):
        self.estimate_psd_idx = estimator
        return self

    def forward(self, x, future=1):
        x_static = x["static"]
        x_psd = x["psd"]
        static_parameters = x_static[:, 0, :]
        psd_data = x_psd[:, 0, :]

        # Compute elastic part
        psd_idx_estimates = self.estimate_psd_idx(static_parameters, psd_data, self.device)
        psd_idx_estimates = to_integer_tensor(psd_idx_estimates)
        before_psd, _ = interpolate_elastic_deformation(psd_data, psd_idx_estimates, max_idx=future,
                                                             padding_value=self.padding_value, device=self.device)

        # Compute plastic part
        h_t = encode_state(static_parameters, self.param_encoder_h, self.lstm_layers)
        c_t = encode_state(static_parameters, self.param_encoder_c, self.lstm_layers)
        
        lstm_input = psd_data.unsqueeze(1)
        if self.output_size > 1:
            initial_strain = torch.zeros_like(lstm_input)
            initial_disloc = torch.zeros_like(lstm_input)
            lstm_input = torch.cat([lstm_input, initial_strain, initial_disloc], dim=2)
        lstm_output_sequence = []

        for t_n in range(future):
            encoded_prediction, (h_t, c_t) = self.rnn(lstm_input, (h_t, c_t))
            prediction = self.lstm_decoder(encoded_prediction)
            lstm_output_sequence += [prediction]
            lstm_input = prediction
        after_psd: torch.tensor = torch.cat(lstm_output_sequence, dim=1)
        
        if self.output_size > 1:
            stress_after_psd  = after_psd[:, :, self.target_stress_idx]
            stress_after_psd  = stress_after_psd.unsqueeze(2)
            padded_predictions = torch.hstack([before_psd, stress_after_psd])
            trimmed_predictions = [pred[future-psd_idx:-psd_idx, :] for pred, psd_idx in zip(padded_predictions, psd_idx_estimates)]
            stress_prediction = torch.stack(trimmed_predictions)
            after_psd[:, :, self.target_stress_idx] = stress_prediction.squeeze(2)
            prediction = after_psd
        
        else:
            
            padded_predictions = torch.hstack([before_psd, after_psd])
            trimmed_predictions = [pred[future-psd_idx:-psd_idx, :] for pred, psd_idx in zip(padded_predictions, psd_idx_estimates)]
            prediction = torch.stack(trimmed_predictions)
        
        prediction = prediction.squeeze(2)

        return prediction
    
    
class RepeatingChunkingEncodedHiddenStateLSTM(BaseLSTM):

    def __init__(self, input_size, lstm_hidden_size, mlp_hidden_layers, lstm_layers=2, output_size=1,
                 learning_rate=2e-3, name="RepChuEncHid", padding_value=-10):
        super().__init__(learning_rate, name)

        self.padding_value = padding_value
        self.estimate_psd_idx = None

        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_layers = lstm_layers

        self.output_size = output_size
        self.input_size = input_size
        self.save_hyperparameters()

        self.param_encoder_c = get_conic_network(input_size, lstm_hidden_size, mlp_hidden_layers)
        self.param_encoder_h = get_conic_network(input_size, lstm_hidden_size, mlp_hidden_layers)
        self.param_encoder_in = get_conic_network(input_size, 4, mlp_hidden_layers)

        rnn_input_size = 4 + 1
        self.rnn = nn.LSTM(rnn_input_size, self.lstm_hidden_size, lstm_layers, batch_first=True)
        self.lstm_decoder = get_conic_network(self.lstm_hidden_size, self.output_size, mlp_hidden_layers)

    def with_estimator(self, estimator):
        self.estimate_psd_idx = estimator
        return self

    def forward(self, x, future=1):
        x_static = x["static"]
        x_psd = x["psd"]
        static_parameters = x_static[:, 0, :]
        psd_data = x_psd[:, 0, :]

        # Compute elastic part
        psd_idx_estimates = self.estimate_psd_idx(static_parameters, psd_data, self.device)
        psd_idx_estimates = to_integer_tensor(psd_idx_estimates)
        before_psd, _ = interpolate_elastic_deformation(psd_data, psd_idx_estimates, max_idx=future,
                                                             padding_value=self.padding_value, device=self.device)

        # Compute plastic part
        h_t = encode_state(static_parameters, self.param_encoder_h, self.lstm_layers)
        c_t = encode_state(static_parameters, self.param_encoder_c, self.lstm_layers)
        
        encoded_input = self.param_encoder_in(static_parameters)        
        lstm_input = torch.hstack([psd_data, encoded_input])
        lstm_input = lstm_input.unsqueeze(1)

        lstm_output_sequence = []

        for t_n in range(future):
            encoded_prediction, (h_t, c_t) = self.rnn(lstm_input, (h_t, c_t))
            prediction = self.lstm_decoder(encoded_prediction)
            lstm_output_sequence += [prediction]
            lstm_input = torch.cat([prediction, encoded_input.unsqueeze(1)], dim=2)
        after_psd: torch.tensor = torch.cat(lstm_output_sequence, dim=1)

        padded_predictions = torch.hstack([before_psd, after_psd])
        trimmed_predictions = [pred[future-psd_idx:-psd_idx, :] for pred, psd_idx in zip(padded_predictions, psd_idx_estimates)]
        prediction = torch.stack(trimmed_predictions)
        prediction = prediction.squeeze(-1)

        return prediction
    

class SeqToSeqLSTM(BaseLSTM):
    
    def __init__(self, input_size, lstm_hidden_size, mlp_hidden_layers, output_size=1, lstm_layers=2,
                 learning_rate=2e-3, name="SeqToSeqHybridLSTM", lr_scheduler=None):
        super().__init__(learning_rate, name, lr_scheduler)
        self.output_size = output_size
        self.mlp_hidden_layers = mlp_hidden_layers
        self.lstm_hidden_size = lstm_hidden_size
        self.save_hyperparameters()
        
        self.pre_estimator = None
        self.param_encoder_c = get_conic_network(input_size, lstm_hidden_size, mlp_hidden_layers)
        self.param_encoder_h = get_conic_network(input_size, lstm_hidden_size, mlp_hidden_layers)
        self.lstm_layers = lstm_layers

        self.rnn = nn.LSTM(self.output_size, lstm_hidden_size, lstm_layers, batch_first=True)
        self.lstm_decoder = nn.Linear(self.lstm_hidden_size, out_features=self.output_size)
    
    def with_pre_estimator(self, estimator):
        self.pre_estimator = estimator
        return self
    
    def forward(self, x, future=0):
        x_static = x["static"]
        static_parameters = x_static[:, 0, :]
        n_samples = x_static.size(0)
        
        # Compute Sequence using Pre-Estimator
        pre_estimated_sequences = self.pre_estimator(static_parameters, self.device)
        pre_estimated_sequences = pre_estimated_sequences.unsqueeze(2)

        if self.output_size > 1:
            pre_estimated_sequences = pre_estimated_sequences.view(n_samples, -1, self.output_size)
        
        # Encode Parameters
        h_t = encode_state(static_parameters, self.param_encoder_h, self.lstm_layers)
        c_t = encode_state(static_parameters, self.param_encoder_c, self.lstm_layers)

        # Map prediction using encoded hidden state and pre-estimation
        encoded_prediction, (h_t, c_t) = self.rnn(pre_estimated_sequences, (h_t, c_t))
        prediction = self.lstm_decoder(encoded_prediction)
        
        prediction = prediction.squeeze(-1)
        
        return prediction

  

def interpolate_elastic_deformation(psd_data: torch.tensor, psd_idx: torch.tensor, max_idx: int,
                                    padding_value=-10, device="cpu"):
    assert psd_data.size(0) == psd_idx.size(0), \
        f"Dimension 0 of psd_data and psd_idx must match. Found {psd_data.size(0)} {psd_idx.size(0)}"
    n_samples = psd_data.size(0)
    elastic_deformation = []

    for n in range(n_samples):
        # Estimate linear part
        psd_idx_n = int(psd_idx[n])
        psd_n = float(psd_data[n])

        elastic_n = torch.linspace(start=0, end=psd_n, steps=psd_idx_n, device=device)
        elastic_n = left_pad_to_length(elastic_n, length=max_idx, value=padding_value)
        elastic_deformation += [elastic_n]

    elastic_deformation = torch.stack(elastic_deformation)
    elastic_deformation = elastic_deformation.unsqueeze(-1)
    return elastic_deformation, psd_idx


def encode_state(data, encoder, repeat):
    encoded_state = encoder(data)
    encoded_state = encoded_state.repeat(repeat, 1, 1)
    return encoded_state


def to_integer_tensor(t: torch.tensor):
    if torch.is_floating_point(t):
        t = torch.round(t)
        t = t.int()
    return t


def pad_tensors(tensor_list, length, value=-10):
    padded_tensors = [F.pad(tensor, (0, 0, length - tensor.shape[0], 0), value=value) for tensor in tensor_list]
    return torch.stack(padded_tensors)


def left_pad_to_length(tensor, length, value=-10):
    padded_tensor = F.pad(tensor, (length - tensor.shape[0], 0), value=value)
    return padded_tensor


def mask_from_idx(t, index):
    t = torch.zeros_like(t, dtype=torch.bool)
    
    for i, psd_idx in zip(range(len(index)), index):
        t[i][psd_idx:] = True