from typing import List, Union, Tuple
import pandas as pd
import torch
import torch.utils.data as t_data
from sklearn.base import TransformerMixin
from joblib import Parallel, delayed

pd.options.mode.chained_assignment = None

class FastCDDDataset(t_data.Dataset):

    def __init__(self, data: pd.DataFrame, target: Union[str, List[str]], group: str, time: str, psd: str,
                 drop_cols: List[str]=None, default_transform=None, x_scaler=None, y_scaler=None, compute_split=False, scale_by_group=False, group_index=None):

        self.psd_data = data[psd]
        
        if group_index is None:
            self.group_idx = _build_group_idx(data, group)
        else:
            self.group_idx = group_index
        
        data = _drop_columns(data, drop_cols, additional_drop_cols=[group, time])

        assert bool(x_scaler) == bool(y_scaler) and not (bool(default_transform) and bool(x_scaler)), \
            "Please provide either a transformation or a fitted scaler (not both)"
            
        self.y = data[target]
        self.x = data.drop([target], axis=1)
        
        fit_scalers = False
        if x_scaler is None and y_scaler is None:
            x_scaler = default_transform()
            y_scaler = default_transform()
            fit_scalers = True
            
        self.scale_by_group = scale_by_group
        self.x, self.x_scaler = _apply_scaler(self.x, x_scaler, fit_scalers)
        self.y, self.y_scaler = _apply_scaler(self.y, y_scaler, fit_scalers)
        
        if target == "stress":
            self.psd_data, _ = _apply_scaler(self.psd_data, self.y_scaler, fit=False)
        else:
            self.psd_data = self.x[psd]

        self.split = compute_split
        self.groups = self._collect_groups(self.group_idx, compute_split=compute_split)
        
    def _collect_groups(self, group_idx, compute_split):
        def group_from_idx(idx):
            psd_data = _tensor_from_idx(self.psd_data, idx)
            y_data = _tensor_from_idx(self.y, idx)
            
            if self.scale_by_group:
                # TODO Unify this with actual scaling
                y_data, scalars = self.rescale(y_data)
                psd_data, _ = self.rescale(psd_data, scalars)
            psd_data = psd_data.unsqueeze(-1)
            
            x = {
                "static": _tensor_from_idx(self.x, idx),
                "psd": psd_data,
                "names": list(self.x.columns)
            }
            y = {
                "y": y_data
            }
            
            if compute_split:
                split_idx = _get_split_idx(y_data, psd_data)
                y["psd_idx"] = split_idx
                
            return (x, y)
        
        return [group_from_idx(idx) for idx in group_idx]  
    
    def rescale(self, t: torch.tensor, scalars=None):
        n_dims = len(t.shape)
        t_min = 0
        t_max = scalars
        if scalars is None and n_dims <= 1:
            t_max = t.max()
        if scalars is None and n_dims == 2:
            t_max = t.max(dim=0)[0]
            t_max = t_max.unsqueeze(0)
        
        return (t - t_min) / t_max, scalars

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        item, label = self.groups[idx]
        return item, label


def _apply_scaler(data, scaler, fit=True):
    if scaler is None:
        return data
    
    data_np = data.to_numpy()

    if len(data_np.shape) < 2:
        data_np = data_np.reshape(-1, 1)

    if fit:
        scaler.fit(data_np)

    data_np = scaler.transform(data_np)

    if type(data) == pd.DataFrame:
        data = pd.DataFrame(data_np, columns=data.columns, index=data.index)
    elif type(data) == pd.Series:
        data_np = data_np.squeeze()
        data = pd.Series(data_np, index=data.index)

    return data, scaler


class TabularDataset:

    def __init__(self, data: Union[Tuple[pd.DataFrame], List[pd.DataFrame], pd.DataFrame],
                 scaler: Union[List[TransformerMixin], TransformerMixin], x_names=None):
        self.raw_data = data
        self.scaler = scaler
        self.dataset = None
        self.x_names = x_names

    @staticmethod
    def psd_from_cdd_dataset(data: FastCDDDataset):
        assert data.split, "Split must be available in data. Initialize dataset with compute_split set to True"
        params = [x["static"][1, :] for x, _ in data]
        params = torch.stack(params)

        psd_data = [x["psd"][1, :] for x, _ in data]
        psd_data = torch.stack(psd_data)

        x_data = torch.hstack([params, psd_data])

        y_data = [y["psd_idx"] for _, y in data]
        y_data = torch.stack(y_data)

        raw_data = data.x, data.y
        scaler = [data.x_scaler, data.y_scaler]
        tab_data = TabularDataset(raw_data, scaler)
        tab_data.dataset = x_data, y_data
        return tab_data
    
    @staticmethod
    def max_y_from_cdd_dataset(data: FastCDDDataset):
        assert data.split, "Split must be available in data. Initialize dataset with compute_split set to True"
        params = [x["static"][1, :] for x, _ in data]
        params = torch.stack(params)

        x_data = params
        
        x_names = [x["names"] for x, _ in data]

        y_data = [y["y"][-1] for _, y in data]
        y_data = torch.stack(y_data)

        raw_data = data.x, data.y
        scaler = [data.x_scaler, data.y_scaler]
        tab_data = TabularDataset(raw_data, scaler, x_names)
        tab_data.dataset = x_data, y_data
        return tab_data

    def get_dataset(self):
        return self.dataset


def _drop_columns(data: pd.DataFrame, drop_cols: List[str], additional_drop_cols: List[str]) -> pd.DataFrame:
    drop = additional_drop_cols
    if drop_cols is not None:
        drop += drop_cols
    return data.drop(drop, axis=1)


def _build_group_idx(data, group):
    group_keys = data[group].unique()
    map_to_group_idx = lambda data, group, group_key: data[group] == group_key
    group_idx = [map_to_group_idx(data, group, group_key) for group_key in group_keys]
    return group_idx


def _tensor_from_idx(collection, idx):
    return torch.tensor(collection[idx].values).float()


def _get_split_idx(data: torch.tensor, split: torch.tensor) -> int:
    # Bring tensor in 3-dim shape

    if len(data.shape) <= 2:
        data = data.unsqueeze(0)

    if len(data.shape) == 2:
        data = data.unsqueeze(2)

    if len(split.shape) <= 2:
        split = split.unsqueeze(0)

    if len(split.shape) == 2:
        split = split.unsqueeze(2)

    assert len(data.shape) == 3 and len(split.shape) == 3 and data.shape[0] == data.shape[2] == 1, \
        f"Please make sure input tensors have either 1 or 3 dimensions, current input split: {split.shape}, " \
        f"data: {data.shape}"
    data_greater_split_val = data > split
    data_greater_split_val = data_greater_split_val * 1  # TrueFalse -> 01
    split_idx = torch.argmax(data_greater_split_val, dim=1)
    split_idx = split_idx.squeeze()

    return split_idx


class MultiTargetDataset(t_data.Dataset):
    
    STRESS_KEY = "stress"
    STRAIN_KEY = "strain"
    DISLOC_KEY = "dislocation"

    def __init__(self, data: pd.DataFrame, group: str, time: str, psd: str,
                 drop_cols: List[str]=None, default_transform=None, x_scaler=None, y_scaler=None, compute_split=False, group_index=None):

        self.psd_data = data[psd]


        if group_index is None:
            self.group_idx = _build_group_idx(data, group)
        else:
            self.group_idx = group_index
        
        data = _drop_columns(data, drop_cols, additional_drop_cols=[group, time])

        assert bool(x_scaler) == bool(y_scaler) and not (bool(default_transform) and bool(x_scaler)), \
            "Please provide either a transformation or a fitted scaler (not both)"
        
        target = [self.STRESS_KEY, self.STRAIN_KEY, self.DISLOC_KEY]
        self.y = data[target]
        self.x = data.drop(target, axis=1)
        
        fit_scaler = False
        
        if x_scaler is None and y_scaler is None:
            x_scaler = default_transform()
            stress_scaler = default_transform()
            strain_scaler = default_transform()
            disloc_scaler = default_transform()
            fit_scaler = True
        else:
            stress_scaler, strain_scaler, disloc_scaler = y_scaler
            
        self.x, self.x_scaler = _apply_scaler(self.x, x_scaler)
            
        self.y.loc[:, self.STRESS_KEY], stress_scaler = _apply_scaler(self.y.loc[:, self.STRESS_KEY], stress_scaler, fit=fit_scaler)
        self.y.loc[:, self.STRAIN_KEY], strain_scaler = _apply_scaler(self.y.loc[:, self.STRAIN_KEY], strain_scaler, fit=fit_scaler)
        self.y.loc[:, self.DISLOC_KEY], disloc_scaler = _apply_scaler(self.y.loc[:, self.DISLOC_KEY], disloc_scaler, fit=fit_scaler)
        
        self.y_scaler = stress_scaler, strain_scaler, disloc_scaler
        
        self.split = compute_split
        self.psd_data, _ = _apply_scaler(self.psd_data, stress_scaler, fit=False)
        self.groups = self._collect_groups(self.group_idx, compute_split=compute_split)
        
    def _collect_groups(self, group_idx, compute_split):
        def group_from_idx(idx):
            psd_data = _tensor_from_idx(self.psd_data, idx)
            y_data = _tensor_from_idx(self.y, idx)
            psd_data = psd_data.unsqueeze(-1)
            
            x = {
                "static": _tensor_from_idx(self.x, idx),
                "psd": psd_data,
                "names": list(self.x.columns)
            }
            y = {
                "y": y_data,
                "names": list(self.y.columns)
            }
            
            if compute_split:
                stress_t = _tensor_from_idx(self.y[self.STRESS_KEY], idx)
                split_idx = _get_split_idx(stress_t, psd_data)
                y["psd_idx"] = split_idx
                
            return (x, y)
        
        return [group_from_idx(idx) for idx in group_idx]  
    
    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        item, label = self.groups[idx]
        return item, label