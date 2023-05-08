from functools import reduce
from typing import Optional, List, Union
import pytorch_lightning as pl
import torch.utils.data as t_data
import pandas as pd
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from sklearn.model_selection import train_test_split, KFold

from fccd.data.datasets import FastCDDDataset, MultiTargetDataset


def _binary_encode_values(df, key):
    unique_values = df[key].unique()
    col_names = unique_values.astype(str)
    new_df = pd.DataFrame([df[key] == unique_value for unique_value in unique_values], index=col_names).T
    new_df = new_df.astype(int)
    return new_df


def _sum_index_list(idx_list):
    return reduce(lambda a, b: a + b, idx_list)


class BaseDataModule(pl.LightningModule):

    def __init__(self, data: pd.DataFrame, target: Union[str, List[str]], group: str, time: str, psd: str, batch_size: int,
                 categoricals: List[str] = None, drop_cols: List[str] = None, num_dataloader_workers: int = 1, 
                 transform=None, split_dataset=False, multi_target=False, scale_by_group=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.psd = psd
        self.train_df = None
        self.test_df = None
        self.val_df = None
        self.predict_df = None
        self.num_dataloader_workers = num_dataloader_workers
        self.persistent_workers = True if num_dataloader_workers > 0 else False
        self.split_dataset = split_dataset
        self.scale_by_group = scale_by_group
        
        self.multi_target = multi_target

        self.y_scaler = None
        self.x_scaler = None
        self.transform = transform
        self.batch_size = batch_size
        self.drop_cols = drop_cols
        self.categoricals = categoricals
        self.time = time
        self.group = group
        self.target = target
        self.data = data
        
        self.__train_dataset = None
        self.__val_dataset = None
        self.__test_dataset = None
        self.__predict_dataset = None

        if self.categoricals is not None:
            encoded_categoricals = [_binary_encode_values(self.data, cat) for cat in self.categoricals]
            self.data = pd.concat([self.data] + encoded_categoricals, axis=1)
            self.data = self.data.drop(self.categoricals, axis=1)

        groups = data[group].unique()
        self.group_idx = [data[group] == group_key for group_key in groups]

    def setup(self, stage: Optional[str] = None) -> None:
        raise NotImplementedError

    def train_dataset(self):
        if self.__train_dataset is None:
            
            if not self.multi_target:
                self.__train_dataset = FastCDDDataset(
                    data=self.train_df,
                    target=self.target,
                    group=self.group,
                    time=self.time,
                    drop_cols=self.drop_cols,
                    default_transform=self.transform,
                    psd=self.psd,
                    scale_by_group=self.scale_by_group,
                    compute_split=self.split_dataset
                )
            else:
                self.__train_dataset = MultiTargetDataset(
                    data=self.train_df,
                    group=self.group,
                    time=self.time,
                    drop_cols=self.drop_cols,
                    default_transform=self.transform,
                    psd=self.psd,
                    compute_split=self.split_dataset
                )

        return self.__train_dataset

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        cdd_dataset = self.train_dataset()
        self.x_scaler, self.y_scaler = cdd_dataset.x_scaler, cdd_dataset.y_scaler
        train_dataloader = t_data.DataLoader(
            cdd_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_dataloader_workers,
            persistent_workers=self.persistent_workers
        )
        return train_dataloader

    def val_dataset(self):
        if self.__val_dataset is None:
            if not self.multi_target:
                self.__val_dataset = FastCDDDataset(
                    data=self.val_df,
                    target=self.target,
                    group=self.group,
                    time=self.time,
                    drop_cols=self.drop_cols,
                    x_scaler=self.x_scaler,
                    y_scaler=self.y_scaler,
                    psd=self.psd,
                    scale_by_group=self.scale_by_group,
                    compute_split=self.split_dataset
                )
            else:
                self.__val_dataset = MultiTargetDataset(
                    data=self.val_df,
                    group=self.group,
                    time=self.time,
                    drop_cols=self.drop_cols,
                    x_scaler=self.x_scaler,
                    y_scaler=self.y_scaler,
                    psd=self.psd,
                    compute_split=self.split_dataset
                )
                
        return self.__val_dataset


    def val_dataloader(self) -> EVAL_DATALOADERS:
        return t_data.DataLoader(
            self.val_dataset(),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_dataloader_workers,
            persistent_workers=self.persistent_workers
        )

    def test_dataset(self):
        if self.__test_dataset is None:
            if not self.multi_target:
                self.__test_dataset = FastCDDDataset(
                    data=self.test_df,
                    target=self.target,
                    group=self.group,
                    time=self.time,
                    drop_cols=self.drop_cols,
                    x_scaler=self.x_scaler,
                    y_scaler=self.y_scaler,
                    psd=self.psd,
                    scale_by_group=self.scale_by_group,
                    compute_split=self.split_dataset
                )
            else:
                self.__test_dataset = MultiTargetDataset(
                data=self.test_df,
                group=self.group,
                time=self.time,
                drop_cols=self.drop_cols,
                x_scaler=self.x_scaler,
                y_scaler=self.y_scaler,
                psd=self.psd,
                compute_split=self.split_dataset
            )
        return self.__test_dataset


    def test_dataloader(self) -> TRAIN_DATALOADERS:
        return t_data.DataLoader(
            self.test_dataset(),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_dataloader_workers,
            persistent_workers=self.persistent_workers
            )

    def predict_dataset(self):
        if self.__predict_dataset is None:
            if not self.multi_target:
                self.__predict_dataset = FastCDDDataset(
                data=self.predict_df,
                target=self.target,
                group=self.group,
                time=self.time,
                drop_cols=self.drop_cols,
                x_scaler=self.x_scaler,
                y_scaler=self.y_scaler,
                psd=self.psd,
                scale_by_group=self.scale_by_group,
                compute_split=self.split_dataset
                )
            else:
                self.__predict_dataset = MultiTargetDataset(
                data=self.predict_df,
                group=self.group,
                time=self.time,
                drop_cols=self.drop_cols,
                x_scaler=self.x_scaler,
                y_scaler=self.y_scaler,
                psd=self.psd,
                compute_split=self.split_dataset
                )
                

        return self.__predict_dataset

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return t_data.DataLoader(
            self.predict_dataset(),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_dataloader_workers,
            persistent_workers=self.persistent_workers
        )
        
    def reset_data(self):
        self.__train_dataset = None
        self.__val_dataset = None
        self.__test_dataset = None
        self.__predict_dataset = None


class CDDDataModule(BaseDataModule):

    def __init__(self, data: pd.DataFrame, target: Union[str, List[str]], group: str, time: str, psd: str, batch_size: int,
                 categoricals: List[str] = None, drop_cols: List[str] = None, num_dataloader_workers: int = 1,
                 transform=None, split_dataset=False, scale_by_group=False, multi_target=False,
                 test_size: float = 0.1, val_size: float = 0.1,
                 *args, **kwargs):
        """ Datamodule to train pytorch lightning models. handles preprocessing like encoding categorical values and scaling

        Args:
            data (pd.DataFrame): pd.DataFrame of time series
            target (Union[str, List[str]]): name of target in df
            group (str): name of group idx in df
            time (str): name of time idx in df
            psd (str): name of psd in df
            batch_size (int): batch size for training
            categoricals (List[str], optional): names of categorical columns to encode (binary encoding). Defaults to None.
            drop_cols (List[str], optional): list of columns to drop before training (like other targets). Defaults to None.
            num_dataloader_workers (int, optional): Dataloader workers. For GPU training use value > 1, for CPU training and debugging use 0. Defaults to 1.
            transform (_type_, optional): Provide a scaler to be fitted or an already fitted one. Defaults to None.
            split_dataset (bool, optional): Compute split True/False. True will add an additional value to resulting y-dict. Increases computation time, use only for LSTM+PDP model Defaults to False.
            scale_by_group (bool, optional): Experimental by group (scale each series to 0/1 interval). Distorts validity of result because it removes magnitude Defaults to False.
            multi_target (bool, optional): True for multi-target training. Defaults to False.
            test_size (float, optional): percentage of total data to create test set. Defaults to 0.1.
            val_size (float, optional): percentage of training data to use as validation sets Defaults to 0.1.
        """
        
        super().__init__(data, target, group, time, psd, batch_size, categoricals, drop_cols, num_dataloader_workers, 
                         transform, split_dataset, multi_target, scale_by_group, *args, **kwargs)

        self.val_size = val_size
        self.test_size = test_size
        self.save_hyperparameters(logger=False)

    def setup(self, stage: Optional[str] = None) -> None:
        train_idx_list, test_idx_list = train_test_split(self.group_idx, test_size=self.test_size)
        train_idx_list, val_idx_list = train_test_split(train_idx_list, test_size=self.val_size)

        train_idx = _sum_index_list(train_idx_list)
        test_idx = _sum_index_list(test_idx_list)
        val_idx = _sum_index_list(val_idx_list)

        self.train_df = self.data[train_idx]
        self.test_df = self.data[test_idx]
        self.val_df = self.data[val_idx]
        self.predict_df = self.data
        
        self.train_dataloader()  # Assert scaler is set up


class KfoldCDDDataModule(BaseDataModule):
    
    def __init__(self, data: pd.DataFrame, k: int, num_folds: int, target: str, group: str, time: str,
                 psd: str, batch_size: int, categoricals: List[str] = None, drop_cols: List[str] = None, transform=None, split_dataset=False, scale_by_group=False, 
                 num_dataloader_workers: int = 1, multi_target=False, split_seed=None, *args, **kwargs):
        """ Datamodule to train pytorch lightning models. handles preprocessing like encoding categorical values and scaling
        Returns cross-validated dataset
        
        
        Example usage: 
        
            kth_fold = partial(
            KfoldCDDDataModule,
            data=train_data,
            target=target,
            psd="psd",
            group="id",
            drop_cols=drop_cols,
            time="t",
            batch_size=batch_size,
            categoricals=categoricals,
            num_dataloader_workers=4,
            transform=MinMaxScaler,
            split_dataset=do_split
            )       

            for k in range(num_folds):
                datamodule = kth_fold(k=k, num_folds=num_folds)
                datamodule.setup()

        Args:
            data (pd.DataFrame): pd.DataFrame of time series
            k (int): Fold index
            num_folds: int: Number of total folds
            target (Union[str, List[str]]): name of target in df
            group (str): name of group idx in df
            time (str): name of time idx in df
            psd (str): name of psd in df
            batch_size (int): batch size for training
            categoricals (List[str], optional): names of categorical columns to encode (binary encoding). Defaults to None.
            drop_cols (List[str], optional): list of columns to drop before training (like other targets). Defaults to None.
            num_dataloader_workers (int, optional): Dataloader workers. For GPU training use value > 1, for CPU training and debugging use 0. Defaults to 1.
            transform (_type_, optional): Provide a scaler to be fitted or an already fitted one. Defaults to None.
            split_dataset (bool, optional): Compute split True/False. True will add an additional value to resulting y-dict. Increases computation time, use only for LSTM+PDP model Defaults to False.
            scale_by_group (bool, optional): Experimental by group (scale each series to 0/1 interval). Distorts validity of result because it removes magnitude Defaults to False.
            multi_target (bool, optional): True for multi-target training. Defaults to False.
            test_size (float, optional): percentage of total data to create test set. Defaults to 0.1.
            val_size (float, optional): percentage of training data to use as validation sets Defaults to 0.1.
            split_seed (_type_, optional): Seed index. Must be consistent when computing multiple folds !!! Defaults to None.

        """
        
        super().__init__(data, target, group, time, psd, batch_size, categoricals, drop_cols, num_dataloader_workers, 
                         transform, split_dataset, multi_target, scale_by_group, *args, **kwargs)
        self.k = k
        self.num_splits = num_folds
        self.split_seed = split_seed
        self.save_hyperparameters(logger=False)

    def setup(self, stage: Optional[str] = None) -> None:
        kf = KFold(n_splits=self.num_splits, shuffle=True, random_state=self.split_seed)
        all_splits = [k for k in kf.split(self.group_idx)]

        train_fold_idx_list, test_fold_idx_list = all_splits[self.k]

        train_idx_list = [self.group_idx[fold_idx] for fold_idx in train_fold_idx_list]
        test_idx_list = [self.group_idx[fold_idx] for fold_idx in test_fold_idx_list]
        train_idx_list, val_idx_list = train_test_split(train_idx_list, test_size=0.1)

        train_idx = _sum_index_list(train_idx_list)
        test_idx = _sum_index_list(test_idx_list)
        val_idx = _sum_index_list(val_idx_list)

        self.train_df = self.data[train_idx]
        self.test_df = self.data[test_idx]
        self.val_df = self.data[val_idx]
        self.predict_df = self.data

        self.train_dataloader()  # Assert scaler is set up
