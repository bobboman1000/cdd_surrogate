from hashlib import new
from unittest import TestCase
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from fccd.data.datasets import _apply_scaler, _drop_columns, _build_group_idx, DeformPointSplitCDDDataset, \
    FastCDDDataset, _split_tensor_at

import pandas as pd

from fccd.data.modules import SplitCDDDataModule


class TestDeformation(TestCase):

    data = pd.read_csv("./resources/test.csv", index_col=0)

    data["stress"] = data["stress"] * 1000

    # Wrong unit conversion. This is for testing purposes
    data["psd"] = data["psd"] * 100 + 5

    def test_splitdataset(self):
        split_cdd_dataset = DeformPointSplitCDDDataset(
            data=self.data,
            target="stress",
            group="id",
            plastic_deformation_point="psd",
            time="t",
            drop_cols=["euler_angles", "material"],
            default_transform=MinMaxScaler,
            statics=["E", "nu"],
            dynamics=["strain"]
        )
        self.x_scaler, self.y_scaler = split_cdd_dataset.x_scaler, split_cdd_dataset.y_scaler
        self.assertTrue(len(split_cdd_dataset.groups) == 10)
        self.assertTrue(np.all(0 <= split_cdd_dataset.psd_data) and np.all(1 >= split_cdd_dataset.psd_data))
        self.assertTrue("euler_angles" not in split_cdd_dataset.x_stat and "euler_angles" not in split_cdd_dataset.x_dyn)

    def test_split_datamodule(self):
        split_stress_dm = SplitCDDDataModule(
            self.data,
            target="stress",
            psd_key="psd",
            group="id",
            drop_cols=["strain"],
            time="t",
            batch_size=2,
            categoricals=["material", "euler_angles"],
            num_workers=4,
            transform=MinMaxScaler
        )
        split_stress_dm.setup()

        index_sum = split_stress_dm.train_df.index.union(split_stress_dm.val_df.index)
        index_sum = index_sum.union(split_stress_dm.test_df.index)
        self.assertTrue(len(index_sum) == 100)

        for x, y in iter(split_stress_dm.train_dataloader()):
            self.assertTrue(x["static"].shape == (2, 10, 22))
            self.assertTrue(x["static"].shape == (2, 10, 22))
            self.assertTrue(x["static"][0, :].shape == (2, 22))
            self.assertTrue(y["before"].shape[0] == 2)
            self.assertTrue(y["before"].shape[1] <= 10)
            self.assertTrue(y["before"].shape[2] == 1)
            self.assertTrue(y["after"].shape[0] == 2)
            self.assertTrue(y["after"].shape[1] <= 10)
            self.assertTrue(y["after"].shape[2] == 1)

    def test_fastdata(self):
        split_cdd_dataset = FastCDDDataset(
            data=self.data,
            target="stress",
            group="id",
            time="t",
            drop_cols=["euler_angles", "material"],
            default_transform=MinMaxScaler,
            statics=["E", "nu"],
            dynamics=["strain"],
            psd="psd"
        )
        self.x_scaler, self.y_scaler = split_cdd_dataset.x_scaler, split_cdd_dataset.y_scaler

    def test_scaler(self):
        scaler = MinMaxScaler()
        num_data = self.data.drop(["material", "euler_angles", "id"], axis=1)
        scaled_data, scaler = _apply_scaler(num_data, scaler)

        self.assertTrue(np.all(0 <= scaled_data) and np.all(scaled_data <= 1))
        self.assertTrue(np.all(num_data.shape == scaled_data.shape))
        self.assertTrue(np.all(num_data.index == scaled_data.index))
        self.assertTrue(np.all(num_data.columns == scaled_data.columns))

        new_scaled_data, new_scaler = _apply_scaler(num_data, scaler, fit=False)
        self.assertTrue(np.all(0 <= new_scaled_data) and np.all(new_scaled_data <= 1))
        self.assertTrue(np.all(num_data.shape == scaled_data.shape))
        self.assertTrue(np.all(num_data.index == scaled_data.index))
        self.assertTrue(np.all(num_data.columns == scaled_data.columns))

        self.assertTrue(scaler == new_scaler)

        # Avoid floating point errors
        self.assertTrue(np.all(new_scaled_data - scaled_data < 0.001))

    def test_drop_columns(self):
        dropped_data = _drop_columns(self.data, drop_cols=["strain"], additional_drop_cols=["psd"])

        self.assertTrue(np.all(self.data.shape[1] -2 == dropped_data.shape[1]))
        self.assertTrue(np.all(self.data.index == dropped_data.index))

    def test_build_group_idx(self):
        group_idx = _build_group_idx(self.data, group="id")

        self.assertTrue(len(group_idx) == 10)
        self.assertTrue(np.all([len(idx) == 100 for idx in group_idx]))
        self.assertTrue(np.all([sum(idx) == 10 for idx in group_idx]))

    def test_split_tensor_at(self):
        arr = np.asarray(self.data["t"])
        split = 4

        arr = torch.tensor(arr)
        _split_tensor_at(arr, split)
