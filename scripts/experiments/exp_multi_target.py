from functools import partial
import pandas as pd
import pytorch_lightning as pl
from sklearn.preprocessing import MinMaxScaler
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from fccd.data.modules import KfoldCDDDataModule
from fccd.models.GBT import LightGBMModel
from fccd.models.LSTM import EncodedHiddenStateGRU, EncodedHiddenStateLSTM, ChunkingEncodedHiddenStateLSTM, RepeatedInputLSTM, RepeatingChunkingEncodedHiddenStateLSTM, RawHiddenStateLSTM, SeqToSeqLSTM
import torch.multiprocessing
import argparse

from fccd.util import collect_dataloader, limit_psd

from sklearn.model_selection import GridSearchCV
from fccd.data.datasets import TabularDataset
from sklearn.ensemble import RandomForestRegressor

# Avoid "too many open files" errors with multi-gpu training
torch.multiprocessing.set_sharing_strategy('file_system')


# Eval Loop
def run(train_data, target, drop_cols, name, gbt_jobs=None, device=0, num_folds=10, batch_size=64, max_epochs=1500):
    
    categoricals = ["material", "mesh", "euler_angles"]
    input_size = 20 # TODO Move this to dataset, make dataset express its size
    
    kth_fold = partial(
        KfoldCDDDataModule,
        data=train_data,
        target=target,
        multi_target=True,
        psd="psd",
        group="id",
        drop_cols=drop_cols,
        time="t",
        batch_size=batch_size,
        categoricals=categoricals,
        num_dataloader_workers=4,
        transform=MinMaxScaler,
        split_dataset=True
    )

    # ------- Train & Evaluation Loop -------
    def evaluate():
        losses = {}
        for k in range(num_folds):
            datamodule = kth_fold(k=k, num_folds=num_folds)
            datamodule.setup()
            
            models = [
                SeqToSeqLSTM(input_size, 30, 2, 3, 1).with_pre_estimator(get_gbt_estimate(datamodule, gbt_jobs)),
                EncodedHiddenStateLSTM(input_size, 30, 2, 3, 1),
                EncodedHiddenStateGRU(input_size, 30, 2, 3, 1),
                RepeatedInputLSTM(input_size, 30, 3, 1),
                RawHiddenStateLSTM(input_size, 3, 1)
            ]
            
            if datamodule.split_dataset:
                psd_estimator = get_rf_regressor(datamodule)
                split_only_models = [
                    ChunkingEncodedHiddenStateLSTM(input_size, 30, 2, 1, 3, target_stress_idx=0).with_estimator(psd_estimator),
                ]
                models += split_only_models

            print(f"------------------------------------ Starting with fold {k} ----------------------------------------")
            for model in models:
                test_loss = train_torch_model(model, datamodule, device, max_epochs=max_epochs)
                
                if str(model) not in losses:
                    losses[str(model)] = []
                losses[str(model)].append(test_loss)
        return losses

    loss = evaluate()
    results = pd.DataFrame(loss)
    results.to_csv(f"./{name}.csv")
    
    
# -------------------- Utility Functions ------------------------

def train_torch_model(model, datamodule, device, max_epochs):
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=3e-6, patience=100, verbose=False, mode="min")
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu",
        devices=[device],
        callbacks=[early_stop_callback],
        log_every_n_steps=2,
        auto_lr_find=True
    )
    trainer.tune(model, datamodule=datamodule)
    trainer.fit(model, datamodule=datamodule)

    test_loss = trainer.test(model, datamodule=datamodule)[0]["test_loss"]
    return test_loss


def get_rf_regressor(datamodule):
    train_data = datamodule.train_dataset()
    val_data = datamodule.val_dataset()
    
    psd_train_data = TabularDataset.psd_from_cdd_dataset(train_data)
    psd_val_data = TabularDataset.psd_from_cdd_dataset(val_data)
    
    x_train, y_train = psd_train_data.dataset
    x_val, y_val = psd_val_data.dataset
    
    stackedx_train = torch.cat([x_train, x_val], dim=0)
    stackedy_train = torch.cat([y_train, y_val], dim=0)
    
    rf = RandomForestRegressor(n_estimators=1000)
    rf.fit(stackedx_train, stackedy_train)
    
    def estimate_rounded_rf(static_parameters, psd_data, device):
        regression_input = torch.hstack([static_parameters, psd_data])
        regression_input = regression_input.detach().clone()
        
        if regression_input.device != "cpu":
            regression_input = regression_input.cpu()

        prediction = rf.predict(regression_input)
        result = torch.from_numpy(prediction)
        result = result.to(device)
        return result
    
    return estimate_rounded_rf


def get_gbt_estimate(datamodule, gbt_jobs):
    # Prepare Data
    train_dataloader = datamodule.train_dataloader()
    val_dataloader = datamodule.val_dataloader()

    X_train, y_train = collect_dataloader(train_dataloader)
    X_val, y_val = collect_dataloader(val_dataloader)

    # Prepare Estimator
    pre_estimator = GridSearchCV(
        LightGBMModel(n_jobs=gbt_jobs), 
        {
            "max_depth": [2],
            "early_stopping_patience": [1],
            "n_estimators": [500],
            "learning_rate": [0.01, 0.1]
        }, 
        scoring="neg_mean_squared_error")
    pre_estimator.fit(X_train, y_train, X_val=X_val, y_val=y_val)

    model = pre_estimator
    # Wrap into function
    def pre_estimate_curve(static_parameters, device):
        regression_input = static_parameters.detach().clone()
        if regression_input.device != "cpu":
            regression_input = regression_input.cpu()

        prediction = model.predict(regression_input)
        result = torch.from_numpy(prediction)
        result = result.to(device, dtype=torch.float)
        return result  
    return pre_estimate_curve

# ------------------------------ Main call ------------------------------------------

if __name__ == '__main__':
    
    pl.seed_everything(42)
    
    parser = argparse.ArgumentParser(
                        prog = 'Cdd Surrogate Experiment',
                        description = 'Running exps',
                        epilog = 'WIP')
        
    parser.add_argument('datasets', metavar='data', type=str, nargs='+', help='Datasets in data/processed')
    parser.add_argument('-n', '--nproc', default=0, type=int)      # option that takes a value
    parser.add_argument('-g', '--gpu', default=0, type=int)      # option that takes a value
    parser.add_argument('-t', '--test', default=False, action="store_true")      # option that takes a value

    args = vars(parser.parse_args())
    
    device=args["gpu"]
    test_run = args["test"]
    gbt_jobs = args["nproc"]
    
    dataset_names = args["datasets"]
    datasets = [pd.read_csv(f"../../data/processed/{name}.csv", index_col=0) for name in dataset_names]
    data = pd.concat(datasets)
    data = data.reset_index(drop=True)

    # GPa to MPa
    data["stress"] = data["stress"] * 1000
    data["psd"] = data["psd"] * 1000
    
    # 0.01 to 1%
    data["strain"] = data["strain"] * 100
    
    
    #----- Global Training Params -------
    
    num_folds = 10
    batch_size = 64
    max_epochs = 1500
    
    # ----------------------------------
    
    if test_run:
        
        # Override params for quick test run
        data = data[data["t"] < 30]
        num_folds = 2
        max_epochs = 2
        
        print("Running in test mode")
    
    
    # Limit PSD to max(stress)
    data = limit_psd(data)
    
    print(f"Runing experiments with on data {dataset_names}, using gpu {device}.")
    
    suffix = ""
    for name in dataset_names:
        suffix += f"_{name}"
    
    run(data, "all", ["time_ns"], "stress_strain_disloc_" + suffix, gbt_jobs, device, num_folds, batch_size, max_epochs)
