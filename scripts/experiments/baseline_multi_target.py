from functools import partial
import pandas as pd
import pytorch_lightning as pl
from sklearn.preprocessing import MinMaxScaler

from fccd.data.modules import KfoldCDDDataModule
from fccd.models.GBT import LightGBMModel
from fccd.models.baseline import KNNBaseline, ResponseSurfaceMethod
import torch.multiprocessing
import argparse

from fccd.util import collect_dataloader, limit_psd

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import mean_squared_error

# Avoid "too many open files" errors with multi-gpu training
torch.multiprocessing.set_sharing_strategy('file_system')


# Eval Loop
def run(train_data, target, drop_cols, name, gbt_jobs=None, cv_mult=None, num_folds=10, batch_size=64):
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
        categoricals=["material", "mesh", "euler_angles"],
        num_dataloader_workers=8,
        transform=MinMaxScaler,
        split_dataset=True
    )       

    # ------- Train & Evaluation Loop -------
    def evaluate():
        losses = {}
        params_log = []
        for k in range(num_folds):
            datamodule = kth_fold(k=k, num_folds=num_folds)
            datamodule.setup()
            
            
            models = {
           "kNN": GridSearchCV(
               KNNBaseline(), 
               param_grid={
                   "k": [1, 3, 10],
                   "force_different_neighbour_columns": [True, False]
               },
               scoring="neg_mean_squared_error"
               ),
           
            "LightGBM": GridSearchCV(LightGBMModel(n_jobs=gbt_jobs), {
                "max_depth": [2, 10],
                "early_stopping_patience": [-1, 1, 10],
                "n_estimators": [100, 500],
                "learning_rate": [0.01, 0.1, 1]
                }, scoring="neg_mean_squared_error", n_jobs=cv_mult),
            
            "RSM": GridSearchCV(ResponseSurfaceMethod(), {
                "degree": [2, 3]
                }, scoring="neg_mean_squared_error"),
        }
            
            

            print(f"------------------------------------ Starting with fold {k} ----------------------------------------")
            for name, model in models.items():
                
                print(f"Training {name}...")
                print(f"Validating parameters {model.param_grid}")
                test_loss = train_sklearn_model(model, datamodule)
                print(f"Best parameters using {model.n_splits_} folds: {model.best_params_}")
                
                best_params =  [[name, k, key, value] for key, value in model.best_params_.items()]
                
                params_log += best_params
                
                if name not in losses:
                    losses[name] = []
                losses[name].append(test_loss)

                print(f" => Test loss: {test_loss}")
                

        return losses, params_log

    def train_sklearn_model(model, datamodule):
        train_dataloader = datamodule.train_dataloader()
        val_dataloader = datamodule.val_dataloader()

        X_train, y_train = collect_dataloader(train_dataloader)
        X_val, y_val = collect_dataloader(val_dataloader)
        model.fit(X_train, y_train, X_val=X_val, y_val=y_val)

        test_dataloader = datamodule.test_dataloader()
        X_test, y_test = collect_dataloader(test_dataloader)
        y_test_pred = model.predict(X_test)

        test_loss = mean_squared_error(y_test, y_test_pred)
        return test_loss

    loss, params = evaluate()
    results = pd.DataFrame(loss)
    results.to_csv(f"./{name}.csv")
    
    params = pd.DataFrame(params, columns=["model_name", "fold", "param_name", "param_value"])
    params.to_csv(f"./{name}_params.csv")
    
    
# -------------------- Utility Functions ------------------------

def train_sklearn_model(model, datamodule):
    train_dataloader = datamodule.train_dataloader()
    val_dataloader = datamodule.val_dataloader()

    X_train, y_train = collect_dataloader(train_dataloader)
    X_val, y_val = collect_dataloader(val_dataloader)
    model.fit(X_train, y_train, X_val=X_val, y_val=y_val)

    test_dataloader = datamodule.test_dataloader()
    X_test, y_test = collect_dataloader(test_dataloader)
    y_test_pred = model.predict(X_test)

    test_loss = mean_squared_error(y_test, y_test_pred)
    return test_loss

# ------------------------------ Main call ------------------------------------------

if __name__ == '__main__':
    
    pl.seed_everything(42)
    
    parser = argparse.ArgumentParser(
                        prog = 'Cdd Surrogate Experiment',
                        description = 'Running exps',
                        epilog = 'WIP')
        
    parser.add_argument('datasets', metavar='data', type=str, nargs='+', help='Datasets in data/processed')
    parser.add_argument('-n', '--nproc', default=0, type=int)      # option that takes a value
    parser.add_argument('-c', '--cvmult', default=None, type=int)      # option that takes a value
    parser.add_argument('-t', '--test', default=False, action="store_true")      # option that takes a value


    args = vars(parser.parse_args())
    test_run = args["test"]

    cvmult=args["cvmult"]
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
    
    # ----------------------------------
    
    if test_run:
        data = data[data["t"] < 30]
        num_folds = 2
        print("Running in test mode")
    
    # Limit PSD to max(stress)
    data = limit_psd(data)
    
    print(f"Runing experiments with on data {dataset_names}, using {gbt_jobs} jobs with CV-multi {cvmult}")
    
    suffix = ""
    for name in dataset_names:
        suffix += f"_{name}"
        
    prefix = "baseline_"
    
    run(data, "all", ["time_ns"], prefix + "stress_strain_disloc" + suffix, gbt_jobs, cvmult, num_folds, batch_size)

