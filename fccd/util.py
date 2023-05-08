import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


def plot_prediction_vs_truth_sklearn(dataloader, models: dict, range_):
    import seaborn as sns
    import matplotlib.pyplot as plt

    X, y = collect_dataloader(dataloader)

    predictions = [
        (key, models[key].predict(X)) for key in models
    ]

    for i in range(range_):
        for k, p in predictions:
            sns.lineplot(data=p[i], label=k)
            print("RMSE: ", np.sqrt(mean_squared_error(p[i], y[i])))
        sns.lineplot(data=y[i], label="Truth")
        
        sns.set(font_scale=1.6)
        sns.set_style({'font.family':'serif', 'font.serif':'Linux Libertine O'})

        plt.xlabel("t")
        plt.ylabel("Stress")
        
        plt.savefig(f"figure{i}.pdf", bbox_inches="tight")
        plt.show()


def plot_prediction_vs_truth(dataloader, models: dict, range_, prediction_length):
    import seaborn as sns
    import matplotlib.pyplot as plt

    pred_dl = dataloader
    x, y = next(iter(pred_dl))

    predictions = [(key, models[key](x, future=prediction_length)) for key in models]

    for i in range(range_):
        for k, p in predictions:
            sns.lineplot(data=p.detach().numpy()[i], label=k)
        sns.lineplot(data=y["y"].detach().numpy()[i], label="Truth")
        sns.lineplot(data=x["psd"].detach().numpy()[i], label="psd")
        plt.show()
        
        
def find_lr(model, datamodule, verbose=True, update_lr=False):
    import pytorch_lightning as pl
    import torch

    accelerator="cpu"
    num_devices=1
    if torch.cuda.is_available():
        accelerator="gpu"

    trainer = pl.Trainer(model, datamodule, accelerator=accelerator, devices=num_devices, max_epochs=100,
                         enable_checkpointing=False)
    # Run learning rate finder
    lr_finder = trainer.tuner.lr_find(model, datamodule=datamodule)

    if verbose:
        fig = lr_finder.plot(suggest=True)
        fig.show()

    # Pick point based on plot, or get suggestion
    new_lr = lr_finder.suggestion()

    if update_lr:
        model.hparams.lr = new_lr

    print(f"LR suggestion: {new_lr}")
    return new_lr


def collect_dataloader(dataloader):
    X_ = []
    y_ = []
    for x, y in iter(dataloader):
        x_data = x["static"][:, 0, :].detach().numpy()
        y_data = y["y"].detach().numpy()
        
        for x_sample, y_sample in zip(x_data, y_data):
            X_.append(x_sample)
            if len(y_sample.shape) == 2:
                y_sample = y_sample.flatten()
            y_.append(y_sample)
    X_ = np.asarray(X_)
    y_ = np.asarray(y_)

    return X_, y_


def limit_psd(data: pd.DataFrame) -> pd.DataFrame:
    ids = data["id"].unique()

    for id_ in ids:
        mask = data["id"] == id_
        max_stress = data[mask]["stress"].max()
        if np.all(max_stress < data.loc[mask, "psd"]):
            data.loc[mask, "psd"] = max_stress - 1e-4
    return data