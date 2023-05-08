import re
import sys, os
from os.path import join, isdir

import numpy as np
import pandas as pd


# Format of name: SERIES1_SERIES2. Used for naming columns
TO_COLLECT = {
    ("strain", "stress"): "strain-stress_yy_pre_plot.dat",
    ("time_ns", "dislocation"): "dislocation_density_pre_plot.dat",
}

EXP_DIR_PATTERN = "tensile_.*"


def collect_exp(path: str, id: str) -> pd.DataFrame:
    params = pd.read_csv(path + "/params.csv", index_col=0)
    params["id"] = id

    series = []
    col_names = []
    for exp_name, filename in TO_COLLECT.items():
        filepath = path + "/" + filename
        data = np.loadtxt(filepath)

        if isinstance(exp_name, tuple):
            for idx in range(data.shape[1]):
                series.append(data[:, idx])
                col_names.append(exp_name[idx])
        else:
            series.append(data[:])
            col_names.append(exp_name)

    series_df = pd.DataFrame(series, index=col_names).T

    exp_df = series_df.join(params, how="cross")
    exp_df = exp_df.reset_index(drop=False)
    exp_df = exp_df.rename({"index": "t"}, axis=1)

    return exp_df


def exp_id_from_foldername(name: str) -> str:
    assert re.match(EXP_DIR_PATTERN, name), f"Experiment name {name} doesnt match the pattern {EXP_DIR_PATTERN}"
    exp_id = name
    return exp_id


def is_exp_dir(root_folder, folder_name):
    full_path = join(root_folder, folder_name)
    dir_name_matches = re.match(EXP_DIR_PATTERN, folder_name) is not None
    return isdir(full_path) and dir_name_matches


def main(argv):
    assert len(argv) == 2, f"Please provide 2 args (path to experiment, experiment name), {len(argv)} were provided"
    path = argv[0]
    target_path = argv[1]
    ls = os.listdir(path)

    experiment_directories = [(path, subdir) for subdir in ls if is_exp_dir(path, subdir)]

    dfs_list = []
    for root_dir, directory in experiment_directories:

        # Get ids from folder name
        exp_id = exp_id_from_foldername(directory)

        full_path = os.path.join(root_dir, directory)
        dfs_list.append(collect_exp(path=full_path, id=exp_id))

    assert len(dfs_list) != 0, "No experiment folders found in given directory"
    dataset = pd.concat(dfs_list)
    dataset = dataset.reset_index(drop=True)

    if target_path.endswith("/"):
        target_path = target_path[:-1]
    dataset.to_csv(target_path + "/dataset.csv")


if __name__ == "__main__":
    main(sys.argv[1:])