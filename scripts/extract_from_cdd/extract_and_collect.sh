#!/bin/bash
path=$1 # Path to experiment folder. Should contain folders of the form "data_studie*"
target_path=$2 # Path where to save dataset

# Number of processes for extracting data
N=4

process_exp() {
   # Extract all data
   python extract.py "$1"

   # Compute time series
   ./pre_plot_stress.sh "$1"
   ./pre_plot_distribution.sh "$1"
   ./pre_plot_dislocation_density.sh "$1"

    echo "Finished $file $finished / $((todo - 1))"
}

export -f process_exp
find $path/tensile* | parallel process_exp

wait

echo "Collecting results..."
python collect.py "$path" "$target_path"

echo "******* FINISHED ＼(＾O＾)／ *******"