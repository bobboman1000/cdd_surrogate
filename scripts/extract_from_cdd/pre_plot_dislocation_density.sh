#!/bin/bash

gawk '{print $2, $10}' "$1"/dislocation_sum.dat > "$1"/dislocation_density_pre_plot.dat
