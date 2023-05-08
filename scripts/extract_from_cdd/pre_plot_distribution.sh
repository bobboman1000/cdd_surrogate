#!/bin/bash

gawk '{print $2,$4}' "$1"/distribution_moment.dat > "$1"/distribution_moment_pre_plot.dat

