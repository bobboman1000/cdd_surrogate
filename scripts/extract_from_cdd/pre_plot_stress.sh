#!/bin/bash
gawk '{print $3}' "$1"/stress_norm.dat > "$1"/stress_norm_pre_plot.dat
gawk '{print $3}' "$1"/strain_norm.dat > "$1"/strain_norm_pre_plot.dat
gawk '{print $4,$6}' "$1"/strain-stress_yy.dat > "$1"/strain-stress_yy_pre_plot.dat
gawk '{print $4,$6}' "$1"/strain-stress_xy.dat > "$1"/strain-stress_xy_pre_plot.dat
