#!/bin/bash

gawk '{print $2,$6,$8,$10,$12,$14,$16,$18,$20,$22,$24,$26,$28,$30,$32,$34}' "$1"/dislocation_sum.dat > "$1"/dislocation_sum_pre_plot.dat

gawk '$4==0 {print $2,$6,$8,$10,$12,$14,$16,$18,$20,$22,$24,$26,$28,$30,$32,$34}' "$1"/dislocation_ns.dat > "$1"/dislocation_ns_pre_plot_0.dat
gawk '$4==1 {print $2,$6,$8,$10,$12,$14,$16,$18,$20,$22,$24,$26,$28,$30,$32,$34}' "$1"/dislocation_ns.dat > "$1"/dislocation_ns_pre_plot_1.dat
gawk '$4==2 {print $2,$6,$8,$10,$12,$14,$16,$18,$20,$22,$24,$26,$28,$30,$32,$34}' "$1"/dislocation_ns.dat > "$1"/dislocation_ns_pre_plot_2.dat
gawk '$4==3 {print $2,$6,$8,$10,$12,$14,$16,$18,$20,$22,$24,$26,$28,$30,$32,$34}' "$1"/dislocation_ns.dat > "$1"/dislocation_ns_pre_plot_3.dat
gawk '$4==4 {print $2,$6,$8,$10,$12,$14,$16,$18,$20,$22,$24,$26,$28,$30,$32,$34}' "$1"/dislocation_ns.dat > "$1"/dislocation_ns_pre_plot_4.dat
gawk '$4==5 {print $2,$6,$8,$10,$12,$14,$16,$18,$20,$22,$24,$26,$28,$30,$32,$34}' "$1"/dislocation_ns.dat > "$1"/dislocation_ns_pre_plot_5.dat
gawk '$4==6 {print $2,$6,$8,$10,$12,$14,$16,$18,$20,$22,$24,$26,$28,$30,$32,$34}' "$1"/dislocation_ns.dat > "$1"/dislocation_ns_pre_plot_6.dat
gawk '$4==7 {print $2,$6,$8,$10,$12,$14,$16,$18,$20,$22,$24,$26,$28,$30,$32,$34}' "$1"/dislocation_ns.dat > "$1"/dislocation_ns_pre_plot_7.dat
gawk '$4==8 {print $2,$6,$8,$10,$12,$14,$16,$18,$20,$22,$24,$26,$28,$30,$32,$34}' "$1"/dislocation_ns.dat > "$1"/dislocation_ns_pre_plot_8.dat
gawk '$4==9 {print $2,$6,$8,$10,$12,$14,$16,$18,$20,$22,$24,$26,$28,$30,$32,$34}' "$1"/dislocation_ns.dat > "$1"/dislocation_ns_pre_plot_9.dat
gawk '$4==10 {print $2,$6,$8,$10,$12,$14,$16,$18,$20,$22,$24,$26,$28,$30,$32,$34}' "$1"/dislocation_ns.dat > "$1"/dislocation_ns_pre_plot_10.dat
gawk '$4==11 {print $2,$6,$8,$10,$12,$14,$16,$18,$20,$22,$24,$26,$28,$30,$32,$34}' "$1"/dislocation_ns.dat > "$1"/dislocation_ns_pre_plot_11.dat
