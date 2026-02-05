#!/bin/bash


python='/Users/jjv/miniconda3/envs/main/bin/python'
# pvpython='/Applications/ParaView-6.0.1.app/Contents/bin/pvpython'

for method in doste bayer; do
    $python main_${method}.py
    $python validation_${method}.py
    # $pvpython paraview_${method}.py
done
