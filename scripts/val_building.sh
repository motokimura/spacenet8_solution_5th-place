#!/bin/bash
# example usage: ./scripts/this_script.sh 10100 15100

foundation=$1
flood=$2

python tools/building/postproc_building.py --val --foundation /wdata/_val/preds/exp_$foundation --flood /wdata/_val/preds/exp_$flood