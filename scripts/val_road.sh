#!/bin/bash
# example usage: ./scripts/this_script.sh 10100 15100

foundation=$1
flood=$2

python tools/road/vectorize.py --val --foundation /wdata/_val/preds/exp_$foundation
python tools/road/to_graph.py --val --vector /wdata/_val/road_vectors/exp_$foundation
python tools/road/insert_flood.py --val --graph /wdata/_val/road_graphs/exp_$foundation --flood /wdata/_val/preds/exp_$flood
