#!/bin/bash
# example usage: ./scripts/this_script.sh 10100 15100

foundation_1=$1
foundation_2=$2
flood_1=$3
flood_2=$4
road=$5

foundation_dir=$(printf "/wdata/_val/ensembled_preds/exp_%05d-%5d" "$foundation_1" "$foundation_2")
flood_dir=$(printf "/wdata/_val/ensembled_preds/exp_%05d-%5d" "$flood_1" "$flood_2")
road_dir=$(printf "/wdata/_val/preds/exp_%05d" "$road")

refined_dir=$(printf "/wdata/_val/refined_preds/exp_%05d-%5d" "$foundation_1" "$foundation_2")
vector_dir=$(printf "/wdata/_val/road_vectors/exp_%05d-%5d" "$foundation_1" "$foundation_2")
graph_dir=$(printf "/wdata/_val/road_graphs/exp_%05d-%5d" "$foundation_1" "$foundation_2")

#python tools/refine_road_mask.py --val --foundation $foundation_dir --road $road_dir
#python tools/road/vectorize.py --val --foundation $refined_dir
#python tools/road/to_graph.py --val --vector $vector_dir
python tools/road/insert_flood.py --val --graph $graph_dir --flood $flood_dir
