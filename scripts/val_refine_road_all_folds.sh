#!/bin/bash
# example usage: ./scripts/this_script.sh 10100 15100

foundation_1=$1
foundation_2=$2
flood_1=$3
flood_2=$4
road=$5

# XXX: 5-fold
./scripts/val_refine_road.sh $(($foundation_1+0)) $(($foundation_2+0)) $(($flood_1+0)) $(($flood_2+0)) $(($road+0))
./scripts/val_refine_road.sh $(($foundation_1+1)) $(($foundation_2+1)) $(($flood_1+1)) $(($flood_2+1)) $(($road+1))
./scripts/val_refine_road.sh $(($foundation_1+2)) $(($foundation_2+2)) $(($flood_1+2)) $(($flood_2+2)) $(($road+2))
./scripts/val_refine_road.sh $(($foundation_1+3)) $(($foundation_2+3)) $(($flood_1+3)) $(($flood_2+3)) $(($road+3))
./scripts/val_refine_road.sh $(($foundation_1+4)) $(($foundation_2+4)) $(($flood_1+4)) $(($flood_2+4)) $(($road+4))
