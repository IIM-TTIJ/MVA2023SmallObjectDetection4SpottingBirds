#!/usr/bin/env bash

CONFIG=$1

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \

python3 $(dirname "$0")/train.py $CONFIG --seed 0 --launcher none
