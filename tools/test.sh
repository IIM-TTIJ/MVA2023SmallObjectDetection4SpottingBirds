#!/usr/bin/env bash

CONFIG=$1
CHECKPOINT=$2

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \

python3 $(dirname "$0")/test.py $CONFIG $CHECKPOINT --launcher none ${@:3}