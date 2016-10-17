#!/bin/bash

if [ -x $1 ]; then
    echo "Usage: get-model.sh {1,2,3}"
    exit 1
fi

GPU_MACHINE=$(cat ~/GPU_MACHINE$1)

scp -i ~/avital2.pem ubuntu@${GPU_MACHINE}:~/hebrew-please-4/train/architecture.json .
scp -i ~/avital2.pem ubuntu@${GPU_MACHINE}:~/hebrew-please-4/train/weights.hdf5 .
