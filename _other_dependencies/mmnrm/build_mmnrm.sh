#!/bin/bash

cd /home/tiagoalmeida/mmnrm

rm -r ./dist

/home/tiagoalmeida/Spatial-RNN-GRU/tf2/bin/python3 setup.py sdist

/home/tiagoalmeida/Spatial-RNN-GRU/tf2/bin/pip install ./dist/mmnrm-0.0.2.tar.gz

