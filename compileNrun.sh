#!/bin/bash

set -x

file=$1

python3 -OO -m py_compile $file
python3 __pycache__/${file::-3}.cpython-36.opt-1.pyc
