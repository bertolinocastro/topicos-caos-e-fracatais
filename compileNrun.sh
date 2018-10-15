#!/bin/bash

set -x

file=$1

python3 -O -m py_compile $file
python3 __pycache__/$file.cpython-36.opt-1.pyc 
