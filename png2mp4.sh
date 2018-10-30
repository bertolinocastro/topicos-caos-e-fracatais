#!/bin/bash

set -x

path=$1

var=$2

cd bifurcation/$path/$var
ffmpeg -framerate 2 -i phase_space_$var=%03d.png -c:v libx264 -pix_fmt yuv420p -r 30 -y out.mp4
