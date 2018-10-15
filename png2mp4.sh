#!/bin/bash
cd bifurcation/out
ffmpeg -framerate 2 -i phase_space_q=%03d.png -c:v libx264 -pix_fmt yuv420p -r 30 -y out.mp4
cd ../..
