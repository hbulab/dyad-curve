#!/bin/bash

# remove old images
rm ../data/figures/02_07_meta_trajectories/*.pdf
rm ../data/figures/02_08_velocity_curvature/*.pdf
rm ../data/figures/02_08_trajectories_with_velocity_curvature/*.pdf
rm ../data/figures/02_08_velocity_curvature_binned_velocity.pdf
rm ../data/figures/02_08_velocity_curvature_binned_log_velocity_radius.pdf
rm ../data/figures/02_08_velocity_curvature_binned_log_velocity.pdf
rm ../data/figures/02_08_velocity_curvature.pdf

# run the scripts
echo "Running 02_07_compute_meta_trajectories_with_spatial_binning.py"
python 02_07_compute_meta_trajectories_with_spatial_binning.py
echo "Running 02_08_compute_velocity_curvature.py"
python 02_08_compute_velocity_curvature.py
