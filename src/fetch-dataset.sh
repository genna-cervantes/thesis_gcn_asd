#!/bin/bash

OUT_DIR="./abide_timeseries"
mkdir -p "$OUT_DIR"

echo "downloading full CC200 time series..."
aws s3 cp \
  s3://fcp-indi/data/Projects/ABIDE_Initiative/Outputs/cpac/filt_global/rois_cc200/ \
  "$OUT_DIR/cc200/" \
  --recursive --no-sign-request

echo "downloading full AAL time series..."
aws s3 cp \
  s3://fcp-indi/data/Projects/ABIDE_Initiative/Outputs/cpac/filt_global/rois_aal/ \
  "$OUT_DIR/aal/" \
  --recursive --no-sign-request

echo "all ROI time series downloaded"
