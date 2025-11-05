#!/bin/bash

OUT_DIR="src/abide_timeseries"

aws s3 cp \
  s3://fcp-indi/data/Projects/ABIDE_Initiative/Phenotypic_V1_0b.csv \
  "$OUT_DIR/phenotypic.csv" \
  --no-sign-request

aws s3 cp \
  s3://fcp-indi/data/Projects/ABIDE_Initiative/Phenotypic_V1_0b_preprocessed.csv \
  "$OUT_DIR/phenotypic_preprocessed1.csv" \
  --no-sign-request

aws s3 cp \
  s3://fcp-indi/data/Projects/ABIDE_Initiative/Phenotypic_V1_0b_preprocessed1.csv \
  "$OUT_DIR/phenotypic_preprocessed2.csv" \
  --no-sign-request

echo "phenotypic data fetched"
