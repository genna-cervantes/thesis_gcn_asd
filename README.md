How to run

1. Fetch dataset (should be in abide_timeseries), use fetch-dataset.sh and fetch-labels.sh
2. Run final_encoding file, one pass, output should be stored in gcn_input folder (run using kernel since ipynb)
3. Run bare-gcn (command: python src/bare-gcn.py) should print out results per epoch
