How to run

1. Fetch dataset (should be in abide_timeseries), use fetch-dataset.sh and fetch-labels.sh
2. Run final_encoding file, one pass, output should be stored in gcn_input folder (run using kernel since ipynb)
3. Run bare-gcn (command: python src/bare-gcn.py) should print out results per epoch

Results to be replicated:
Epoch 005 | loss=0.6863 | train_acc=0.540 | val_acc=0.576 | test_acc=0.511 | val_loss=0.6982 | test_loss=0.6944
Epoch 010 | loss=0.6543 | train_acc=0.630 | val_acc=0.554 | test_acc=0.596 | val_loss=0.7166 | test_loss=0.6697
Epoch 015 | loss=0.5966 | train_acc=0.682 | val_acc=0.543 | test_acc=0.606 | val_loss=0.8656 | test_loss=0.7536
Epoch 020 | loss=0.5616 | train_acc=0.711 | val_acc=0.554 | test_acc=0.638 | val_loss=1.0306 | test_loss=0.7684
Epoch 025 | loss=0.4967 | train_acc=0.779 | val_acc=0.565 | test_acc=0.670 | val_loss=1.0728 | test_loss=0.8108
Epoch 030 | loss=0.4650 | train_acc=0.802 | val_acc=0.543 | test_acc=0.691 | val_loss=1.1128 | test_loss=0.7821
Epoch 035 | loss=0.4288 | train_acc=0.818 | val_acc=0.543 | test_acc=0.713 | val_loss=1.1546 | test_loss=0.7638
Epoch 040 | loss=0.4539 | train_acc=0.821 | val_acc=0.565 | test_acc=0.723 | val_loss=1.1677 | test_loss=0.7459
Epoch 045 | loss=0.4510 | train_acc=0.825 | val_acc=0.576 | test_acc=0.745 | val_loss=1.1555 | test_loss=0.7223
Epoch 050 | loss=0.4051 | train_acc=0.829 | val_acc=0.565 | test_acc=0.745 | val_loss=1.1606 | test_loss=0.7254
