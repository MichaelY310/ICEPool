import pickle
import torch
import matplotlib.pyplot as plt
import numpy as np

# 'train_accs', 'val_accs', 'test_accs', 'best_val_result'
# with open("COLLAB_edgefeat1_DSN1_lr0.001_/full_result", 'rb') as f:
# with open("COLLAB_edgefeat0_DSN1_lr0.001_/full_result", 'rb') as f:
# with open("COLLAB_edgefeat1_DSN1_lr0.001_SVDonly_/full_result", 'rb') as f:
# with open("COLLAB_edgefeat0_DSN1_lr0.001_/full_result", 'rb') as f:
# with open("/scr/yuan/eigenpooling-master/ENZYMES_edgefeat0_DSN1_lr0.001_/full_result", 'rb') as f:
with open("/scr/yuan/eigenpooling-master/ENZYMES_edgefeatTrue_DSNTrue_lr0.001_/full_result", 'rb') as f:
    all_results = pickle.load(f)

bests = []
test_accs = []
for result in all_results:
    print(result["best_val_result"])
    print(result["test_accs"])
    indices = np.where(np.array(result["val_accs"]) == result["best_val_result"])[0]
    test_acc = max(np.array(result["test_accs"])[indices])
    bests.append(result["best_val_result"])
    test_accs.append(test_acc)

print(np.mean(sorted(bests)))
print(test_accs)
print(np.mean(sorted(test_accs)))
