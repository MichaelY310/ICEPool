import pickle
import torch
import matplotlib.pyplot as plt
import numpy as np

# 'train_accs', 'val_accs', 'test_accs', 'best_val_result'
# with open("ENZYMES_edgefeatFalse_DSN0_lr0.001_/full_result", 'rb') as f:
# with open("ENZYMES_edgefeatFalse_DSNFalse_lr0.001_det_/full_result", 'rb') as f:
# with open("ENZYMES_edgefeatTrue_DSNFalse_lr0.001_det_/full_result", 'rb') as f:
# with open("DD_edgefeatFalse_DSN1_lr0.001_" + "/full_result", 'rb') as f:
# with open("DD_edgefeatTrue_DSN1_lr0.001_" + "/full_result", 'rb') as f:
# with open("DD_edgefeatFalse_DSNTrue_lr0.001_det_" + "/full_result", 'rb') as f:
# with open("DD_edgefeatTrue_DSNTrue_lr0.001_det_svdonly_" + "/full_result", 'rb') as f:
# with open("NCI1_edgefeatFalse_DSN1_lr0.001_" + "/full_result", 'rb') as f:
# with open("NCI1_edgefeatTrue_DSNTrue_lr0.001_det_" + "/full_result", 'rb') as f:
with open("PROTEINS_edgefeatFalse_DSNFalse_lr0.001_det_" + "/full_result", 'rb') as f:

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
# print(np.mean(sorted(bests[1:-1])))
# print(np.mean(sorted(bests)[2:]))
print(test_accs)
print(var(test_accs))
print(np.mean(sorted(test_accs)))
# print(np.mean(sorted(test_accs[1:-1])))
# print(np.mean(sorted(test_accs)[2:]))