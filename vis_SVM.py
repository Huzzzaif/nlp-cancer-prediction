import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# === Step 1: Convert true labels to binary (1 for BRCA, 0 for all others) ===
target_class_label = 2  # Assuming BRCA is class 2 in your label mapping
y_true_binary = (y_true == target_class_label).astype(int)

# === Step 2: Get decision scores from your OvR SVM model ===
# decision_function gives shape (n_samples, n_classes) — pick BRCA column
y_score = clf.decision_function(X)[:, target_class_label]

# === Step 3: Compute ROC curve and AUC ===
fpr, tpr, _ = roc_curve(y_true_binary, y_score)
roc_auc = auc(fpr, tpr)

# === Step 4: Plot the ROC curve ===
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2,
         label=f'ROC Curve for BRCA vs All (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve – SVM (BRCA vs All Classes)')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.savefig("results/svm_roc_curve_brca_vs_all.png")
plt.show()
