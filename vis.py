import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

dataset_csv = "CSVs/InTheWild.csv"
dataset_name = "InTheWild"
scores_folder = "./scores"
strategy = "ms"

df_dataset = pd.read_csv(dataset_csv)
y = df_dataset['label'].tolist()

scores_path = os.path.join(scores_folder, f"{dataset_name}_{strategy}.csv")
df_results = pd.read_csv(scores_path)
scores = df_results['value'].tolist()

fpr, tpr, thresholds = metrics.roc_curve(y, scores)
auc = metrics.auc(fpr, tpr)


optimal_threshold_index = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_threshold_index]

plt.plot(np.arange(0,1.1,0.1),np.arange(0,1.1,0.1),color="gray", linestyle="--")
plt.plot(fpr,tpr,color="purple",linestyle="-",label=f"AUC {auc}")
plt.legend()
plt.scatter(fpr[optimal_threshold_index], tpr[optimal_threshold_index], color='red', label=f'Optimal Threshold ({optimal_threshold:.2f})')
print(f"Optimal Threshold {optimal_threshold:.4f} FPR: {fpr[optimal_threshold_index]:.4f} TPR {tpr[optimal_threshold_index]:.4f}")
plt.title(f"PoIForensics-Audio on {dataset_name}")
plt.ylabel("TPR")
plt.xlabel("FPR")
plt.show()

