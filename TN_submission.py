import numpy as np
from TN_Grid_Search import tune_catboost_with_grid
from TN_2 import ThyroidCancer2

from sklearn.metrics import f1_score, classification, precision_recall_curve
import matplotlib.pyplot as plt


def find_best_threshold(model, X_val, y_val):
    y_probs = model.predict_proba(X_val)[:, 1]
    thresholds = np.linspace(0, 1, 100)
    f1_scores = []
    best_f1, best_threshold = 0, 0.5

    for t in thresholds:
        preds = (y_probs >= t).astype(int)
        score = f1_score(y_val, preds)
        f1_scores.append(score)
        if score > best_f1:
            best_f1 = score
            best_f1, best_threshold = score, t

    print(f"Best Threshold: {best_threshold:.4f}")
    print(f"Best F1 Score: {best_f1:.4f}")

    # 시각화
    plt.figure(figsize=(8,5))
    plt.plot(thresholds, f1_scores, color='darkgreen')
    plt.axvline(best_threshold, color='red', linestyle='--', label=f'Best: {best_threshold:.4f}')
    plt.xlabel("Threshold")
    plt.ylabel("F1-score")
    plt.title("Threshold vs F1-score")
    plt.grid(True)
    plt.legend()
    plt.show()

    return best_threshold