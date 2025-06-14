from TN_2 import ThyroidCancer2
from TN_Grid_Search import tune_catboost_with_grid

from sklearn.metrics import f1_score, classification_report, precision_recall_curve
import matplotlib.pyplot as plt
import numpy as np


#  Find the best threshold Tuning
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
    
def main():
    clf = ThyroidCancer2(
        train_path='/Users/yanghyejeong/Documents/Thyroid_Neoplasm/open/train.csv',
        test_path='/Users/yanghyejeong/Documents/Thyroid_Neoplasm/open/test.csv'
    )
    clf.load_data()
    clf.preprocess_data()

    best_model = tune_catboost_with_grid(clf.X_train, clf.y_train)

    # Threshold Tuning
    best_threshold = find_best_threshold(best_model, clf.X_val, clf.y_val)

    # pred & eval
    y_probs = best_model.predict_proba(clf.X_val)[:, 1]
    y_pred = (y_probs >= best_threshold).astype(int)
    print("\n📊 Classification Report (Best Threshold)")
    print(classification_report(clf.y_val, y_pred))

if __name__ == '__main__':
    main()
