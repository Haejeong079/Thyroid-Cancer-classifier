import numpy as np
from TN_Grid_Search import tune_catboost_with_grid
from TN_2 import ThyroidCancer2

from sklearn.metrics import f1_score, classification_report, precision_recall_curve
import matplotlib.pyplot as plt
import pandas as pd



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

def create_submission(model, test_data, 
                          submission_path, 
                          threshold = 0.5455,
                          output_path = 'submission.csv'):
        
    y_probs = model.predict_proba(test_data)[:, 1]
    pred = (y_probs >= threshold).astype(int)

    submission = pd.read_csv(submission_path)
    submission['Cancer'] = pred
    submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


def main():
    clf = ThyroidCancer2(
        train_path='/Users/yanghyejeong/Documents/Thyroid_Neoplasm/open/train.csv',
        test_path='/Users/yanghyejeong/Documents/Thyroid_Neoplasm/open/test.csv'
    )
    clf.load_data()
    clf.preprocess_data()

    #  Model Tuning (Catboost + Grid Search)
    best_model = tune_catboost_with_grid(clf.X_train, clf.y_train)

    # Threshold Tuning
    best_threshold = find_best_threshold(best_model, clf.X_val, clf.y_val)

    # Classification Report
    y_pred = (best_model.predict_proba(clf.X_val)[:, 1] >= best_threshold).astype(int)
    print("\n📊 Classification Report (Best Threshold)")
    print(classification_report(clf.y_val, y_pred))

    create_submission(
        model=best_model,
        test_data=clf.X_test,
        submission_path='/Users/yanghyejeong/Documents/Thyroid_Neoplasm/open/sample_submission.csv',
        threshold=best_threshold,
        output_path='submission_thresh.csv'
    )


if __name__ == '__main__':
    main()

    
