from TN_2 import ThyroidCancer2
from TN_Grid_Search import tune_catboost_with_grid

import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score

def plot_precision_recall(model, X_val, y_val):
    if model is None:
        print("Model is not trained.")
        return
    
    y_scores = model.predict_proba(X_val)[:, 1]
    precision, recall, _ = precision_recall_curve(y_val, y_scores)
    average_precision = average_precision_score(y_val, y_scores)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='purple', lw=2,
             label=f'PRC (AP = {average_precision:.4f})')
    plt.xlabel('Recall (민감도)')
    plt.ylabel('Precision (정밀도)')
    plt.title('Precision-Recall Curve')
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.show()


def main():
    clf = ThyroidCancer2(
        train_path='/Users/yanghyejeong/Documents/Thyroid_Neoplasm/open/train.csv',
        test_path = '/Users/yanghyejeong/Documents/Thyroid_Neoplasm/open/test.csv'
    )
    clf.load_data()
    clf.preprocess_data()

    # Grid Search로 CatBoost 모델 튜닝
    best_model = tune_catboost_with_grid(clf.X_train, clf.y_train)

    # PRC 시각화
    plot_precision_recall(best_model, clf.X_val, clf.y_val)

if __name__ == '__main__':
    main()