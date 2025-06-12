from TN_2 import ThyroidCancer2
from sklearn.metrics import auc, roc_curve

import matplotlib.pyplot as plt

def plot_roc_auc(model, X_val, y_val):
    if model is None:
        print("Model is not trained.")
        return
    
    y_prob = model.predict_proba(X_val)[:, 1]
    fpr, tpr, _ = roc_curve(y_val, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    clf = ThyroidCancer2(
        train_path='/Users/yanghyejeong/Documents/Thyroid_Neoplasm/open/train.csv',
        test_path='/Users/yanghyejeong/Documents/Thyroid_Neoplasm/open/test.csv'
    )
    clf.load_data()
    clf.preprocess_data()
    clf.train_model()

    plot_roc_auc(clf.best_model, clf.X_val, clf.y_val)