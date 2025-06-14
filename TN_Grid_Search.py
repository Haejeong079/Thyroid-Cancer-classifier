from sklearn.model_selection import GridSearchCV
from catboost import CatBoostClassifier
from TN_2 import ThyroidCancer2

from sklearn.metrics import make_scorer, f1_score
from imblearn.over_sampling import SMOTE

import warnings
warnings.filterwarnings("ignore")


def tune_catboost_with_grid(X_train, y_train):
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    param_gird ={
        'depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'iterations': [200, 500]
    }

    scorer = make_scorer(f1_score)
    model = CatBoostClassifier(random_state=42, verbose=0)

    grid = GridSearchCV(
        estimator=model,
        param_grid=param_gird,
        scoring=scorer,
        cv=3,
        n_jobs=-1
    )

    grid.fit(X_res, y_res)
    print("✅ Grid Search 완료")
    print("Best Params:", grid.best_params_)
    print("Best F1 Score (CV):", grid.best_score_)

    return grid.best_estimator_


def main():
    clf = ThyroidCancer2(
        train_path ='/Users/yanghyejeong/Documents/Thyroid_Neoplasm/open/train.csv',
        test_path = '/Users/yanghyejeong/Documents/Thyroid_Neoplasm/open/test.csv'
    )
    clf.load_data()
    clf.preprocess_data()

    best_model = tune_catboost_with_grid(clf.X_train, clf.y_train)

    y_pred = best_model.predict(clf.X_val)
    val_f1 = f1_score(clf.y_val, y_pred)
    print(f"Validation F1 Score (best model): {val_f1:.4f}")

    # 필요 시 clf.best_model = best_model 로 저장도 가능


if __name__ == '__main__':
    main()