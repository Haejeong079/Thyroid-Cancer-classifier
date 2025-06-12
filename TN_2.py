import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression

from imblearn.over_sampling import SMOTE
import shap

class ThyroidCancer2:
    def __init__(self, train_path, test_path=None):
        self.train_path = train_path
        self.test_path = test_path
        self.train_data = None
        self.test_data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_val = None
        self.y_val = None
        self.model = None
        self.final_model = None
        self.label_encoders = {}

    
    def load_data(self):
        self.train_data = pd.read_csv(self.train_path)
        if self.test_path:
            self.test_data = pd.read_csv(self.test_path)
        print("Data loaded successfully.")


    def preprocess_data(self):
        self.X = self.train_data.drop(columns=['ID', 'Cancer'])
        self.y = self.train_data['Cancer']

        if self.test_data is not None:
            self.X_test = self.test_data.drop(columns=['ID'])

        cat_cols = self.X.select_dtypes(include=['object']).columns
        for col in cat_cols:
            le = LabelEncoder()
            self.X[col] = le.fit_transform(self.X[col])
            self.label_encoders[col] = le

            if self.test_data is not None:
                for val in np.unique(self.X_test[col]):
                    if val not in le.classes_:
                        le.classes_ = np.append(le.classes_, val)
                self.X_test[col] = le.transform(self.X_test[col])

            
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.X, self.y, test_size= 0.2, random_state=42
        )
        print("Data preprocessed successfully.")

    def run_eda(self):
        # 타겟 레이블 붙여서 새로운 데이터프레임 생성
        df = self.X.copy()
        df['Cancer'] = self.y

        # 클래스 분포 시각화
        sns.countplot(x='Cancer', data=df)
        plt.title("Cancer Class Distribution")
        plt.show()

        # 상관관계 히트맵 시각화
        plt.figure(figsize=(10, 8))
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
        plt.title("Feature Correlation Heatmap")
        plt.show()

    def train_model(self):
        smote = SMOTE(random_state=42)
        X_train_s, y_train_s = smote.fit_resample(self.X_train, self.y_train)

        self.model = {
            'XGBoost' : XGBClassifier(random_state=42),
            'LightGBM' : LGBMClassifier(random_state=42),
            'CatBoost' : CatBoostClassifier(random_state=42, verbose=0),
            'LogisticRegression' : LogisticRegression(max_iter=1000, random_state=42)
        }

        # 모델별 F1-score 측정 및 최고 모델 선정
        best_f1 = 0
        for name, model in self.model.items():
            model.fit(X_train_s, y_train_s)
            y_pred = model.predict(self.X_val)
            f1 = f1_score(self.y_val, y_pred)
            print(f"{name} F1-score: {f1:.4f}")
            if f1 > best_f1:
                best_f1 = f1
                self.best_model = model
        print("✅ Best model selected.")


    def explain_with_shap(self):
        if self.best_model is None:
            print("No model to explain.")
            return
        
        # SHAP explainer를 생성하고 summary plot 그리기
        explainer = shap.Explainer(self.best_model)
        shap_values = explainer(self.X_val)
        shap.summary_plot(shap_values, self.X_val)

 

def main():
    clf = ThyroidCancer2(
        train_path = '/Users/yanghyejeong/Documents/Thyroid_Neoplasm/open/train.csv',
        test_path = '/Users/yanghyejeong/Documents/Thyroid_Neoplasm/open/test.csv'
    )

    clf.load_data()
    clf.preprocess_data()
    clf.run_eda()
    clf.train_model()      # 모델 학습 및 비교
    clf.explain_with_shap() # 모델 해석

if __name__ == "__main__":
    main()