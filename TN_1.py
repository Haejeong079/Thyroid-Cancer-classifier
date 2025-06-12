import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from imblearn.over_sampling import SMOTE

class ThyroidCancer:
    def __init__ (self, train_path, test_path = None):
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

    def load_data(self):
        self.train_data = pd.read_csv(self.train_path)
        print("Train data loaded successfully.")

        if self.test_path:
            self.test_data = pd.read_csv(self.test_path)
            print("Test data loaded successfully.")
    
    def preprocess_data(self):
        self.X = self.train_data.drop(columns=['ID', 'Cancer'])
        self.y = self.train_data['Cancer']

        if self.test_data is not None:
            self.X_test = self.test_data.drop(columns=['ID'])

            categorical_features = [col for col in self.X.columns if self.X[col].dtype == 'object']
            for col in categorical_features:
                le  = LabelEncoder()
                self.X[col] = le.fit_transform(self.X[col])
                for val in np.unique(self.X_test[col]):
                    if val not in le.classes_:
                        le.classes_ = np.append(le.classes_, val)
                self.X_test[col] = le.transform(self.X_test[col])

    def split_data(self):
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        print("Train/Validation split 완료")


    def train_and_eval(self, X_tr, y_tr, X_val, y_val, label):
        model = XGBClassifier(random_state=42)
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_val)
        f1 = f1_score(y_val, y_pred)
        print(f"[{label}] Validation F1-score: {f1:.4f}")
        return model, f1
    
    def train_final_model(self):
        # (1) SMOTE 적용
        model_raw, f1_raw = self.train_and_eval(self.X_train, self.y_train, self.X_val, self.y_val, "Raw")

        # (2) SMOTE 적용
        smote = SMOTE(random_state=42)
        X_train_smote, y_train_smote = smote.fit_resample(self.X_train, self.y_train)
        model_smote, f1_smote = self.train_and_eval(X_train_smote, y_train_smote, self.X_val, self.y_val, "SMOTE")


        # 비교 후 최종 학습
        if f1_smote >= f1_raw:
            X_final, y_final = SMOTE(random_state=42).fit_resample(self.X, self.y)
            self.final_model = XGBClassifier(random_state=42)
            self.final_model.fit(X_final, y_final)
            print("✅ 최종 모델: SMOTE 적용")
        else:
            self.final_model = XGBClassifier(random_state=42)
            self.final_model.fit(self.X, self.y)
            print("✅ 최종 모델: SMOTE 미적용")

    def predict_and_submit(self, submission_path, output_path='baseline_submission.csv'):
        if self.final_model is None:
            raise ValueError("Final model has not been trained yet.")

        if self.X_test is None:
            raise ValueError("Test data has not been loaded.")
        
        final_pred = self.final_model.predict(self.X_test)

        submission = pd.read_csv(submission_path)
        submission['Cancer'] = final_pred
        submission.to_csv(output_path, index=False)
        print(f"Predictions saved to {output_path}")


    
def main():
    train_path = '/Users/yanghyejeong/Documents/Thyroid_Neoplasm/open/train.csv'
    test_path = '/Users/yanghyejeong/Documents/Thyroid_Neoplasm/open/test.csv'
    submission_path = '/Users/yanghyejeong/Documents/Thyroid_Neoplasm/open/sample_submission.csv'

    clf = ThyroidCancer(train_path, test_path)
    clf.load_data()
    clf.preprocess_data()
    clf.split_data()
    clf.train_final_model()
    clf.predict_and_submit(submission_path)

if __name__ == "__main__":
    main()
    print("ThyroidCancer model training completed.")