# src/train_model.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from preprocess import prepare_data

def train_and_predict(train_path, test_path, submission_path):
    X_train, y_train, X_test, test_ids = prepare_data(train_path, test_path)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    
    submission = pd.DataFrame({
        'PassengerId': test_ids,
        'Survived': predictions
    })
    submission.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")

if __name__ == "__main__":
    train_path = "../data/train.csv"
    test_path = "../data/test.csv"
    submission_path = "../submission/submission.csv"
    train_and_predict(train_path, test_path, submission_path)
