print("Starting train_model.py")

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from preprocess import prepare_data  # make sure your preprocess.py has this function

print("Import successful")

def train_and_predict(train_path, test_path, submission_path):
    print("Loading and preprocessing data...")
    X_train, y_train, X_test, test_ids = prepare_data(train_path, test_path)
    
    print("Training Random Forest model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    print("Making predictions on test data...")
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
    print("Script finished successfully")
