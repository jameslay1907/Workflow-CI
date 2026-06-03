import mlflow
import pandas as pd
from xgboost import XGBClassifier
import numpy as np
import sys
import warnings
from pathlib import Path

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(42)

    datasetPath = sys.argv[4] if len(sys.argv) > 4 else 'ai_student_impact_preprocessing'

    BASE_DIR = Path(__file__).resolve().parent

    def returndataset(datasetname):
        dataset = pd.read_csv(f'{BASE_DIR}/{datasetPath}/{datasetname}.csv')
        dataset = dataset.to_numpy()
        return dataset

    X_train = returndataset('X_train')
    X_test = returndataset('X_test')
    y_train = np.ravel(returndataset('y_train'))
    y_test = np.ravel(returndataset('y_test'))

    input_example = X_train[0:5]
    n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    learning_rate = float(sys.argv[3]) if len(sys.argv) > 3 else 0.05

    with mlflow.start_run():
        model = XGBClassifier(
                objective='multi:softprob',
                num_class=3,
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                random_state=42
        )

        model.fit(X_train, y_train)

        predicted_data = model.predict(X_test)

        mlflow.sklearn.log_model(sk_model=model, artifact_path="model", input_example=input_example)

        # Log Metrics
        accuracy = model.score(X_test, y_test)
        mlflow.log_metric("accuracy", accuracy)

