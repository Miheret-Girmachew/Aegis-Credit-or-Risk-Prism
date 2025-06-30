# src/train.py

import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import logging

# Set up logging for clear output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def eval_metrics(actual, pred, pred_proba):
    """Calculates and returns a dictionary of classification metrics."""
    accuracy = accuracy_score(actual, pred)
    precision = precision_score(actual, pred, zero_division=0)
    recall = recall_score(actual, pred, zero_division=0)
    f1 = f1_score(actual, pred, zero_division=0)
    auc = roc_auc_score(actual, pred_proba)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "auc": auc
    }

def main():
    """Main function to train, evaluate, and register the best model."""
    
    
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    # ----------------------

    # --- 1. Load Data ---
    try:
        df = pd.read_csv('data/processed/processed_data.csv')
        logging.info("Processed data loaded successfully.")
    except FileNotFoundError:
        logging.error("Processed data not found. Run the data processing script first.")
        return

    # --- 2. Split Data ---
    X = df.drop('is_high_risk', axis=1)
    y = df['is_high_risk']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logging.info(f"Data split into training and testing sets. Test set size: {len(X_test)}")

    # Set an experiment name
    mlflow.set_experiment("Credit Risk Modeling")

    # --- 3. Train and Log Logistic Regression ---
    with mlflow.start_run(run_name="LogisticRegression"):
        mlflow.set_tag("model_type", "Logistic Regression")
        logging.info("Training Logistic Regression model...")
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('log_reg', LogisticRegression(random_state=42))
        ])
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
        
        metrics = eval_metrics(y_test, y_pred, y_pred_proba)
        logging.info(f"Logistic Regression Metrics: {metrics}")
        
        mlflow.log_params({'model_family': 'LogisticRegression'})
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(pipeline, "model")

    # --- 4. Train and Log Random Forest ---
    with mlflow.start_run(run_name="RandomForest"):
        mlflow.set_tag("model_type", "Random Forest")
        logging.info("Training Random Forest model...")
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
        
        metrics = eval_metrics(y_test, y_pred, y_pred_proba)
        logging.info(f"Random Forest Metrics: {metrics}")
        
        mlflow.log_params({'model_family': 'RandomForest', 'n_estimators': 100})
        mlflow.log_metrics(metrics)
        # Note: We just log the model artifact here. We register the best one later.
        mlflow.sklearn.log_model(pipeline, "model")

    # --- 5. Find the Best Model and Register It ---
    logging.info("Comparing models to find the best one based on AUC...")
    
    experiment_id = mlflow.get_experiment_by_name("Credit Risk Modeling").experiment_id
    runs = mlflow.search_runs(experiment_ids=[experiment_id], order_by=["metrics.auc DESC"], max_results=1)
    
    if runs.empty:
        logging.error("No runs found. Something went wrong with training.")
        return
        
    best_run_id = runs.iloc[0]['run_id']
    best_model_auc = runs.iloc[0]['metrics.auc']
    best_model_type = runs.iloc[0]['tags.model_type']
    
    logging.info(f"Best model is '{best_model_type}' with AUC: {best_model_auc:.4f} (Run ID: {best_run_id})")
    
    # Construct the model URI and register it
    model_uri = f"runs:/{best_run_id}/model"
    registered_model_name = "CreditRiskModel"
    
    logging.info(f"Registering the best model as '{registered_model_name}'")
    mlflow.register_model(model_uri=model_uri, name=registered_model_name)
    
    logging.info("Model training and registration complete.")


if __name__ == '__main__':
    main()