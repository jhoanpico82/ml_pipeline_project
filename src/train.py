# src/train.py

import mlflow
import mlflow.sklearn
import yaml

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from mlflow.models.signature import infer_signature

from preprocess import (
    load_data,
    split_features_target,
    get_preprocessor
)


def load_config():
    """
    Carga el archivo de configuración YAML.
    """
    with open("src/config.yaml", "r") as file:
        config = yaml.safe_load(file)
    return config


def main():
    config = load_config()

    mlflow.set_tracking_uri(config['paths']['mlflow_tracking_uri'])
    mlflow.set_experiment(config['paths']['experiment_name'])

    print("📊 Cargando y preprocesando datos...")
    data = load_data(config['paths']['data_path'])
    X, y = split_features_target(data)

    preprocessor = get_preprocessor()
    X_processed = preprocessor.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y,
        test_size=0.2,
        random_state=config['hyperparameters']['random_state']
    )

    n_estimators = config['hyperparameters']['n_estimators']
    max_depth = config['hyperparameters']['max_depth']
    random_state = config['hyperparameters']['random_state']

    print("🎯 Iniciando entrenamiento...")
    with mlflow.start_run():
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("random_state", random_state)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2", r2)

        signature = infer_signature(X_train, model.predict(X_train))
        input_example = X_train[:5]

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=input_example,
            registered_model_name="VideoGameSalesModel"
        )

        print(f"✅ Entrenamiento completo. MSE: {mse:.4f} | R2: {r2:.4f}")
        print("📁 Modelo y métricas guardadas en MLflow.")


if __name__ == "__main__":
    print("🚀 Ejecutando script de entrenamiento...")
    main()



