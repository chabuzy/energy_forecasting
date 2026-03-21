
import xgboost as xgb
from sklearn.metrics import mean_absolute_error

def run_xgboost(X_train, y_train, X_test, y_test):
    """
    Train an XGBoost regressor and return:
    (model, predictions, mae)
    """
    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    print(f"XGBoost MAE: {mae:.2f}")

    return model, preds, mae
