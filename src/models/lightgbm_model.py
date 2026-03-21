import lightgbm as lgb
from sklearn.metrics import mean_absolute_error

def run_lightgbm(X_train, y_train, X_test, y_test):
    """
    Train a LightGBM regressor and return:
    (model, predictions, mae)
    """
    model = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=-1,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="regression",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    print(f"LightGBM MAE: {mae:.2f}")

    return model, preds, mae
