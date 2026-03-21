import xgboost as xgb
from sklearn.model_selection import GridSearchCV

def tune_xgboost(X_train, y_train):
    model = xgb.XGBRegressor(objective="reg:squarederror", n_jobs=-1)
    param_grid = {
        "n_estimators": [200, 500],
        "max_depth": [4, 6],
        "learning_rate": [0.05, 0.1],
        "subsample": [0.8, 1.0],
    }
    grid = GridSearchCV(model, param_grid, cv=3, scoring="neg_mean_absolute_error", verbose=1)
    grid.fit(X_train, y_train)
    print("Best params:", grid.best_params_)
    return grid.best_estimator_
