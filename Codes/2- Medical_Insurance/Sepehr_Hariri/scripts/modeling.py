from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler




def build_model(model_type='linear', preprocessor=None):
    if model_type == 'linear':
        model = LinearRegression(n_jobs=-1)

    elif model_type == 'rf':
        model = RandomForestRegressor(n_estimators=200, max_depth=None, n_jobs=-1,
                                    min_samples_leaf=1, random_state=32)
    elif model_type == 'xgb':
        model = XGBRegressor(n_estimators=65, learning_rate=0.1)
        
    else:
        raise ValueError("Unsupported model type")

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])
    return pipeline