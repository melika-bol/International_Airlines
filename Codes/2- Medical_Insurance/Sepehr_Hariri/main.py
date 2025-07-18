import pandas as pd
from sklearn.model_selection import train_test_split
from scripts.preprocessing import preprocess_data
from scripts.modeling import build_model
from scripts.evaluation import evaluate_model
from scripts.save_model import save_model


from numpy import absolute
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import explained_variance_score

from sklearn.datasets import make_regression
from sklearn.preprocessing import PowerTransformer
from sklearn.linear_model import Ridge
from sklearn.discriminant_analysis import StandardScaler

# Load dataset
df = pd.read_csv('data/medical_insurance.csv')


# Clean outliers from dataset
Q1 = df['bmi'].quantile(0.25)  
Q3 = df['bmi'].quantile(0.75)  
IQR = Q3 - Q1 
df_clean = df[(df['bmi'] >= Q1 - 1.5*IQR) & (df['bmi'] <= Q3 + 1.5*IQR)]
outliers = df[(df['bmi'] < Q1 - 1.5*IQR) | (df['bmi'] > Q3 + 1.5*IQR)]
outliers['bmi'] = Q3
df_combined = pd.concat([df_clean, outliers])
df_combined.reset_index(drop=True, inplace=True)




# Create input & output 
X = df_combined.drop(['expenses', 'premium'], axis='columns')
y = df_combined[['premium']]



# Separate dataset into Train and Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Target Transformation ( log-transform )
y_train_log = np.log1p(y_train)


# Preprocessor
preprocessor = preprocess_data(df_combined)

# Model pipeline   
# select regression model :  xgb or rf or linear
model_type='rf'
model = build_model(model_type, preprocessor=preprocessor)



# Train
model_fit = model.fit(X_train, y_train_log)

y_train_pred = model_fit.predict(X_train)

# Train -- Reverse Target Transformation 
y_train_pred = np.expm1(y_train_pred)


# Predict
y_pred = model.predict(X_test)

# Reverse Target Transformation
y_pred = np.expm1(y_pred)

# Evaluate
evaluate_model(y_test, y_pred)



# Save model
save_model(model)

print(f"model_type: {model_type}")
print("Training complete and model saved.")

print("Train :")
print('score: %.7f '% model_fit.score(X_train, y_train))
print('RMSE: %.7f'% np.sqrt(mean_squared_error(y_train, y_train_pred)))
print('r2: %.7f'% r2_score(y_train, y_train_pred))
print('variance: %.7f'% explained_variance_score(y_train, y_train_pred))