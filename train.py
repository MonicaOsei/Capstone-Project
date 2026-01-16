import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# -------------------------
# 1. Load your dataset
# -------------------------
df = pd.read_csv('_dataset.csv')  

# -------------------------
# 2. Drop unnecessary columns
# -------------------------
drop_cols = ['user_id','item_id','movie_id','title','timestamp','zip_code','release_date']
df = df.drop(columns=drop_cols, errors='ignore')

# -------------------------
# 3. Split dataset
# -------------------------
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)

# Reset index
df_train, df_val, df_test = df_train.reset_index(drop=True), df_val.reset_index(drop=True), df_test.reset_index(drop=True)

# Separate target
y_train = df_train['liked'].values
y_val = df_val['liked'].values
y_test = df_test['liked'].values

X_train = df_train.drop(columns=['liked'])
X_val = df_val.drop(columns=['liked'])
X_test = df_test.drop(columns=['liked'])

# -------------------------
# 4. Identify categorical and numeric features
# -------------------------
categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()
numeric_features = X_train.select_dtypes(exclude=['object']).columns.tolist()

# -------------------------
# 5. Preprocessing pipeline
# -------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_features),  # Random Forest does not need scaling
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

# -------------------------
# 6. Model + pipeline
# -------------------------
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=1))
])

# -------------------------
# 7. Hyperparameter tuning
# -------------------------
rf_params = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [None, 5, 10],
    'classifier__min_samples_split': [2, 5],
    'classifier__min_samples_leaf': [1, 2],
    'classifier__criterion': ['gini','entropy']
}

grid = GridSearchCV(
    pipeline,
    param_grid=rf_params,
    cv=3,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

# -------------------------
# 8. Train model
# -------------------------
grid.fit(X_train, y_train)

best_model = grid.best_estimator_
print("✅ Training complete")
print("Best Hyperparameters:", grid.best_params_)

# -------------------------
# 9. Evaluate model
# -------------------------
def evaluate_model(model, X_val, y_val, X_test, y_test):
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)
    
    metrics = {
        'Val Accuracy': accuracy_score(y_val, y_val_pred),
        'Val F1': f1_score(y_val, y_val_pred),
        'Test Accuracy': accuracy_score(y_test, y_test_pred),
        'Test F1': f1_score(y_test, y_test_pred)
    }
    return metrics

metrics = evaluate_model(best_model, X_val, y_val, X_test, y_test)
print("Validation Accuracy:", metrics['Val Accuracy'])
print("Validation F1:", metrics['Val F1'])
print("Test Accuracy:", metrics['Test Accuracy'])
print("Test F1:", metrics['Test F1'])

# -------------------------
# 10. Save the trained model
# -------------------------
with open('best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

print("✅ Best model saved to 'best_model.pkl'")
