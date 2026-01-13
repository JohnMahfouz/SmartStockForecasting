import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import json


# Store-level data
df_store = pd.read_csv('C:\\Users\\John\\Desktop\\SmartStock_Forecasting\\data\\egyptian_sales_store_level.csv', parse_dates=['Date'])

# Product-level data
df_product = pd.read_csv('C:\\Users\\John\\Desktop\\SmartStock_Forecasting\\data\\egyptian_sales_product_level.csv', parse_dates=['Date'])

print(f"Store Data: {df_store.shape}")
print(f"Product Data: {df_product.shape}")


# Lag features (multiple time windows)
for lag in [1, 2, 3, 7, 14, 28]:
    df_store[f'Lag_{lag}'] = df_store['Revenue_EGP'].shift(lag)

# Rolling statistics
for window in [7, 14, 28]:
    df_store[f'Rolling_Mean_{window}'] = df_store['Revenue_EGP'].rolling(window).mean()
    df_store[f'Rolling_Std_{window}'] = df_store['Revenue_EGP'].rolling(window).std()

# Day-of-week average (historical performance)
df_store['DOW_Avg'] = df_store.groupby('DayOfWeek')['Revenue_EGP'].transform(
    lambda x: x.expanding().mean()
)

# Month average
df_store['Month_Avg'] = df_store.groupby('Month')['Revenue_EGP'].transform(
    lambda x: x.expanding().mean()
)

# Growth rate
df_store['Growth_Rate_7d'] = (df_store['Lag_1'] - df_store['Lag_7']) / df_store['Lag_7']
df_store['Growth_Rate_28d'] = (df_store['Lag_1'] - df_store['Lag_28']) / df_store['Lag_28']

# Remove rows with NaN (from lag/rolling features)
df_store = df_store.dropna().copy()


# Lag features by product
for lag in [1, 7, 14]:
    df_product[f'Lag_{lag}'] = df_product.groupby('Product_Category')['Units_Sold'].shift(lag)
    df_product[f'Revenue_Lag_{lag}'] = df_product.groupby('Product_Category')['Revenue_EGP'].shift(lag)

# Rolling statistics by product
for window in [7, 14]:
    df_product[f'Rolling_Mean_{window}'] = df_product.groupby('Product_Category')['Units_Sold'].transform(
        lambda x: x.rolling(window).mean()
    )
    df_product[f'Rolling_Std_{window}'] = df_product.groupby('Product_Category')['Units_Sold'].transform(
        lambda x: x.rolling(window).std()
    )

# Product-specific day-of-week patterns
df_product['Product_DOW_Avg'] = df_product.groupby(['Product_Category', 'DayOfWeek'])['Units_Sold'].transform(
    lambda x: x.expanding().mean()
)

# Save product category before one-hot encoding (for later evaluation)
df_product['Product_Category_Original'] = df_product['Product_Category'].copy()

# Product one-hot encoding
product_dummies = pd.get_dummies(df_product['Product_Category'], prefix='Product')
df_product = pd.concat([df_product, product_dummies], axis=1)

# Drop the original Product_Category column (XGBoost can't handle object dtype)
df_product = df_product.drop('Product_Category', axis=1)

df_product = df_product.dropna().copy()


split_date = '2025-11-01'  # More data for training

# Store-level split
train_store = df_store[df_store['Date'] < split_date].copy()
test_store = df_store[df_store['Date'] >= split_date].copy()

# Product-level split
train_product = df_product[df_product['Date'] < split_date].copy()
test_product = df_product[df_product['Date'] >= split_date].copy()

print(f"\n📅 Train period: {train_store['Date'].min()} to {train_store['Date'].max()}")
print(f"📅 Test period: {test_store['Date'].min()} to {test_store['Date'].max()}")
print(f"Train size: {len(train_store)} days, Test size: {len(test_store)} days")



store_features = [
    'Is_White_Friday', 'Is_Weekend', 'Is_Thursday',
    'Ramadan_Phase', 'Is_Ramadan', 'Is_Eid_Fitr', 'Is_Eid_Adha',
    'Is_Coptic_Christmas', 'Is_Payday', 'Is_Payday_Early', 'Is_Payday_Late',
    'Is_Summer', 'Is_Winter', 'Is_School_Season',
    'DayOfWeek', 'Month', 'Day',
    'Lag_1', 'Lag_2', 'Lag_3', 'Lag_7', 'Lag_14', 'Lag_28',
    'Rolling_Mean_7', 'Rolling_Mean_14', 'Rolling_Mean_28',
    'Rolling_Std_7', 'Rolling_Std_14', 'Rolling_Std_28',
    'DOW_Avg', 'Month_Avg',
    'Growth_Rate_7d', 'Growth_Rate_28d'
]

X_train_store = train_store[store_features]
y_train_store = train_store['Revenue_EGP']
X_test_store = test_store[store_features]
y_test_store = test_store['Revenue_EGP']

# XGBoost with better hyperparameters
model_store = xgb.XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.1,
    reg_alpha=0.05,
    reg_lambda=1.0,
    early_stopping_rounds=50,
    random_state=42
)

model_store.fit(
    X_train_store, 
    y_train_store,
    eval_set=[(X_test_store, y_test_store)],
    verbose=50
)

# Predictions
train_pred_store = model_store.predict(X_train_store)
test_pred_store = model_store.predict(X_test_store)

# Evaluation
train_rmse = np.sqrt(mean_squared_error(y_train_store, train_pred_store))
test_rmse = np.sqrt(mean_squared_error(y_test_store, test_pred_store))
test_mae = mean_absolute_error(y_test_store, test_pred_store)
test_r2 = r2_score(y_test_store, test_pred_store)
test_mape = np.mean(np.abs((y_test_store - test_pred_store) / y_test_store)) * 100

print(f"\n📊 STORE MODEL PERFORMANCE")
print(f"{'='*60}")
print(f"Train RMSE: ±{train_rmse:,.0f} EGP")
print(f"Test RMSE:  ±{test_rmse:,.0f} EGP")
print(f"Test MAE:   ±{test_mae:,.0f} EGP")
print(f"Test R²:    {test_r2:.4f}")
print(f"Test MAPE:  {test_mape:.2f}%")

# Feature importance
feature_importance_store = pd.DataFrame({
    'Feature': store_features,
    'Importance': model_store.feature_importances_
}).sort_values('Importance', ascending=False)

print(f"\n🔝 TOP 10 IMPORTANT FEATURES (Store Model)")
print(feature_importance_store.head(10).to_string(index=False))


product_dummy_cols = [col for col in df_product.columns 
                      if col.startswith('Product_') and col != 'Product_Category_Original']

product_features = [
    'Is_White_Friday', 'Is_Weekend', 'Is_Thursday',
    'Ramadan_Phase', 'Is_Ramadan', 'Is_Eid_Fitr', 'Is_Eid_Adha',
    'Is_Coptic_Christmas', 'Is_Payday',
    'Is_Summer', 'Is_Winter',
    'DayOfWeek', 'Month',
    'Lag_1', 'Lag_7', 'Lag_14',
    'Revenue_Lag_1', 'Revenue_Lag_7', 'Revenue_Lag_14',
    'Rolling_Mean_7', 'Rolling_Mean_14',
    'Rolling_Std_7', 'Rolling_Std_14',
    'Product_DOW_Avg'
] + product_dummy_cols

exclude_cols = ['Date', 'Product_Category_Original', 'Units_Sold', 'Revenue_EGP', 
                'Price_Per_Unit', 'Is_Open', 'Year', 'WeekOfYear', 'Day']

product_features = [f for f in product_features if f not in exclude_cols]


available_features = [f for f in product_features if f in train_product.columns]
missing_features = [f for f in product_features if f not in train_product.columns]

if missing_features:
    print(f"⚠️  Warning: {len(missing_features)} features not found in data:")
    for f in missing_features[:5]:  # Show first 5
        print(f"     - {f}")
    product_features = available_features

X_train_product = train_product[product_features].copy()
X_test_product = test_product[product_features].copy()

print(f"   Checking data types...")
dtype_counts = X_train_product.dtypes.value_counts()
print(f"   Data types: {dtype_counts.to_dict()}")

object_cols = X_train_product.select_dtypes(include=['object']).columns.tolist()

if object_cols:
    print(f"⚠️  Found {len(object_cols)} object-type columns:")
    for col in object_cols:
        print(f"     - {col}: {X_train_product[col].dtype}")
    print(f"   Removing object columns...")
    product_features = [f for f in product_features if f not in object_cols]
    X_train_product = train_product[product_features].copy()
    X_test_product = test_product[product_features].copy()

bool_cols = X_train_product.select_dtypes(include=['bool']).columns.tolist()
if bool_cols:
    print(f"   Converting {len(bool_cols)} boolean columns to int...")
    for col in bool_cols:
        X_train_product[col] = X_train_product[col].astype(np.int8)
        X_test_product[col] = X_test_product[col].astype(np.int8)

y_train_product = train_product['Units_Sold'].values
y_test_product = test_product['Units_Sold'].values

X_train_product = X_train_product.reset_index(drop=True)
X_test_product = X_test_product.reset_index(drop=True)

print(f"✅ Final product features: {len(product_features)} features")
print(f"   Product dummies: {len([c for c in product_features if c.startswith('Product_')])}")
print(f"   Training shape: {X_train_product.shape}")
print(f"   Test shape: {X_test_product.shape}")
print(f"   Dtypes: {X_train_product.dtypes.value_counts().to_dict()}")


X_train_product_np = X_train_product.values
X_test_product_np = X_test_product.values


print(f"   X_train type: {type(X_train_product_np)}, shape: {X_train_product_np.shape}")
print(f"   y_train type: {type(y_train_product)}, shape: {y_train_product.shape}")

model_product = xgb.XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=5,
    min_child_weight=2,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.1,
    reg_alpha=0.05,
    reg_lambda=1.0,
    early_stopping_rounds=50,
    random_state=42
)

model_product.fit(
    X_train_product_np, 
    y_train_product,
    eval_set=[(X_test_product_np, y_test_product)],
    verbose=50
)

# Predictions
train_pred_product = model_product.predict(X_train_product_np)
test_pred_product = model_product.predict(X_test_product_np)

# Evaluation
train_rmse_p = np.sqrt(mean_squared_error(y_train_product, train_pred_product))
test_rmse_p = np.sqrt(mean_squared_error(y_test_product, test_pred_product))
test_mae_p = mean_absolute_error(y_test_product, test_pred_product)
test_r2_p = r2_score(y_test_product, test_pred_product)
test_mape_p = np.mean(np.abs((y_test_product - test_pred_product) / (y_test_product + 1))) * 100

print(f"PRODUCT MODEL PERFORMANCE")
print(f"{'='*60}")
print(f"Train RMSE: ±{train_rmse_p:.2f} units")
print(f"Test RMSE:  ±{test_rmse_p:.2f} units")
print(f"Test MAE:   ±{test_mae_p:.2f} units")
print(f"Test R²:    {test_r2_p:.4f}")
print(f"Test MAPE:  {test_mape_p:.2f}%")

# Per-product performance
print(f"PER-PRODUCT PERFORMANCE")
print(f"{'='*60}")
test_product_eval = test_product.copy()
test_product_eval['Predicted'] = test_pred_product
test_product_eval['Error'] = test_product_eval['Predicted'] - test_product_eval['Units_Sold']
test_product_eval['Abs_Error'] = np.abs(test_product_eval['Error'])

for product in sorted(test_product_eval['Product_Category_Original'].unique()):
    product_data = test_product_eval[test_product_eval['Product_Category_Original'] == product]
    mae = product_data['Abs_Error'].mean()
    mape = (product_data['Abs_Error'] / (product_data['Units_Sold'] + 1)).mean() * 100
    print(f"{product:20s}: MAE = {mae:6.2f} units, MAPE = {mape:5.2f}%")





# Save XGBoost models
model_store.save_model('C:\\Users\\John\\Desktop\\SmartStock_Forecasting\\models\\smartstock_store_model.json')
model_product.save_model('C:\\Users\\John\\Desktop\\SmartStock_Forecasting\\models\\smartstock_product_model.json')

# Save feature lists
with open('C:\\Users\\John\\Desktop\\SmartStock_Forecasting\\models\\model_features.json', 'w') as f:
    json.dump({
        'store_features': store_features,
        'product_features': product_features
    }, f, indent=2)

# Save model performance metrics
metrics = {
    'store_model': {
        'train_rmse': float(train_rmse),
        'test_rmse': float(test_rmse),
        'test_mae': float(test_mae),
        'test_r2': float(test_r2),
        'test_mape': float(test_mape)
    },
    'product_model': {
        'train_rmse': float(train_rmse_p),
        'test_rmse': float(test_rmse_p),
        'test_mae': float(test_mae_p),
        'test_r2': float(test_r2_p),
        'test_mape': float(test_mape_p)
    }
}

with open('C:\\Users\\John\\Desktop\\SmartStock_Forecasting\\models\\model_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)


feature_importance_store.to_csv('C:\\Users\\John\\Desktop\\SmartStock_Forecasting\\models\\feature_importance_store.csv', index=False)

print("✅ Models saved:")
print("   - smartstock_store_model.json")
print("   - smartstock_product_model.json")
print("   - model_features.json")
print("   - model_metrics.json")
print("   - feature_importance_store.csv")


print(f"\n🌙 RAMADAN PREDICTION ACCURACY")
print(f"{'='*60}")

ramadan_test = test_store[test_store['Is_Ramadan'] == 1]
if len(ramadan_test) > 0:
    ramadan_idx = ramadan_test.index
    ramadan_pred = model_store.predict(test_store.loc[ramadan_idx, store_features])
    ramadan_actual = test_store.loc[ramadan_idx, 'Revenue_EGP']
    
    ramadan_mae = mean_absolute_error(ramadan_actual, ramadan_pred)
    ramadan_mape = np.mean(np.abs((ramadan_actual - ramadan_pred) / ramadan_actual)) * 100
    
    print(f"Ramadan Days in Test: {len(ramadan_test)}")
    print(f"Ramadan MAE: ±{ramadan_mae:,.0f} EGP")
    print(f"Ramadan MAPE: {ramadan_mape:.2f}%")
    
    # Compare to regular days
    regular_test = test_store[test_store['Is_Ramadan'] == 0]
    regular_idx = regular_test.index
    regular_pred = model_store.predict(test_store.loc[regular_idx, store_features])
    regular_actual = test_store.loc[regular_idx, 'Revenue_EGP']
    regular_mape = np.mean(np.abs((regular_actual - regular_pred) / regular_actual)) * 100
    
    print(f"Regular Days MAPE: {regular_mape:.2f}%")
    print(f"Ramadan Impact on Accuracy: {ramadan_mape - regular_mape:+.2f}% difference")
