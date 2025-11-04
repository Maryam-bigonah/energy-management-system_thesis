# Complete LSTM Pipeline - Results Summary

## Overview

Complete pipeline for training an LSTM model to forecast next-hour building load for Torino building with 20 apartments.

## Pipeline Steps (2-9)

### Step 2-3: Prepare Features and Create LSTM Windows
**File:** `prepare_lstm_data.py`

**Functions:**
- `prepare_features(df)` - Calculates total_load and selects features
- `make_lstm_dataset(df, feature_cols, target_col, window=24, horizon=1)` - Creates 3D LSTM windows
- `prepare_lstm_data(df, window=24, horizon=1)` - Complete pipeline

**Features Used:**
1. `total_load` - Sum of all 20 apartment loads
2. `pv_1kw` - PV generation
3. `hour` - Hour of day (0-23)
4. `dayofweek` - Day of week (0=Monday, 6=Sunday)
5. `is_weekend` - Binary (0/1)
6. `season` - Season (0=winter, 1=spring, 2=summer, 3=autumn)

**Output:**
- `X`: shape `(n_samples, 24, 6)` - 3D array for LSTM
- `y`: shape `(n_samples, 1)` - next-hour total_load

---

### Step 4: Time-Based Train/Val/Test Split
**File:** `prepare_lstm_data.py`

**Function:** `split_train_val_test(X, y, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)`

**Split:**
- First 70% → train
- Next 15% → val
- Last 15% → test

**Output:**
- `X_train, X_val, X_test, y_train, y_val, y_test`

---

### Step 5: Scale Features
**File:** `prepare_lstm_data.py`

**Functions:**
- `scale_3d(X_train, X_val, X_test)` - Scales 3D feature arrays
- `scale_target(y_train, y_val, y_test)` - Scales target values (optional)

**Process:**
1. Reshape 3D → 2D: `(samples, 24, features)` → `(samples*24, features)`
2. Fit MinMaxScaler on training data only
3. Transform all sets
4. Reshape back to 3D: `(samples*24, features)` → `(samples, 24, features)`

**Output:**
- `X_train_scaled, X_val_scaled, X_test_scaled, scaler_X`
- `y_train_scaled, y_val_scaled, y_test_scaled, scaler_y` (optional)

---

### Step 6: Build LSTM Model
**File:** `build_lstm_model.py`

**Function:** `build_lstm_model(window_size=24, n_features=6, lstm_units=64, learning_rate=1e-3)`

**Architecture:**
- Input shape: `(24, n_features)`
- 1 LSTM layer: 64 units
- 1 Dense(1) output layer

**Compilation:**
- Optimizer: Adam(learning_rate=1e-3)
- Loss: MSE (Mean Squared Error)
- Metric: MAE (Mean Absolute Error)

**Output:**
- Compiled Keras Sequential model

---

### Step 7: Train Model
**File:** `train_lstm_model.py`

**Function:** `train_lstm_model(model, X_train, y_train, X_val, y_val, epochs=30, batch_size=32)`

**Configuration:**
- Epochs: 30
- Batch size: 32
- EarlyStopping: patience=5, restore_best_weights=True
- Monitor: val_loss

**Output:**
- Training history
- Trained model with best weights restored

---

### Step 8: Evaluate and Plot
**File:** `evaluate_lstm_model.py`

**Functions:**
- `evaluate_model(model, X_test, y_test, scaler_y=None)` - Evaluates on test set
- `plot_predictions(y_true, y_pred, n_hours=200)` - Plots true vs predicted
- `evaluate_and_plot(model, X_test, y_test, scaler_y=None, n_hours=200)` - Complete evaluation

**Metrics:**
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- MAPE (Mean Absolute Percentage Error)

**Plot:**
- True vs predicted load for last 200 hours of test set
- Shows MAE and RMSE on plot

**Output:**
- `mae, rmse, y_pred, y_true`

---

### Step 9: Save Model and Scalers
**File:** `save_load_lstm.py`

**Functions:**
- `save_lstm_model(model, scaler_X, scaler_y=None, ...)` - Saves model and scalers
- `load_lstm_model(model_path, scaler_X_path, scaler_y_path)` - Loads model and scalers

**Saved Files:**
- `lstm_building_load.h5` - Keras model
- `scaler_X.pkl` - Feature scaler (joblib)
- `scaler_y.pkl` - Target scaler (joblib, optional)

---

## Complete Usage Example

```python
import pandas as pd
from prepare_lstm_data import prepare_lstm_data, split_train_val_test, scale_3d, scale_target
from build_lstm_model import build_lstm_model
from train_lstm_model import train_lstm_model
from evaluate_lstm_model import evaluate_and_plot
from save_load_lstm import save_lstm_model

# Step 2-3: Prepare data
df_master = pd.read_csv('data/master_dataset_2024.csv', index_col=0, parse_dates=True)
X, y, feature_cols, df_features = prepare_lstm_data(df_master, window=24, horizon=1)

# Step 4: Split
X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(X, y)

# Step 5: Scale
X_train_scaled, X_val_scaled, X_test_scaled, scaler_X = scale_3d(X_train, X_val, X_test)
y_train_scaled, y_val_scaled, y_test_scaled, scaler_y = scale_target(y_train, y_val, y_test)

# Step 6: Build model
model = build_lstm_model(window_size=24, n_features=len(feature_cols), lstm_units=64)

# Step 7: Train
history, model = train_lstm_model(
    model,
    X_train_scaled, y_train_scaled,
    X_val_scaled, y_val_scaled,
    epochs=30,
    batch_size=32
)

# Step 8: Evaluate
mae, rmse, y_pred, y_true = evaluate_and_plot(
    model,
    X_test_scaled, y_test,
    scaler_y=scaler_y,
    n_hours=200
)

# Step 9: Save
save_lstm_model(
    model,
    scaler_X,
    scaler_y=scaler_y,
    model_path='lstm_building_load.h5',
    scaler_X_path='scaler_X.pkl',
    scaler_y_path='scaler_y.pkl'
)
```

---

## Forecasting New Data

After training and saving, you can load the model and forecast new 24h profiles:

```python
from save_load_lstm import load_lstm_model
import numpy as np

# Load model and scalers
model, scaler_X, scaler_y = load_lstm_model(
    model_path='lstm_building_load.h5',
    scaler_X_path='scaler_X.pkl',
    scaler_y_path='scaler_y.pkl'
)

# Prepare new 24h profile (from PVGIS and LPG)
# new_features shape: (24, 6) - last 24 hours of features
new_features = np.array([...])  # Your 24h data

# Scale features
new_features_scaled = scaler_X.transform(new_features).reshape(1, 24, 6)

# Forecast
y_pred_scaled = model.predict(new_features_scaled, verbose=0)

# Inverse transform if targets were scaled
if scaler_y is not None:
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
else:
    y_pred = y_pred_scaled

print(f"Predicted next-hour load: {y_pred[0, 0]:.2f} kW")
```

---

## Files Created

### Data Preparation
- `prepare_lstm_data.py` - Steps 2-5 (features, windows, split, scaling)

### Model Building
- `build_lstm_model.py` - Step 6 (model architecture)

### Training
- `train_lstm_model.py` - Step 7 (training with early stopping)

### Evaluation
- `evaluate_lstm_model.py` - Step 8 (evaluation and plotting)

### Saving/Loading
- `save_load_lstm.py` - Step 9 (save/load model and scalers)

### Complete Pipeline
- `run_complete_lstm_pipeline.py` - Runs all steps (2-9) in sequence

---

## Requirements

```bash
pip install pandas numpy tensorflow scikit-learn matplotlib joblib
```

---

## Expected Results

After running the pipeline, you should see:

1. **Data Preparation Summary:**
   - Total samples created
   - Feature ranges
   - Train/val/test split sizes

2. **Model Summary:**
   - Architecture details
   - Total parameters (~17,000 for 64 LSTM units)

3. **Training Progress:**
   - Training loss and MAE
   - Validation loss and MAE
   - Early stopping (if triggered)
   - Best epoch identified

4. **Evaluation Results:**
   - MAE on test set (e.g., ~2-5 kW)
   - RMSE on test set (e.g., ~3-6 kW)
   - Plot showing true vs predicted for last 200 hours

5. **Saved Files:**
   - `lstm_building_load.h5` (~1-2 MB)
   - `scaler_X.pkl` (~1-2 KB)
   - `scaler_y.pkl` (~1-2 KB)

---

## Next Steps

1. **Build master dataset** (if not done):
   ```bash
   python3 build_master_dataset_final.py
   ```

2. **Run complete pipeline**:
   ```bash
   python3 run_complete_lstm_pipeline.py
   ```

3. **Use model for forecasting** in optimization loop or other applications

---

## Notes

- All scalers are fitted only on training data to prevent data leakage
- Time-based split preserves temporal order (no shuffling)
- Early stopping prevents overfitting and saves best weights
- Model can be used for real-time forecasting with new PVGIS and LPG data

