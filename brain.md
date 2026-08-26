# Project Brain & Technical Workflow Memory

This document stores the complete architecture, data engineering pipeline, training decisions, artifact mappings, and operational guidelines for the **Laptop Price Predictor AI** project. Read this file whenever resuming work on this repository to avoid starting from scratch.

---

## 1. Project Overview & Objective

- **Goal**: Predict laptop retail market prices in Euros (€) and convert dynamically to INR (₹) based on technical specifications and hardware component combinations.
- **Modeling Paradigm**: Feedforward Artificial Neural Network (ANN / Multi-Layer Perceptron) built in TensorFlow/Keras.
- **Inference Engine**: Streamlit application with client-side anti-sleep keep-alive heartbeat and dual-currency output.

---

## 2. Dataset Blueprint ([laptop_price.csv](file:///c:/DL/laptop_price.csv))

The dataset contains **1,594 records** with 13 features:

| Column Name | Data Type | Description & Example Values |
| :--- | :--- | :--- |
| `laptop_ID` | Integer | Unique identifier for each laptop entry |
| `Company` | String | Brand (Apple, Asus, Lenovo, HP, Dell, Acer, MSI, Razer, Microsoft, Samsung, LG, etc.) |
| `Product` | String | Model family name (MacBook Air M3, ROG Zephyrus G14, Victus 15, Legion Pro 5, etc.) |
| `TypeName` | String | Category (Ultrabook, Notebook, Gaming, 2 in 1 Convertible, Workstation, Netbook) |
| `Inches` | Float | Screen diagonal size (13.3, 14.0, 15.6, 16.0, 17.3) |
| `ScreenResolution` | String | Resolution (Full HD 1920x1080, 2.8K OLED 2880x1800, 4K UHD 3840x2160, Retina) |
| `Cpu` | String | Processor model and clock speed (Intel Core i7 13700H 2.4GHz, AMD Ryzen 7 7840HS 3.8GHz, Apple M3 4.05GHz) |
| `Ram` | String | Memory with 'GB' suffix during raw ingest (8GB, 16GB, 32GB, 64GB) |
| `Memory` | String | Storage drive config (512GB SSD, 1TB SSD, 256GB SSD + 1TB HDD, 128GB Flash Storage) |
| `Gpu` | String | Dedicated or integrated graphics (Nvidia GeForce RTX 4060, RTX 3050, Intel Arc, Apple M3 10-Core GPU) |
| `OpSys` | String | Operating System (Windows 11, Windows 10, macOS, Linux, Chrome OS, No OS) |
| `Weight` | String | Weight with 'kg' suffix (1.24kg, 2.20kg, 1.86kg) |
| `Price_euros` | Float | Target price in Euros (€) |

---

## 3. Data Preprocessing & Pipeline Rules

To ensure 100% feature alignment between training and deployment, follow these exact transformation steps:

### Step 1: Cleaning Numerical String Columns
```python
df['Ram'] = df['Ram'].astype(str).str.replace('GB', '').astype(int)
df['Weight'] = df['Weight'].astype(str).str.replace('kg', '').astype(float)
```

### Step 2: Categorical Dummy Encoding
```python
category = ['Company', 'TypeName', 'ScreenResolution', 'Cpu', 'Memory', 'Gpu', 'OpSys']
df_encoded = pd.get_dummies(df, columns=category, drop_first=True)
df_encoded = df_encoded.drop(columns=['laptop_ID', 'Product'])
df_encoded = df_encoded.astype(int)
```

### Step 3: Numerical Feature Scaling
```python
from sklearn.preprocessing import StandardScaler

numerical_feature = ['Inches', 'Ram', 'Weight']
scaler = StandardScaler()
df_encoded[numerical_feature] = scaler.fit_transform(df_encoded[numerical_feature])
```

---

## 4. Model Architecture & Hyperparameters

- **Framework**: `tensorflow.keras.models.Sequential`
- **Layers**:
  - `Dense(128, activation='relu', input_shape=(input_dim,))` + `Dropout(0.2)`
  - `Dense(128, activation='relu')` + `Dropout(0.2)`
  - `Dense(64, activation='relu')` + `Dropout(0.2)`
  - `Dense(32, activation='relu')` + `Dropout(0.2)`
  - `Dense(1, activation='linear')`
- **Compiler**:
  - Optimizer: `Adam`
  - Loss Function: `mean_squared_error`
  - Evaluation Metric: `mean_absolute_error`
- **Callbacks**:
  - `EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)`
- **Validation Split**: 20% validation split, 80% training split (`test_size=0.2, random_state=42`)

---

## 5. Artifact Directory & File Schema

When retraining or modifying the app, ensure all four artifacts are saved in the project root:

1. **`laptop_price_model.keras`**: Serialized Keras ANN model.
2. **`model_columns.pkl`**: Pickled list of column names (`X.columns.tolist()`) used during model training.
3. **`dropdowns.pkl`**: Pickled dictionary of all unique category options:
   - `Company`
   - `TypeName`
   - `Cpu_brand`
   - `Gpu_brand`
   - `OpSys`
   - `Ram`
   - `ScreenResolution`
   - `Memory`
4. **`scaler.pkl`**: Pickled `StandardScaler` instance fitted on `['Inches', 'Ram', 'Weight']`.

---

## 6. Streamlit Inference Execution Logic ([app.py](file:///c:/DL/app.py))

When performing a prediction from user inputs:
1. Initialize an empty zero-filled row matching `model_columns.pkl`:
   ```python
   encoded_row = pd.DataFrame(0, index=[0], columns=model_columns)
   ```
2. Transform numerical inputs with `scaler.pkl`:
   ```python
   num_df = pd.DataFrame([[inches, ram, weight]], columns=["Inches", "Ram", "Weight"])
   encoded_row[["Inches", "Ram", "Weight"]] = scaler.transform(num_df.astype(int))
   ```
3. Set one-hot flags for user-selected categorical features:
   ```python
   active_cols = [
       f"Company_{company}",
       f"TypeName_{type_name}",
       f"ScreenResolution_{screen_res}",
       f"Cpu_{cpu}",
       f"Memory_{memory}",
       f"Gpu_{gpu}",
       f"OpSys_{opsys}"
   ]
   for col in active_cols:
       if col in model_columns:
           encoded_row[col] = 1
   ```
4. Predict with `model.predict(encoded_row.values, verbose=0)[0][0]`.

---

## 7. Cloud Anti-Sleep Heartbeat Solution

Free hosting platforms (Streamlit Community Cloud, Render, Hugging Face) place applications into sleep/idle mode after ~10-15 minutes of inactivity.

- **Solution**: Embed an unobtrusive periodic client-side keep-alive heartbeat in `app.py`:
  ```python
  components.html(
      """
      <script>
          const pingIntervalMs = 45000;
          function sendKeepAlive() {
              try {
                  window.parent.postMessage({ type: 'streamlit:keepAlive', timestamp: Date.now() }, '*');
                  fetch(window.location.href, { method: 'HEAD', cache: 'no-cache', mode: 'no-cors' }).catch(() => {});
              } catch (e) {}
          }
          setInterval(sendKeepAlive, pingIntervalMs);
      </script>
      """,
      height=0,
      width=0
  )
  ```

---

## 8. How to Retrain / Add New Hardware in the Future

1. Add your new verified laptop records to [laptop_price.csv](file:///c:/DL/laptop_price.csv).
2. Open and run all cells in [DL_ANN.ipynb](file:///c:/DL/DL_ANN.ipynb) to re-generate the artifacts.
3. Restart the Streamlit application (`python -m streamlit run app.py`).
