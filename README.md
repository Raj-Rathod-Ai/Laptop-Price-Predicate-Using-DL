# Laptop Price Predictor AI (Deep Learning & Streamlit)

An intelligent deep learning web application that estimates laptop market valuations based on hardware specifications, component tiers, and historical retail market data. Built with an Artificial Neural Network (ANN) in TensorFlow/Keras and deployed with an interactive, modern Streamlit user interface.

---

## Key Highlights

- **Deep Neural Network Model**: Multi-layer feedforward neural network trained on over 1,500+ verified retail laptop records.
- **Modern Hardware Coverage**:
  - **Processors**: Intel 12th/13th/14th Gen, Intel Core Ultra (5/7/9), AMD Ryzen 3000 to 8000 series, and Apple Silicon (M1, M2, M3, Pro, Max).
  - **Graphics**: Nvidia GeForce RTX 2050, RTX 30-Series (3050, 3060, 3070, 3080), RTX 40-Series (4050 through 4090), Intel Arc / Iris Xe, AMD Radeon 600M/700M, and Apple M-Series GPUs.
  - **Displays & Form Factors**: OLED (2.8K, 3K, 4K), Retina displays, High-Refresh Gaming, 2-in-1 Convertibles, Ultrabooks, and Workstations.
- **Dual Currency Valuation**: Real-time price estimations in both Euros (€) and Indian Rupees (₹) with dynamic exchange rate adjustments.
- **Anti-Sleep / Inactivity Fix**: Built-in client-side keep-alive heartbeat that prevents free cloud hosts (Streamlit Cloud, Render, Hugging Face) from going to sleep during idle sessions.
- **Interactive Quick Presets**: Instant configuration loaders for popular gaming, creator, ultrabook, and budget categories.

---

## Project Structure

```text
├── laptop_price.csv          # Expanded retail laptop dataset (1,594 records)
├── DL_ANN.ipynb              # Complete model training, data preprocessing, and evaluation notebook
├── laptop_price_model.keras  # Trained TensorFlow/Keras ANN model
├── model_columns.pkl         # Aligned one-hot encoded feature column schema
├── dropdowns.pkl             # Dynamic categories for UI dropdowns and filters
├── scaler.pkl                # Pre-fitted StandardScaler for numerical features (Inches, RAM, Weight)
├── app.py                    # Streamlit web application frontend
├── brain.md                  # Comprehensive developer workflow and architectural memory
└── README.md                 # Project documentation and quickstart guide
```

---

## Technology Stack

- **Machine Learning & Deep Learning**: TensorFlow, Keras, Scikit-Learn
- **Data Engineering**: Pandas, NumPy
- **Frontend & Web Framework**: Streamlit, HTML5, Vanilla CSS
- **Model Storage**: Keras Native Format (.keras), Pickle (.pkl)

---

## Installation & Setup

### 1. Clone or Download Repository
Ensure you are in the project root directory:
```bash
cd DL
```

### 2. Install Required Dependencies
```bash
pip install tensorflow keras streamlit pandas numpy scikit-learn
```

### 3. Launch the Application
Start the Streamlit development server:
```bash
python -m streamlit run app.py
```

Open your browser and navigate to:
```text
http://localhost:8501
```

---

## Model Architecture & Training Summary

- **Input Dimensions**: 490+ one-hot encoded features and standard scaled numerical features.
- **Layers**:
  - Dense Layer (128 units, ReLU activation) + Dropout (0.2)
  - Dense Layer (128 units, ReLU activation) + Dropout (0.2)
  - Dense Layer (64 units, ReLU activation) + Dropout (0.2)
  - Dense Layer (32 units, ReLU activation) + Dropout (0.2)
  - Output Layer (1 unit, Linear activation)
- **Optimizer**: Adam
- **Loss Function**: Mean Squared Error (MSE)
- **Evaluation Metrics**:
  - $R^2$ Score: ~0.84 - 0.88
  - Mean Absolute Error (MAE): ~€190 - €200

---

## Usage Guide

1. **Select Brand & Form Factor**: Choose the manufacturer (e.g., Apple, Asus, Lenovo, HP, Dell) and category.
2. **Filter & Pick Components**: Select your desired Processor (CPU) and Graphics Card (GPU) using the sidebar hardware filters.
3. **Configure Memory & Storage**: Pick RAM capacity and SSD/HDD storage configuration.
4. **Choose Display & OS**: Select screen size, resolution, and operating system.
5. **Estimate Price**: Click **Calculate Estimated Market Value** to view valuation breakdown, currency equivalent, and estimated market range.
