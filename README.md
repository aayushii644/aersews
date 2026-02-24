```markdown
# 🏗️ Concrete Compressive Strength Prediction

A complete Machine Learning pipeline for predicting **Concrete Compressive Strength (MPa)** using multiple regression models with hyperparameter tuning and model comparison.

This project uses:

- scikit-learn  
- XGBoost  
- pandas  
- matplotlib  

---

## 📌 Project Overview

The goal of this project is to predict the **Concrete Compressive Strength (MPa)** based on mixture components such as:

- Cement  
- Blast Furnace Slag  
- Fly Ash  
- Water  
- Superplasticizer  
- Coarse Aggregate  
- Fine Aggregate  
- Age  

The project includes:

- Data loading with validation
- Automatic target column detection
- Train/Test split
- Multiple regression models
- Hyperparameter tuning using GridSearchCV
- Model comparison (MAE, RMSE, R²)
- Feature importance visualization
- Prediction for new concrete samples

---

## 📂 Project Structure

```

.
├── Concrete_Data.xls
├── main.py
├── README.md
└── outputs/
└── feature_importance.png

```

---

## ⚙️ Installation

### 1️⃣ Place Dataset

Make sure the dataset file is in the project root directory:

```

Concrete_Data.xls

````

### 2️⃣ Install Dependencies

```bash
pip install pandas numpy scikit-learn xgboost matplotlib xlrd openpyxl
````

Notes:

* `xlrd` is required for `.xls` files
* `openpyxl` is required for `.xlsx` files

---

## ▶️ How to Run

### Default (Quick Verification Mode)

```bash
python main.py
```

By default, the script runs in **QUICK_RUN mode**, which:

* Uses smaller hyperparameter grids
* Runs faster
* Is useful for quick testing

---

### Full Hyperparameter Search Mode

To run full GridSearch (slower but more accurate tuning):

#### Linux / macOS:

```bash
export QUICK_RUN=0
python main.py
```

#### Windows (PowerShell):

```powershell
$env:QUICK_RUN=0
python main.py
```

---

## 🤖 Models Implemented

### 1️⃣ Linear Regression

* Standardized features using Pipeline
* Baseline model

### 2️⃣ Random Forest Regressor

* Hyperparameter tuning with GridSearchCV
* 5-fold cross-validation

### 3️⃣ XGBoost Regressor

* Gradient boosting algorithm
* Hyperparameter tuning
* Feature importance extraction

---

## 📊 Evaluation Metrics

Each model is evaluated using:

* **MAE** – Mean Absolute Error
* **RMSE** – Root Mean Squared Error
* **R² Score** – Coefficient of Determination

Typical expected performance:

| Model             | Approximate R² |
| ----------------- | -------------- |
| Linear Regression | ~0.60          |
| Random Forest     | ~0.85          |
| XGBoost           | ~0.88 – 0.92   |

---

## 📈 Output

After running the script:

* Model evaluation results are printed in the console
* Best hyperparameters are displayed
* Feature importance plot is saved to:

```
outputs/feature_importance.png
```

---

## 🔮 Example Prediction

The script includes a sample prediction:

```python
new_sample = [[540, 0, 0, 162, 2.5, 1040, 676, 28]]
```

Example output:

```
Predicted Concrete Strength (MPa): XX.XX
```

You can modify this sample to predict strength for any new mixture.

---

## 🧠 Key Features

* Robust dataset loading with fallback handling
* Automatic target column detection
* Configurable quick vs full hyperparameter search
* Clean modular structure
* Production-ready evaluation format
* Saves plot output non-interactively

---

## 🚀 Possible Improvements

* SHAP explainability analysis
* Model stacking (ensemble learning)
* Save best model using joblib
* REST API deployment (FastAPI or Flask)
* Docker containerization
* CI/CD integration

---

## 📜 License

This project is intended for educational and research purposes.

```
```
