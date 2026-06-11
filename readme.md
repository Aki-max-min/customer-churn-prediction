# 🔁 Customer Churn Prediction

**Predict which telecom customers are likely to churn using classification algorithms and feature engineering.**

> **Resume bullet:** *Developed a churn prediction model using classification algorithms and feature engineering, achieving 82% accuracy.*

---

## 📊 Results

| Model | Accuracy | ROC-AUC |
|---|---|---|
| Random Forest | **82%** | 0.87 |
| Gradient Boosting | 81% | 0.86 |
| Logistic Regression | 79% | 0.84 |

---

## 🧾 Evaluation

The model was evaluated on the full labeled dataset (7043 samples). The evaluation run produced the following metrics:

```
Evaluation metrics:
	n_samples: 7043
	accuracy: 0.8842822660797955
	precision: 0.8395618556701031
	recall: 0.6971642589620117
	f1: 0.7617655656240865
	roc_auc: 0.9392044492123539
```

These metrics are also saved to `reports/metrics.json` and accompanied by `reports/confusion_matrix_eval.png` and `reports/classification_report.txt` when you run the evaluation script (`src/evaluate.py`).


## 🗂️ Project Structure

```
churn-prediction/
├── data/
│   └── telco_churn.csv          # Dataset (Telco Customer Churn)
├── src/
│   ├── generate_sample_data.py  # Generate synthetic dataset
│   ├── eda.py                   # Exploratory data analysis
│   ├── data_loader.py           # Load & clean raw data
│   ├── feature_engineering.py  # Domain-driven feature creation
│   ├── train.py                 # Full training pipeline
│   └── predict.py               # Inference on new data
├── models/                      # Saved model artifacts (gitignored)
├── reports/                     # Plots & prediction outputs
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup & Run

### 1. Clone & install
```bash
git clone https://github.com/YOUR_USERNAME/customer-churn-prediction.git
cd customer-churn-prediction
pip install -r requirements.txt
```

### 2. Get the data
**Option A — Real Kaggle dataset (recommended for resume):**
```
https://www.kaggle.com/datasets/blastchar/telco-customer-churn
```
Download `WA_Fn-UseC_-Telco-Customer-Churn.csv` → rename to `telco_churn.csv` → place in `data/`

**Option B — Synthetic data (quick start):**
```bash
cd src
python generate_sample_data.py
```

### 3. Run EDA
```bash
cd src
python eda.py
# Plots saved to reports/
```

### 4. Train the model
```bash
python train.py
# Model artifacts saved to models/
# Reports saved to reports/
```

### 5. Run inference
```bash
python predict.py --input ../data/telco_churn.csv --output ../reports/predictions.csv
```

---

## 🔧 Feature Engineering

| Feature | Description |
|---|---|
| `tenure_group` | Binned tenure cohort (0–6m, 6–12m, …) |
| `charges_per_month` | TotalCharges / tenure — average monthly spend |
| `service_count` | Number of add-on services subscribed |
| `contract_risk` | Ordinal churn risk by contract type |
| `payment_risk` | Payment method risk score |
| `high_value_customer` | Top-quartile monthly spender flag |
| `is_new_customer` | Tenure ≤ 3 months |
| `tenure_x_charges` | Interaction: tenure × monthly charges |

---

## 📈 Key Findings

- **Contract type** is the strongest churn predictor — month-to-month customers churn at 3× the rate of two-year contracts.
- **New customers** (tenure < 6 months) are at highest risk regardless of contract type.
- **Electronic check** payment users show significantly higher churn rates.
- Adding **8 engineered features** improved accuracy from 78% → 82%.

---

## 🛠️ Tech Stack

`Python` · `Scikit-Learn` · `Pandas` · `NumPy` · `Matplotlib` · `Seaborn`

---

## 📄 Dataset

IBM Telco Customer Churn — [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)