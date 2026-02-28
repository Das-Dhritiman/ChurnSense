# Customer Churn Prediction System

A machine learning project that predicts whether a telecom customer will churn (leave the service), built with Python, scikit-learn, XGBoost, and deployed as an interactive Streamlit dashboard.

---

## Business Problem

Customer churn costs telecom companies **5–25x more** than retaining existing customers. This project builds a predictive model that identifies at-risk customers **before** they leave, enabling proactive retention strategies.

---

## Tech Stack

| Layer               | Tools                                      |
|---------------------|--------------------------------------------|
| Language            | Python 3.9+                                |
| Data Manipulation   | pandas, NumPy                              |
| Visualization       | matplotlib, seaborn, plotly                 |
| Machine Learning    | scikit-learn, XGBoost                       |
| Imbalance Handling  | imbalanced-learn (SMOTE)                    |
| Model Explainability| SHAP                                        |
| Web Application     | Streamlit                                   |
| Serialization       | joblib                                      |

---

## Project Structure

```
Customer Churn Prediction System/
│
├── data/
│   ├── raw data/
│   │   └── WA_Fn-UseC_-Telco-Customer-Churn.csv   # Real Telco churn dataset
│   └── processed data/
│       └── processed_data.csv                      # Cleaned & encoded data
│
├── notebooks/
│   ├── 01_eda.ipynb                # Exploratory Data Analysis & visualizations
│   └── 02_model_training.ipynb     # Model experimentation & comparison
│
├── src/
│   ├── __init__.py
│   ├── generate_data.py            # (Optional) Generate synthetic churn data
│   ├── data_preprocessing.py       # Data cleaning, encoding & feature engineering
│   └── train_model.py              # Train, evaluate & save the best model
│
├── models/
│   └── churn_model.pkl             # Saved best-performing model
│
├── app.py                          # Streamlit web application
├── requirements.txt                # Python dependencies
├── .gitignore
└── README.md
```

---

## Key Features

- **Real Dataset**: IBM Telco Customer Churn data (7,043 customers)
- **3 ML Models Compared**: Logistic Regression, Random Forest, XGBoost
- **SMOTE Oversampling**: Handles class imbalance (26.5% churn rate)
- **SHAP Explainability**: Understand *why* a customer is predicted to churn
- **Interactive Dashboard**: Streamlit app for real-time predictions & insights
- **Modular Codebase**: Clean, reusable Python modules

---

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train Model
```bash
python src/train_model.py
```

### 3. Launch the Dashboard
```bash
streamlit run app.py
```

---

## Model Performance

| Model                | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|----------------------|----------|-----------|--------|----------|---------|
| Logistic Regression  | 76.2%    | 54.1%     | 67.6%  | 60.1%    | 83.1%   |
| **Random Forest**    | **75.7%**| **52.9%** | **74.9%** | **62.0%** | **83.9%** |
| XGBoost              | 75.6%    | 53.4%     | 63.9%  | 58.2%    | 82.5%   |

*Trained on real Telco Customer Churn dataset (7,043 customers, 26.5% churn rate)*

---

## Business Insights

1. **Contract type** is the strongest churn predictor — month-to-month customers churn ~3x more
2. **Tenure < 12 months** customers are the highest risk segment
3. **Fiber optic** internet users churn more (likely due to pricing/competition)
4. Customers **without tech support or online security** are significantly more likely to leave
5. **Electronic check** payment method correlates with higher churn

---

## What I Learned

- End-to-end ML pipeline: data → EDA → preprocessing → modeling → deployment
- Handling imbalanced classification with SMOTE
- Model interpretability using SHAP values
- Building interactive ML dashboards with Streamlit
- Feature engineering and business-driven analysis

---

## License

This project is for educational and portfolio purposes.
