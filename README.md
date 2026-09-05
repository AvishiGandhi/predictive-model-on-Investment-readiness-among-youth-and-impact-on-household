# predictive-model-on-Investment-readiness-among-youth-and-impact-on-household
.# train_model.py
# Usage:
#  - If you have a CSV of surveys named 'surveys.csv', place it in the same folder and run: python train_model.py
#  - Otherwise this script will create synthetic 100 rows and proceed.

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import os

np.random.seed(42)

# --- Helper: create synthetic data if surveys.csv not found ---
def make_synthetic(n=100):
    salaries = np.random.normal(50000, 30000, size=n).clip(5000, 300000).astype(int)
    ages = np.random.randint(20, 70, size=n)
    family_cond = np.random.choice(['single','married_no_kids','married_with_kids','retired'], size=n, p=[0.2,0.3,0.4,0.1])
    interest_rate = np.random.normal(6.0, 1.5, size=n).clip(0.1, 15.0)
    inflation = np.random.normal(4.5, 1.5, size=n).clip(0.0, 12.0)
    economic = np.random.choice(['boom','peak','recession','trough'], size=n, p=[0.25,0.15,0.4,0.2])
    # optional personal fields to improve model
    current_savings = (salaries * np.random.uniform(0.2, 4, size=n)).astype(int)
    months_emergency = np.random.randint(0, 12, size=n)
    risk_self = np.random.choice(['low','medium','high'], size=n, p=[0.3,0.5,0.2])
    df = pd.DataFrame({
        'salary': salaries,
        'age': ages,
        'family_condition': family_cond,
        'interest_rate': interest_rate,
        'inflation_rate': inflation,
        'economic_cycle': economic,
        'current_savings': current_savings,
        'months_emergency': months_emergency,
        'risk_self': risk_self
    })
    return df

# --- rule-based readiness calculation (fallback & for pseudo target) ---
def rule_based_score(row):
    # Normalize salary (log scale)
    sal_score = min(1.0, np.log1p(row['salary'])/12.0)  # 0..~1
    # savings buffer
    savings_score = min(1.0, row.get('months_emergency', 0)/6.0)
    # age factor (younger more risk tolerance)
    age = row['age']
    age_score = 1 - ((age - 30)/60)  # ~higher for younger; may be <0 or >1
    age_score = np.clip(age_score, 0, 1)
    # family condition
    fam_map = {'single':1.0, 'married_no_kids':0.9, 'married_with_kids':0.7, 'retired':0.5}
    fam_score = fam_map.get(row['family_condition'], 0.8)
    # macro factor (worse macro => reduce readiness)
    macro_penalty = 0.0
    # inflation and interest increase risk for some assets; penalize if high
    macro_penalty += 0.02 * max(0, row['inflation_rate'] - 4.0)
    macro_penalty += 0.01 * max(0, row['interest_rate'] - 5.0)
    econ_map = {'boom': 1.05, 'peak': 0.95, 'recession': 0.85, 'trough': 0.9}
    econ_factor = econ_map.get(row['economic_cycle'], 0.95)
    base = (0.45 * sal_score) + (0.25 * savings_score) + (0.10 * age_score) + (0.20 * fam_score)
    score = base * econ_factor - macro_penalty
    score = np.clip(score, 0, 1)
    return round(score * 100, 1)

# --- load or generate data ---
if os.path.exists('surveys.csv'):
    df = pd.read_csv('surveys.csv')
else:
    df = make_synthetic(100)
    df.to_csv('surveys_sample_generated.csv', index=False)
    print("No 'surveys.csv' found — synthetic sample saved as 'surveys_sample_generated.csv'.")

# If the dataset already contains a labeled 'IRS' column, we will use it.
if 'IRS' in df.columns:
    y = df['IRS']
else:
    # compute rule-based IRS to use as a pseudo-target (so model learns)
    df['IRS'] = df.apply(rule_based_score, axis=1)
    y = df['IRS']

# Features to use — include optional fields if present
features = ['salary','age','family_condition','interest_rate','inflation_rate','economic_cycle']
for opt in ['current_savings','months_emergency','risk_self']:
    if opt in df.columns:
        features.append(opt)

X = df[features].copy()

# Preprocessing pipelines
numeric_feats = X.select_dtypes(include=['int64','float64']).columns.tolist()
categorical_feats = X.select_dtypes(include=['object','category']).columns.tolist()

num_transform = Pipeline([
    ('scaler', StandardScaler())
])

cat_transform = Pipeline([
    ('ohe', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', num_transform, numeric_feats),
    ('cat', cat_transform, categorical_feats)
])

# Model pipeline
model = Pipeline([
    ('pre', preprocessor),
    ('rf', RandomForestRegressor(n_estimators=200, random_state=42))
])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)
preds = model.predict(X_test)
print("RMSE on test:", mean_squared_error(y_test, preds, squared=False))

# Predict full dataset
df['predicted_IRS'] = model.predict(X)

# Bucket readiness
def bucket(score):
    if score < 40: return 'Conservative'
    if score < 70: return 'Moderate'
    return 'Aggressive'

df['readiness_category'] = df['predicted_IRS'].apply(bucket)

# Suggest investments (simple rule)
def suggest_tools(row):
    cat = row['readiness_category']
    age = row['age']
    fam = row['family_condition']
    suggestions = []
    if cat == 'Conservative':
        suggestions = ['Fixed Deposits','PPF','Government Bonds','Gold','NPS']
    elif cat == 'Moderate':
        suggestions = ['Mutual Funds','ETFs','Gold','NPS','Government Bonds']
    else:
        suggestions = ['Stock Market','Mutual Funds','ETFs','Real Estate','Cryptocurrency']
    # Age/family tweaks
    if age < 30 and 'Stock Market' not in suggestions:
        suggestions = ['Mutual Funds','ETFs','Stock Market'] + suggestions
    if fam == 'retired':
        suggestions = ['Government Bonds','PPF','NPS','Gold']
    # remove duplicates while preserving order
    seen = set()
    out = []
    for s in suggestions:
        if s not in seen:
            out.append(s); seen.add(s)
    return out

df['suggested_investments'] = df.apply(suggest_tools, axis=1)

# Salary bifurcation calculation
def salary_split(row):
    base = {'expenses':0.50, 'savings':0.20, 'investment':0.20, 'emergency':0.10}
    score = row['predicted_IRS']
    cat = bucket(score)
    if cat == 'Moderate':
        base['investment'] += 0.05; base['savings'] -= 0.05
    elif cat == 'Aggressive':
        base['investment'] += 0.10; base['savings'] -= 0.10
    # family adjustment
    if row['family_condition'] == 'married_with_kids':
        base['expenses'] += 0.05; base['emergency'] += 0.05; base['investment'] -= 0.05
    if row['family_condition'] == 'retired':
        base['expenses'] += 0.05; base['investment'] -= 0.05
    # ensure no negative
    for k in base:
        base[k] = max(0, base[k])
    # normalize to 1
    s = sum(base.values())
    for k in base:
        base[k] = base[k]/s
    amounts = {k: round(row['salary'] * base[k], 2) for k in base}
    return amounts

splits = df.apply(salary_split, axis=1)
df['expense_amount'] = splits.apply(lambda x: x['expenses'])
df['savings_amount'] = splits.apply(lambda x: x['savings'])
df['investment_amount'] = splits.apply(lambda x: x['investment'])
df['emergency_amount'] = splits.apply(lambda x: x['emergency'])

# Export predictions for Power BI
outcols = ['salary','age','family_condition','interest_rate','inflation_rate','economic_cycle',
           'current_savings','months_emergency','risk_self',
           'predicted_IRS','readiness_category','suggested_investments',
           'expense_amount','savings_amount','investment_amount','emergency_amount']
available = [c for c in outcols if c in df.columns]
df[available].to_csv('predictions_for_powerbi.csv', index=False)
print("Wrote predictions_for_powerbi.csv")
