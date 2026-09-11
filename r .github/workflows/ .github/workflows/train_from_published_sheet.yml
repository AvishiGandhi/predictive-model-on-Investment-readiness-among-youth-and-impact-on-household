#!/usr/bin/env python3
"""
train_model_from_googleform.py

Place this file in your repository root. The GitHub Actions workflow
will download the published Google Sheets CSV as 'surveys.csv' into the repo
before running this script.

What the script does:
- Loads surveys.csv (or surveys.xlsx if present)
- Auto-detects key form columns (age, income range, save/invest %) using fuzzy matching
- Creates numeric proxies, computes a rule-based IRS target
- Trains a RandomForestRegressor to learn IRS
- Produces predictions_for_powerbi.csv with predicted_IRS, readiness_category,
  suggested_investments, and salary bifurcation columns
- Saves the trained pipeline to models/irs_pipeline.joblib
"""

import os
import sys
import math
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib

# ---------------------------
# Configuration / mappings
# ---------------------------
# Adjust these maps if your form uses different exact label text
AGE_MAP = {
    '15 - 16': 15.5, '15-16': 15.5, '15 – 16': 15.5,
    '17 - 18': 17.5, '17-18': 17.5, '17 – 18': 17.5,
    '19 - 20': 19.5, '19-20': 19.5, '19 – 20': 19.5,
    '21 - 22': 21.5, '21-22': 21.5, '21 – 22': 21.5
}

INCOME_RANGE_MAP = {
    '<5,000': 3000, '<5000': 3000,
    '5,000 - 10,000': 7500, '5000-10000': 7500,
    '10,000 - 15,000': 12500,
    '15,000 - 20,000': 17500, '15000 - 20,000': 17500,
    '>20,000': 25000, '20,000+': 25000
}

PERCENT_BAND_MAP = {
    '<20%': 10, '<20': 10,
    '20% - 40%': 30, '20-40': 30,
    '40% - 60%': 50, '40-60': 50,
    '60% - 80%': 70, '60-80': 70,
    '80% - 100%': 90, '80-100': 90,
    '> 20%': 30, '20%': 20
}

FAMILY_DISCUSS_MAP = {'always': 1.0, 'frequently': 0.8, 'rarely': 0.3, 'never': 0.0}

INVEST_TOOLS = ['Mutual Funds','Stock Market','Fixed Deposits','Gold','PPF','NPS','Government Bonds','ETFs','Real Estate','Cryptocurrency','Other']

# ---------------------------
# Helpers
# ---------------------------
def find_col(df, keywords):
    """Return the first column name in df that contains any of the substrings in keywords (case-insensitive)."""
    if df is None: return None
    cols = df.columns.tolist()
    for c in cols:
        lc = c.lower()
        for k in keywords:
            if k in lc:
                return c
    return None

def safe_get(d, k, default=np.nan):
    return d[k] if k in d else default

def parse_age(v):
    if pd.isna(v): return np.nan
    s = str(v).strip()
    return AGE_MAP.get(s, None) if s in AGE_MAP else (float(s) if s.replace('.','',1).isdigit() else np.nan)

def parse_income_range(v):
    if pd.isna(v): return np.nan
    s = str(v).strip()
    return INCOME_RANGE_MAP.get(s, np.nan)

def parse_percent_band(v):
    if pd.isna(v): return np.nan
    s = str(v).strip()
    return PERCENT_BAND_MAP.get(s, np.nan)

def parse_yesno(v):
    if pd.isna(v): return 0
    s = str(v).strip().lower()
    return 1 if s.startswith('y') else 0

def parse_invest_where_cell(cell):
    if pd.isna(cell) or str(cell).strip() == '':
        return set()
    items = [p.strip() for p in str(cell).replace(',', ';').split(';') if p.strip()]
    return set(items)

# ---------------------------
# Load data
# ---------------------------
def load_survey():
    # Prefer surveys.csv, fallback to surveys.xlsx
    if os.path.exists('surveys.csv'):
        print("Loading surveys.csv")
        df = pd.read_csv('surveys.csv', dtype=str, keep_default_na=False, na_values=[''])
    elif os.path.exists('surveys.xlsx'):
        print("Loading surveys.xlsx")
        df = pd.read_excel('surveys.xlsx', dtype=str)
    else:
        raise FileNotFoundError("No surveys.csv or surveys.xlsx found in current directory.")
    # Normalize column names (strip)
    df.columns = [c.strip() for c in df.columns]
    return df

# ---------------------------
# Feature engineering & model
# ---------------------------
def build_features(df):
    W = pd.DataFrame()
    W['respondent_id'] = df.index.astype(str)

    # detect columns
    col_age = find_col(df, ['age'])
    col_income_range = find_col(df, ['range of income','income range','income'])
    col_save_pct = find_col(df, ['percentage of your income do you save','what percentage of your income do you save','save'])
    col_invest_flag = find_col(df, ['do you invest','invest?','do you invest?'])
    col_invest_where = find_col(df, ['where do you currently invest','where do you invest','if yes, where'])
    col_invest_pct = find_col(df, ['percentage of your income do you invest','what percentage of your income do you invest','percentage invest'])
    col_family_discuss = find_col(df, ['discuss money with your family','how often do you discuss money','discuss with family'])
    col_family_changed = find_col(df, ['family changed','changed spending','spending habit due to inflation'])
    col_interest = find_col(df, ['interest rate'])
    col_inflation = find_col(df, ['inflation'])

    print("Detected columns:")
    for k,v in [('age',col_age),('income_range',col_income_range),('save_pct',col_save_pct),
                ('invest_flag',col_invest_flag),('invest_where',col_invest_where),('invest_pct',col_invest_pct),
                ('family_discuss',col_family_discuss),('family_changed',col_family_changed),
                ('interest_rate',col_interest),('inflation_rate',col_inflation)]:
        print(" ", k, "->", v)

    # age numeric
    if col_age:
        W['age'] = df[col_age].apply(lambda x: parse_age(x))
    else:
        W['age'] = np.nan

    # monthly_income proxy from income range
    if col_income_range:
        W['monthly_income'] = df[col_income_range].apply(lambda x: parse_income_range(x))
    else:
        W['monthly_income'] = np.nan

    # save pct band
    if col_save_pct:
        W['save_pct'] = df[col_save_pct].apply(lambda x: parse_percent_band(x))
    else:
        W['save_pct'] = np.nan

    # invest flag
    if col_invest_flag:
        W['invests_flag'] = df[col_invest_flag].apply(lambda x: parse_yesno(x))
    else:
        W['invests_flag'] = 0

    # invest where flags
    for t in INVEST_TOOLS:
        W['invest_where_' + t.replace(' ','_')] = 0
    if col_invest_where:
        for i, cell in df[col_invest_where].fillna('').items():
            sel = parse_invest_where_cell(cell)
            for t in INVEST_TOOLS:
                if any(t.lower() == s.lower() or t.lower() in s.lower() for s in sel):
                    W.at[i, 'invest_where_' + t.replace(' ','_')] = 1

    # invest pct
    if col_invest_pct:
        W['invest_pct'] = df[col_invest_pct].apply(lambda x: parse_percent_band(x))
    else:
        W['invest_pct'] = np.nan

    # family discuss numeric proxy
    if col_family_discuss:
        W['family_discuss'] = df[col_family_discuss].fillna('').apply(lambda x: FAMILY_DISCUSS_MAP.get(str(x).strip().lower(), np.nan))
    else:
        W['family_discuss'] = np.nan

    # family changed spending
    if col_family_changed:
        W['family_changed_spending'] = df[col_family_changed].apply(lambda x: 1 if str(x).strip().lower().startswith('y') else 0)
    else:
        W['family_changed_spending'] = 0

    # macros
    W['interest_rate'] = pd.to_numeric(df[col_interest], errors='coerce').fillna(5.0) if col_interest else 5.0
    W['inflation_rate'] = pd.to_numeric(df[col_inflation], errors='coerce').fillna(4.0) if col_inflation else 4.0

    # fill defaults
    if W['monthly_income'].isna().all():
        W['monthly_income'] = 7500
    W['save_pct'] = W['save_pct'].fillna(10)
    W['invest_pct'] = W['invest_pct'].fillna(5)

    # derive expense % and amounts
    W['expense_pct'] = 100 - W['save_pct'] - W['invest_pct']
    W.loc[W['expense_pct'] < 5, 'expense_pct'] = 20
    W['monthly_expenses'] = (W['expense_pct'] / 100.0) * W['monthly_income']

    # family influence score
    W['family_influence_score'] = W['family_discuss'].fillna(0) * 0.6 + W['family_changed_spending'] * 0.4
    W['family_influence_score'] = W['family_influence_score'].clip(0,1)

    # proxies
    W['months_emergency_proxy'] = (W['save_pct'] / 100.0) * 3
    W['disposable_income'] = W['monthly_income'] - W['monthly_expenses']

    return W

def compute_IRS(r):
    # income score
    sal = math.log1p(max(1, float(r['monthly_income'])))
    sal_score = max(0.0, min(1.0, (sal - 7)/3.5))
    buffer_score = max(0.0, min(1.0, float(r['save_pct']) / 100.0))
    invest_flag = float(r.get('invests_flag', 0))
    invest_pct_norm = max(0.0, min(1.0, float(r.get('invest_pct', 0)) / 100.0))
    family = float(r.get('family_influence_score', 0))
    willingness = 0.5 * invest_pct_norm + 0.3 * invest_flag + 0.2 * family
    macro_penalty = 0.01 * max(0, float(r.get('inflation_rate', 4.0)) - 4.0) + 0.005 * max(0, float(r.get('interest_rate', 5.0)) - 5.0)
    base = (0.40 * sal_score) + (0.30 * buffer_score) + (0.20 * willingness) + (0.10 * family)
    score = base - macro_penalty
    score = max(0.0, min(1.0, score))
    return round(score * 100, 1)

def bucket(score):
    if score < 40: return 'Conservative'
    if score < 70: return 'Moderate'
    return 'Aggressive'

def suggest_tools(row):
    s = row['predicted_IRS']
    cat = bucket(s)
    age = row.get('age', 20)
    if cat == 'Conservative':
        tools = ['Fixed Deposits','Government Bonds','PPF (if eligible)','Gold']
    elif cat == 'Moderate':
        tools = ['Mutual Funds (balanced)','ETFs','Debt Funds / FDs']
    else:
        tools = ['Equity Mutual Funds','ETFs','Stock Market (small)','Real Estate (long-term)','Cryptocurrency (small %)']
    # family-influence nudge
    if row.get('family_influence_score',0) > 0.6 and (row.get('invest_where_Stock_Market',0) == 1 or row.get('invest_where_Mutual_Funds',0) == 1):
        if 'ETFs' not in tools:
            tools.insert(0, 'ETFs')
    return '; '.join(tools)

def salary_bifurcate(row):
    base = {'expenses':0.50, 'savings':0.20, 'investment':0.20, 'emergency':0.10}
    s = row['predicted_IRS']
    if s >= 70:
        base['investment'] += 0.10; base['savings'] -= 0.05; base['emergency'] -= 0.05
    elif s >= 40:
        base['investment'] += 0.05; base['savings'] -= 0.02; base['emergency'] -= 0.03
    if row.get('family_influence_score',0) > 0.6 and (row.get('invest_where_Stock_Market',0) == 1 or row.get('invest_where_Cryptocurrency',0) == 1):
        base['savings'] += 0.05; base['investment'] -= 0.03
    total = sum(max(0,v) for v in base.values())
    for k in base: base[k] = base[k] / total
    m = float(row['monthly_income']) if float(row['monthly_income'])>0 else 1.0
    return {
        'expense_amount': round(m * base['expenses'],2),
        'savings_amount': round(m * base['savings'],2),
        'investment_amount': round(m * base['investment'],2),
        'emergency_amount': round(m * base['emergency'],2)
    }

# ---------------------------
# Main
# ---------------------------
def main():
    try:
        df_raw = load_survey()
    except Exception as e:
        print("Error loading survey file:", e)
        sys.exit(1)

    W = build_features(df_raw)
    # compute IRS target using rule
    W['IRS_target'] = W.apply(compute_IRS, axis=1)

    # Features for model
    base_feats = ['monthly_income','save_pct','invest_pct','monthly_expenses','disposable_income','family_influence_score','interest_rate','inflation_rate']
    feat_cols = [c for c in base_feats if c in W.columns]
    # add invest-where flags if present
    for t in INVEST_TOOLS:
        key = 'invest_where_' + t.replace(' ','_')
        if key in W.columns:
            feat_cols.append(key)

    X = W[feat_cols].fillna(0)
    y = W['IRS_target']

    # Simple pipeline: scaler + RF
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestRegressor(n_estimators=150, random_state=42))
    ])

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)
    preds_test = pipeline.predict(X_test)
    rmse = math.sqrt(mean_squared_error(y_test, preds_test))
    print(f"Test RMSE (IRS): {rmse:.3f}")

    # Predict for all rows
    W['predicted_IRS'] = pipeline.predict(X).round(1)
    W['readiness_category'] = W['predicted_IRS'].apply(bucket)
    W['suggested_investments'] = W.apply(suggest_tools, axis=1)

    bif = W.apply(salary_bifurcate, axis=1, result_type='expand')
    W = pd.concat([W, bif], axis=1)

    # Prepare output
    out_cols = ['respondent_id','age','monthly_income','save_pct','invest_pct','monthly_expenses','disposable_income',
                'predicted_IRS','readiness_category','suggested_investments','expense_amount','savings_amount','investment_amount','emergency_amount',
                'family_influence_score','interest_rate','inflation_rate']
    # ensure columns exist
    out_cols = [c for c in out_cols if c in W.columns]
    df_out = W[out_cols].copy()
    df_out.to_csv('predictions_for_powerbi.csv', index=False)
    print("Wrote predictions_for_powerbi.csv (rows: {})".format(len(df_out)))

    # Save pipeline for future scoring
    os.makedirs('models', exist_ok=True)
    joblib.dump(pipeline, os.path.join('models','irs_pipeline.joblib'))
    print("Saved trained pipeline to models/irs_pipeline.joblib")

if __name__ == '__main__':
    main()
