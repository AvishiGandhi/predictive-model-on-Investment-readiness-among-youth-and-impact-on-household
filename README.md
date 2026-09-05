# predictive-model-on-Investment-readiness-among-youth-and-impact-on-household
.# train_model.py
# Usage:
#  - If you have a CSV of surveys named 'surveys.csv', place it in the same folder and run: python train_model.py
#  - Otherwise this script will create synthetic 100 rows and proceed.

# train_model_from_googleform.py
# Usage:
#  - Export your Google Form responses as CSV and save as 'surveys.csv' in this folder.
#  - Install requirements: pip install pandas numpy scikit-learn
#  - Run: python train_model_from_googleform.py
# Output:
#  - predictions_for_powerbi.csv (IRS, readiness_category, suggested investments, salary bifurcation)

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Config
INPUT_CSV = 'surveys.csv'
OUTPUT_CSV = 'predictions_for_powerbi.csv'
RANDOM_STATE = 42

# Tune these maps for your local currency/bands if needed
age_map = {
    '15 - 16': 15.5, '15-16': 15.5, '15 – 16': 15.5,
    '17 - 18': 17.5, '17-18': 17.5, '17 – 18': 17.5,
    '19 - 20': 19.5, '19-20': 19.5, '19 – 20': 19.5,
    '21 - 22': 21.5, '21-22': 21.5, '21 – 22': 21.5
}

# income_range_map — adjust numbers to match your form's currency scale if needed
income_range_map = {
    '<5,000': 3000, '<5000':3000,
    '5,000 - 10,000':7500, '5000-10000':7500,
    '10,000 - 15,000':12500,
    '15,000 - 20,000':17500, '15000 - 20,000':17500,
    '>20,000':25000, '20,000+':25000
}

# percent bands mapping used by your form (modify keys if your form text differs)
percent_band_map = {
    '<20%': 10, '<20': 10,
    '20% - 40%': 30, '20-40': 30,
    '40% - 60%': 50, '40-60': 50,
    '60% - 80%': 70, '60-80': 70,
    '80% - 100%': 90, '80-100': 90
}

family_discuss_map = {'always':1.0, 'frequently':0.8, 'rarely':0.3, 'never':0.0}

INVEST_TOOLS = ['Mutual Funds','Stock Market','Fixed Deposits','Gold','PPF','NPS','Government Bonds','ETFs','Real Estate','Cryptocurrency','Other']

# Helper: find a column by keywords (case-insensitive substring)
def find_col(df, keywords):
    for c in df.columns:
        lc = c.lower()
        for k in keywords:
            if k in lc:
                return c
    return None

# Parse helpers
def parse_age(v):
    if pd.isna(v): return np.nan
    return age_map.get(str(v).strip(), np.nan)

def parse_income_range(v):
    if pd.isna(v): return np.nan
    return income_range_map.get(str(v).strip(), np.nan)

def parse_percent_band(v):
    if pd.isna(v): return np.nan
    return percent_band_map.get(str(v).strip(), np.nan)

def parse_yesno(v):
    if pd.isna(v): return 0
    s = str(v).strip().lower()
    return 1 if s.startswith('y') else 0

def parse_invest_where_cell(cell):
    if pd.isna(cell) or str(cell).strip()=='':
        return set()
    items = [p.strip() for p in str(cell).replace(',', ';').split(';') if p.strip()]
    return set(items)

# Synthetic generator if surveys.csv not present (for demonstration)
def synthetic_sample(n=100):
    np.random.seed(0)
    ages = np.random.choice(list(age_map.keys()), n)
    income = np.random.choice(list(income_range_map.keys()), n)
    saves = np.random.choice(list(percent_band_map.keys()), n)
    invests = np.random.choice(['Yes','No'], n, p=[0.6,0.4])
    invest_where = np.random.choice(['Mutual Funds;Stock Market','Fixed Deposits','Gold','ETFs','Cryptocurrency','None'], n)
    discuss = np.random.choice(['Always','Frequently','Rarely','Never'], n)
    changed = np.random.choice(['Yes','No'], n, p=[0.4,0.6])
    df = pd.DataFrame({
        'Timestamp': pd.date_range('2026-01-01', periods=n),
        'Age': ages,
        'Range of income': income,
        'What percentage of your income do you save': saves,
        'Do you invest': invests,
        'If Yes, where do you currently invest your money?': invest_where,
        'What percentage of your income do you invest': np.random.choice(list(percent_band_map.keys()), n),
        'How often do you discuss money with your family': discuss,
        'Has your family changed its spending habit due to inflation?': changed
    })
    return df

# Load CSV
if os.path.exists(INPUT_CSV):
    df_raw = pd.read_csv(INPUT_CSV)
    print("Loaded", INPUT_CSV)
else:
    print("No surveys.csv found — generating surveys_sample.csv for demo. Replace with your export named surveys.csv.")
    df_raw = synthetic_sample(100)
    df_raw.to_csv('surveys_sample.csv', index=False)

# Auto-detect relevant columns (best-effort)
col_age = find_col(df_raw, ['age'])
col_income_range = find_col(df_raw, ['range of income','income range','range of income'])
col_save_pct = find_col(df_raw, ['percentage of your income do you save','what percentage of your income do you save','save'])
col_invest_flag = find_col(df_raw, ['do you invest','invest?','do you invest?'])
col_invest_where = find_col(df_raw, ['where do you currently invest','where do you invest','if yes where'])
col_invest_pct = find_col(df_raw, ['percentage of your income do you invest','what percentage of your income do you invest'])
col_family_discuss = find_col(df_raw, ['discuss money with your family','how often do you discuss money','discuss with family'])
col_family_changed = find_col(df_raw, ['family changed','changed spending','spending habit due to inflation'])
col_interest = find_col(df_raw, ['interest rate'])
col_inflation = find_col(df_raw, ['inflation'])

print("Detected columns (best-effort):")
for label, col in [('age',col_age),('income_range',col_income_range),('save_pct',col_save_pct),
                   ('invest_flag',col_invest_flag),('invest_where',col_invest_where),('invest_pct',col_invest_pct),
                   ('family_discuss',col_family_discuss),('family_changed',col_family_changed),
                   ('interest_rate',col_interest),('inflation_rate',col_inflation)]:
    print(" ", label, "->", col)

# Build working DataFrame W with normalized features
W = pd.DataFrame()
W['respondent_id'] = df_raw.index.astype(str)

# Age numeric
W['age'] = df_raw[col_age].apply(parse_age) if col_age else np.nan

# monthly_income proxy from income range column
if col_income_range:
    W['monthly_income'] = df_raw[col_income_range].apply(parse_income_range)
else:
    W['monthly_income'] = np.nan

# save pct
W['save_pct'] = df_raw[col_save_pct].apply(parse_percent_band) if col_save_pct else np.nan

# invest flag
W['invests_flag'] = df_raw[col_invest_flag].apply(parse_yesno) if col_invest_flag else 0

# invest where flags
for t in INVEST_TOOLS:
    W['invest_where_' + t.replace(' ','_')] = 0
if col_invest_where:
    for i, cell in df_raw[col_invest_where].fillna('').items():
        sel = parse_invest_where_cell(cell)
        for t in INVEST_TOOLS:
            if any(t.lower() == s.lower() or t.lower() in s.lower() for s in sel):
                W.at[i, 'invest_where_' + t.replace(' ','_')] = 1

# invest pct
W['invest_pct'] = df_raw[col_invest_pct].apply(parse_percent_band) if col_invest_pct else np.nan

# family discuss numeric proxy
if col_family_discuss:
    W['family_discuss'] = df_raw[col_family_discuss].fillna('').apply(lambda x: family_discuss_map.get(str(x).strip().lower(), np.nan))
else:
    W['family_discuss'] = np.nan

# family changed spending
if col_family_changed:
    W['family_changed_spending'] = df_raw[col_family_changed].apply(lambda x: 1 if str(x).strip().lower().startswith('y') else 0)
else:
    W['family_changed_spending'] = 0

# optional macros
W['interest_rate'] = pd.to_numeric(df_raw[col_interest], errors='coerce').fillna(5.0) if col_interest else 5.0
W['inflation_rate'] = pd.to_numeric(df_raw[col_inflation], errors='coerce').fillna(4.0) if col_inflation else 4.0

# Fill reasonable defaults if proxies missing
if W['monthly_income'].isna().all():
    W['monthly_income'] = 7500  # default proxy (change if you want)
W['save_pct'] = W['save_pct'].fillna(10)
W['invest_pct'] = W['invest_pct'].fillna(5)

# Derive expense % and amounts
W['expense_pct'] = 100 - W['save_pct'] - W['invest_pct']
W.loc[W['expense_pct'] < 5, 'expense_pct'] = 20
W['monthly_expenses'] = (W['expense_pct'] / 100.0) * W['monthly_income']

# family_influence_score: combine discussion frequency + changed spending
W['family_influence_score'] = W['family_discuss'].fillna(0) * 0.6 + W['family_changed_spending'] * 0.4
W['family_influence_score'] = W['family_influence_score'].clip(0,1)

# proxies for months emergency and disposable income
W['months_emergency_proxy'] = (W['save_pct'] / 100.0) * 3  # simple proxy
W['disposable_income'] = W['monthly_income'] - W['monthly_expenses']

# Rule-based IRS (0..100)
def compute_IRS(r):
    sal = np.log1p(max(1, r['monthly_income']))
    sal_score = np.clip((sal - 7)/3.5, 0, 1)
    buffer_score = np.clip(r['save_pct'] / 100.0, 0, 1)
    invest_flag = r.get('invests_flag', 0)
    invest_pct_norm = np.clip(r.get('invest_pct', 0) / 100.0, 0, 1)
    family = r.get('family_influence_score', 0)
    willingness = 0.5 * invest_pct_norm + 0.3 * invest_flag + 0.2 * family
    macro_penalty = 0.01 * max(0, r.get('inflation_rate', 4.0) - 4.0) + 0.005 * max(0, r.get('interest_rate', 5.0) - 5.0)
    base = (0.40 * sal_score) + (0.30 * buffer_score) + (0.20 * willingness) + (0.10 * family)
    score = base - macro_penalty
    return round(np.clip(score,0,1) * 100, 1)

W['IRS_target'] = W.apply(compute_IRS, axis=1)

# Train a RandomForest to learn IRS (so predictions are stable for later new responses)
feat_cols = ['monthly_income','save_pct','invest_pct','monthly_expenses','disposable_income','family_influence_score','interest_rate','inflation_rate']
# add any invest-where flags present
for t in INVEST_TOOLS:
    key = 'invest_where_' + t.replace(' ','_')
    if key in W.columns:
        feat_cols.append(key)

X = W[feat_cols].fillna(0)
y = W['IRS_target']

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('rf', RandomForestRegressor(n_estimators=120, random_state=RANDOM_STATE))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
pipeline.fit(X_train, y_train)
preds_test = pipeline.predict(X_test)
print("Test RMSE (IRS):", mean_squared_error(y_test, preds_test, squared=False))

# Predict IRS for all rows
W['predicted_IRS'] = pipeline.predict(X).round(1)
def bucket(score):
    if score < 40: return 'Conservative'
    if score < 70: return 'Moderate'
    return 'Aggressive'
W['readiness_category'] = W['predicted_IRS'].apply(bucket)

# Suggested investments mapping
def suggest(row):
    s = row['predicted_IRS']
    cat = bucket(s)
    if cat == 'Conservative':
        tools = ['Fixed Deposits','Government Bonds','PPF (if eligible)','Gold']
    elif cat == 'Moderate':
        tools = ['Mutual Funds (balanced)','ETFs','Debt Funds / FDs']
    else:
        tools = ['Equity Mutual Funds','ETFs','Stock Market (small)','Real Estate (long-term)','Cryptocurrency (small %)']
    # nudge diversified ETFs if family influence strong and family already invested in risky tools
    if row.get('family_influence_score',0) > 0.6 and (row.get('invest_where_Stock_Market',0) == 1 or row.get('invest_where_Mutual_Funds',0) == 1):
        if 'ETFs' not in tools:
            tools.insert(0,'ETFs')
    return '; '.join(tools)

W['suggested_investments'] = W.apply(suggest, axis=1)

# Salary bifurcation (monthly)
def bifurcate(row):
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
    m = row['monthly_income'] if row['monthly_income']>0 else 1
    return {
        'expense_amount': round(m * base['expenses'],2),
        'savings_amount': round(m * base['savings'],2),
        'investment_amount': round(m * base['investment'],2),
        'emergency_amount': round(m * base['emergency'],2)
    }

b = W.apply(bifurcate, axis=1, result_type='expand')
W = pd.concat([W, b], axis=1)

# Export results for Power BI
out_cols = ['respondent_id','age','monthly_income','save_pct','invest_pct','monthly_expenses','disposable_income',
            'predicted_IRS','readiness_category','suggested_investments','expense_amount','savings_amount','investment_amount','emergency_amount',
            'family_influence_score','interest_rate','inflation_rate']
W[out_cols].to_csv(OUTPUT_CSV, index=False)
print("Wrote", OUTPUT_CSV)
