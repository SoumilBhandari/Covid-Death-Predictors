import os
import pandas as pd
import joblib
import argparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

DATA_DIR = "data"

def load_aggregated_data(data_dir):
    dfs = {
        "covid_complete": pd.read_csv(os.path.join(data_dir, "covid_19_clean_complete.csv"), parse_dates=["Date"]),
        "full_grouped": pd.read_csv(os.path.join(data_dir, "full_grouped.csv"), parse_dates=["Date"]),
        "day_wise": pd.read_csv(os.path.join(data_dir, "day_wise.csv"), parse_dates=["Date"]),
        "worldometer": pd.read_csv(os.path.join(data_dir, "worldometer_data.csv")),
        "country_latest": pd.read_csv(os.path.join(data_dir, "country_wise_latest.csv")),
        "state_weekly": pd.read_csv(os.path.join(data_dir, "united_states_covid19_deaths_ed_visits_and_positivity_by_state.csv")),
        "hosp_weekly": pd.read_csv(os.path.join(data_dir, "Weekly_Rates_of_Laboratory-Confirmed_COVID-19_Hospitalizations_from_the_COVID-NET_Surveillance_System_20250420.csv")),
        "rsv_weekly": pd.read_csv(os.path.join(data_dir, "Rates_of_Laboratory-Confirmed_RSV__COVID-19__and_Flu_Hospitalizations_from_the_RESP-NET_Surveillance_Systems_20250420.csv")),
        "weekly_deaths": pd.read_csv(os.path.join(data_dir, "data_table_for_weekly_deaths__the_united_states.csv")),
        "ages": pd.read_csv(os.path.join(data_dir, "deaths_by_age_group.csv")),
        "races": pd.read_csv(os.path.join(data_dir, "deaths_by_race_ethnicity.csv")),
        "sexes": pd.read_csv(os.path.join(data_dir, "deaths_by_sex.csv"))
    }
    return dfs

def merge_features(patient_df, dfs):
    df = patient_df.copy()
    for key in ["day_wise", "full_grouped", "covid_complete"]:
        df = df.merge(dfs[key], on="Date", how="left", suffixes=("", f"_{key}"))
    df = df.merge(dfs["state_weekly"], left_on="State", right_on="State", how="left")
    df = df.merge(dfs["hosp_weekly"], left_on="Date", right_on="Week_Ending", how="left", suffixes=("", "_hosp"))
    df = df.merge(dfs["rsv_weekly"], left_on="Date", right_on="Week_Ending", how="left", suffixes=("", "_rsv"))
    df = df.merge(dfs["ages"], left_on="AgeGroup", right_on="AgeGroup", how="left")
    df = df.merge(dfs["races"], left_on="Race", right_on="Race", how="left")
    df = df.merge(dfs["sexes"], left_on="Sex", right_on="Sex", how="left")
    us_snapshot = dfs["worldometer"].loc[dfs["worldometer"]["Country/Region"] == "US"]
    if not us_snapshot.empty:
        for col in ["Population", "TotalCases", "Deaths/1M pop", "Tests/1M pop"]:
            df[col] = us_snapshot.iloc[0][col]
    return df

def build_and_train(patient_csv, model_path="covid_pipeline.joblib"):
    patient_df = pd.read_csv(patient_csv, parse_dates=["Date"])
    dfs = load_aggregated_data(DATA_DIR)
    df = merge_features(patient_df, dfs)
    target = "Outcome"
    X = df.drop(columns=[target])
    y = df[target]
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ])
    pipeline = Pipeline([
        ("prep", preprocessor),
        ("clf", RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    print(classification_report(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_proba))
    joblib.dump(pipeline, model_path)
    print(f"Model pipeline saved to {model_path}")

def predict_patient(patient_csv, model_path="covid_pipeline.joblib"):
    pipeline = joblib.load(model_path)
    patient_df = pd.read_csv(patient_csv, parse_dates=["Date"])
    dfs = load_aggregated_data(DATA_DIR)
    df = merge_features(patient_df, dfs)
    X_new = df.drop(columns=["Outcome"])
    preds = pipeline.predict(X_new)
    probas = pipeline.predict_proba(X_new)[:, 1]
    results = pd.DataFrame({"Prediction": preds, "Probability": probas}, index=patient_df.index)
    print(results)
    return results

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3 or sys.argv[1] not in ["train", "predict"]:
        print("Usage: python covid_risk_predictor.py [train|predict] <patient_csv>")
        sys.exit(1)
    mode = sys.argv[1]
    patient_csv = sys.argv[2]
    if mode == "train":
        build_and_train(patient_csv)
    else:
        predict_patient(patient_csv)
