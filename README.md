# COVID Risk Predictor

A Python project that merges multiple COVID-19 aggregated datasets with patient-level data to train a machine learning model predicting COVID-19 mortality risk.

## Repository Structure

```
covid_risk_python_repo/
├── LICENSE
├── README.md
├── covid_risk_predictor.py
├── .gitignore
└── data/
    ├── country_wise_latest.csv
    ├── covid_19_clean_complete.csv
    ├── day_wise.csv
    ├── full_grouped.csv
    ├── usa_county_wise.csv
    ├── worldometer_data.csv
    ├── united_states_covid19_deaths_ed_visits_and_positivity_by_state.csv
    ├── Weekly_Rates_of_Laboratory-Confirmed_COVID-19_Hospitalizations_from_the_COVID-NET_Surveillance_System_20250420.csv
    ├── Rates_of_Laboratory-Confirmed_RSV__COVID-19__and_Flu_Hospitalizations_from_the_RESP-NET_Surveillance_Systems_20250420.csv
    ├── data_table_for_weekly_deaths__the_united_states.csv
    ├── deaths_by_age_group.csv
    ├── deaths_by_race_ethnicity.csv
    └── deaths_by_sex.csv
```

## Setup

1. **Clone or download** this repository.
2. **Install dependencies**:
   ```bash
   pip install pandas scikit-learn joblib
   ```
3. **Train the model**:
   ```bash
   python covid_risk_predictor.py train --patient_csv path/to/your/patient.csv
   ```
4. **Predict risk**:
   ```bash
   python covid_risk_predictor.py predict --patient_csv path/to/your/new_patients.csv
   ```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
