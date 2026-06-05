## Installation

Clone the repository and install the required dependencies:

```bash
git clone <your-repository-url>
cd AZENTRIX-FULLSTACK-TASK2
pip install -r requirements.txt
```

If your environment uses the provided typo file, this also works:

```bash
pip install -r requriments.txt
```

---

## Train the Model

Run the training pipeline using the Telco Customer Churn dataset:

```bash
python train.py --data-path "WA_Fn-UseC_-Telco-Customer-Churn.csv"
```

This will:

* Perform EDA
* Apply feature engineering
* Handle class imbalance using SMOTE
* Train and compare multiple models
* Perform hyperparameter tuning
* Save the best model

Generated artifacts:

```text
artifacts/
├── telco_churn_model.joblib
├── eda_summary.md
└── model_comparison.csv
```

---

## Run Predictions (CLI)

Interactive prediction mode:

```bash
python app.py
```

JSON prediction mode:

```bash
python app.py --input-json '{"gender":"Male","SeniorCitizen":0,"Partner":"No","Dependents":"No","tenure":1,"PhoneService":"Yes","MultipleLines":"No","InternetService":"Fiber optic","OnlineSecurity":"No","OnlineBackup":"No","DeviceProtection":"No","TechSupport":"No","StreamingTV":"No","StreamingMovies":"No","Contract":"Month-to-month","PaperlessBilling":"Yes","PaymentMethod":"Electronic check","MonthlyCharges":70.0,"TotalCharges":70.0}'
```

---

## Run Streamlit Web Application

Launch the Streamlit UI:

```bash
streamlit run streamlit_app.py
```

If Streamlit is not recognized:

```bash
python -m streamlit run streamlit_app.py
```

The application will open automatically in your browser at:

```text
http://localhost:8501
```

---

## Project Structure

```text
AZENTRIX-FULLSTACK-TASK2/
│
├── train.py
├── app.py
├── streamlit_app.py
├── churn_pipeline.py
├── requirements.txt
├── REPORT.md
├── README.md
├── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
└── artifacts/
    ├── telco_churn_model.joblib
    ├── eda_summary.md
    └── model_comparison.csv
```
