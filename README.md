# Loan Approval Project - Two Stage ML Application

## Overview :
Two stage model :
1. Classify applicant as Approved / Rejected
2. If Approved, predict the loan amount.

See [`notebooks/analyze.ipynb`](notebooks/analyze.ipynb) for EDA, model training and evaluation.

## Project structure
```
├── app/                  # loading models, input handling, prediction logic
├── data/                 # loan_approval_dataset.csv
├── models/               # trained stage 1 and stage 2 pipelines (.pkl)
├── notebooks/            # analyze.ipynb (EDA + training)
├── config.yaml           # model paths and UI default inputs
├── main.py               # command line app
└── streamlit_app.py      # web app
```

## Quickstart (local)
1. Create venv:
- uv venv
- uv pip install -r requirements.txt
2. Trained models are already included in `models\`. To retrain, run `notebooks/analyze.ipynb`.
3. Run locally :
- CLI : `uv run python main.py`
- Web app : `uv run streamlit run streamlit_app.py`

## Config
See `config.yaml` for runtime parameters (models paths and default UI inputs)

## NOTE :
- Make sure the version used to create the model is same as your local environment where you are testing the main.py or streamlit app.
- Loan term is in **years** (2-20), same as the dataset.
