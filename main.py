from app.loader import load_models
from app.utils import build_applicant_from_dict
from app.predict import two_stage_predict
import yaml

# load models
config = yaml.safe_load(open("config.yaml"))

cls, reg = load_models(config)

def run_cli():
    data = {
        'no_of_dependents' : int(input("Enter number of dependents: ")),
            'education' : input("Enter education level (Graduate/Not Graduate): "),
            'self_employed' : input("Are you self-employed? (Yes/No): "), 
            'income_annum' : float(input("Enter annual income: ")),
            'loan_amount' : float(input("Enter loan amount: ")),
            'loan_term' :  float(input("Enter loan term (in years, 2-20): ")),
            'cibil_score' : float(input("Enter CIBIL score: ")), 
            'residential_assets_value' :   float(input("Enter residential assets value: ")),
            'commercial_assets_value' :  float(input("Enter commercial assets value: ")), 
            'luxury_assets_value' :  float(input("Enter luxury assets value: ")), 
            'bank_asset_value' : float(input("Enter bank asset value: ")),  
    }

    if not 2 <= data['loan_term'] <= 20:
        raise ValueError("Loan term must be between 2 and 20 years")

    df = build_applicant_from_dict(data, list(cls.feature_names_in_))
    print(two_stage_predict(cls, reg, df))


if __name__ == "__main__":
    run_cli()