import sys
import os
import json

# Add parent directory to path to import local modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multi_agent_pipeline import MultiAgentPipeline

def test_dhia_case():
    print("STARTING TEST...")
    pipeline = MultiAgentPipeline()
    
    # The "dhia" case with correct payload keys for the pipeline
    dhia_application = {
        "applicant_name": "dhia",
        "loan_amount": 5000,
        "annual_income": 90000,
        "debt_to_income": 6.9, # Calculated as (1200+5000/12)/(90000/12)*100 ish
        "credit_score": 599,
        "years_employed": 3,
        "emp_title": "Architect", # Pipeline expects emp_title
        "homeownership": "OWN",
        "state": "NY",
        "loan_purpose": "major_purchase"
    }
    
    print("\n--- Running pipeline for 'dhia' ---")
    result = pipeline.process_application(dhia_application)
    
    # Write to file
    with open("scripts/dhia_verification_result.json", "w") as f:
        json.dump(result, f, indent=4)
    
    print(f"\nRESULT WRITTEN TO scripts/dhia_verification_result.json")
    print(f"DECISION: {result.get('decision')}")
    print(f"REASON: {result.get('decision_reasons', [])}")

if __name__ == "__main__":
    test_dhia_case()
