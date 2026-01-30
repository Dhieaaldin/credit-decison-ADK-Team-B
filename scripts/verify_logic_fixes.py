
import sys
import os
import json
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multi_agent_pipeline import MultiAgentPipeline

def run_tests():
    print("STARTING STRATEGIC LOGIC VERIFICATION...")
    pipeline = MultiAgentPipeline()
    
    # CASE 1: The "Impossible Loan" (222% DTI)
    # Expected: Immediate REJECT via Safety Gate (Pre-search) -> Latency < 100ms
    case_impossible = {
        "applicant_name": "Impossible user",
        "loan_amount": 50000,
        "annual_income": 22500, # Monthly: 1875
        "existing_debt": 3500,  # Monthly debt > Income!
        "credit_score": 720,    # Good score shouldn't save this
        "years_employed": 5,
        "emp_title": "Retail",
        "homeownership": "RENT",
        "state": "NY",
        "loan_purpose": "credit_card"
    }
    
    print("\n--- TEST 1: IMPOSSIBLE LOAN (222% DTI) ---")
    start = time.time()
    res1 = pipeline.process_application(case_impossible)
    latency = (time.time() - start) * 1000
    
    print(f"Decision: {res1['recommendation']}")
    print(f"Reason: {res1['decision_reasons'][0]}")
    print(f"Latency: {latency:.1f}ms (Should be fast)")
    
    if res1['recommendation'] == 'REJECT' and "safety ceiling" in res1['decision_reasons'][0] and latency < 500:
        print("✅ PASS: Safety Gate works")
    else:
        print("❌ FAIL: Safety Gate missed or slow")

    # CASE 2: The "Borderline Winner" (599 Score, Low Risk)
    # Expected: APPROVE or soft MANUAL_REVIEW (Multiplier 1.5x shouldn't kill it)
    case_borderline = {
        "applicant_name": "Borderline User",
        "loan_amount": 5000,
        "annual_income": 90000,
        "existing_debt": 200,   # Very low DTI
        "credit_score": 599,    # Multiplier trigger
        "years_employed": 5,
        "emp_title": "Engineer",
        "homeownership": "OWN",
        "state": "NY",
        "loan_purpose": "home_improvement"
    }
    
    print("\n--- TEST 2: BORDERLINE WINNER (599 Score) ---")
    res2 = pipeline.process_application(case_borderline)
    print(f"Decision: {res2['recommendation']}")
    print(f"Reasons: {json.dumps(res2['decision_reasons'], indent=2)}")
    
    # CASE 3: Income Insufficiency (Loan > 1.5x Income)
    # Expected: REJECT (Pre-search)
    case_insufficient = {
        "applicant_name": "Dreamer",
        "loan_amount": 100000,
        "annual_income": 40000,
        "existing_debt": 0,
        "credit_score": 800,
        "years_employed": 5,
        "emp_title": "Artist",
        "homeownership": "RENT",
        "state": "NY",
        "loan_purpose": "major_purchase"
    }
    
    print("\n--- TEST 3: INCOME INSUFFICIENCY ---")
    res3 = pipeline.process_application(case_insufficient)
    print(f"Decision: {res3['recommendation']}")
    print(f"Reason: {res3['decision_reasons'][0]}")
    
    if res3['recommendation'] == 'REJECT' and "excessive compared to" in res3['decision_reasons'][0]:
        print("✅ PASS: Income Check works")
    else:
        print("❌ FAIL: Income Check missed")

if __name__ == "__main__":
    run_tests()
