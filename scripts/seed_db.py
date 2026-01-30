import sqlite3
import json
import uuid
import random
from datetime import datetime, timedelta

DB_PATH = "decisions.db"

DECISIONS = ["APPROVE", "REJECT", "MANUAL_REVIEW"]
EMPLOYMENT_TITLES = ["Software Engineer", "Nurse", "Teacher", "Accountant", "Manager", "Sales Associate", "Driver"]
STATES = ["NY", "CA", "TX", "FL", "IL", "GA", "VA"]

def generate_fake_decision(day_offset):
    decision_id = str(uuid.uuid4())
    # Generate timestamp within the last 7 days
    ts = datetime.now() - timedelta(days=day_offset, hours=random.randint(0, 23), minutes=random.randint(0, 59))
    timestamp_str = ts.isoformat()
    
    decision = random.choice(DECISIONS)
    
    # Logic-based fake data
    if decision == "APPROVE":
        risk_score = random.uniform(0.05, 0.25)
        confidence = random.uniform(0.85, 0.98)
        explanation = "High credit score and stable income profile suggest low default probability."
    elif decision == "REJECT":
        risk_score = random.uniform(0.65, 0.95)
        confidence = random.uniform(0.90, 0.99)
        explanation = "High debt-to-income ratio and recent credit delinquencies indicate significant risk."
    else:
        risk_score = random.uniform(0.30, 0.60)
        confidence = random.uniform(0.70, 0.85)
        explanation = "Application shows mixed indicators; manual verification of employment is recommended."

    component_scores = {
        "credit_history": {"overall_default_rate": risk_score * random.uniform(0.8, 1.2)},
        "employment": {"overall_default_rate": risk_score * random.uniform(0.8, 1.2)},
        "financials": {"overall_default_rate": risk_score * random.uniform(0.8, 1.2)},
        "demographics": {"overall_default_rate": risk_score * random.uniform(0.8, 1.2)}
    }
    
    inputs = {
        "applicant_name": f"User_{random.randint(100, 999)}",
        "loan_amount": random.randint(5000, 50000),
        "annual_income": random.randint(40000, 150000),
        "state": random.choice(STATES),
        "employment_status": random.choice(EMPLOYMENT_TITLES),
        "credit_score": random.randint(600, 850)
    }

    return (
        decision_id,
        timestamp_str,
        "credit-v1",
        "1.0.0",
        "hash_" + str(random.randint(1000, 9999)),
        risk_score,
        decision,
        confidence,
        explanation,
        json.dumps(component_scores),
        json.dumps([str(uuid.uuid4()) for _ in range(2)]),
        random.uniform(1500, 4500), # latency_ms
        None, # pipeline_errors
        json.dumps(inputs)
    )

def seed():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    print(f"Seeding {DB_PATH}...")
    
    # Ensure table exists (though ModelMonitor should have created it)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS decision_log (
            decision_id TEXT PRIMARY KEY,
            timestamp TEXT NOT NULL,
            model_id TEXT NOT NULL,
            model_version TEXT,
            input_hash TEXT,
            risk_score REAL,
            decision TEXT,
            confidence REAL,
            explanation_summary TEXT,
            component_scores TEXT,
            similar_case_ids TEXT,
            latency_ms REAL,
            pipeline_errors TEXT,
            inputs_json TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Generate 30 records over 7 days
    records = []
    for _ in range(30):
        records.append(generate_fake_decision(random.randint(0, 6)))
    
    cursor.executemany("""
        INSERT OR REPLACE INTO decision_log 
        (decision_id, timestamp, model_id, model_version, input_hash, risk_score, decision, confidence, 
         explanation_summary, component_scores, similar_case_ids, latency_ms, pipeline_errors, inputs_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, records)
    
    conn.commit()
    conn.close()
    print(f"Successfully seeded {len(records)} records.")

if __name__ == "__main__":
    seed()
