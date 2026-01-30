import json
from flask import Flask, render_template, request, jsonify
import sys
import os
from datetime import datetime
from collections import Counter # Added for dashboard stats

app = Flask(__name__)

# Import the orchestrator module (assuming it's in the same directory)
try:
    from orchestrator import LoanScreeningOrchestrator
    ORCHESTRATOR_AVAILABLE = True
except ImportError:
    ORCHESTRATOR_AVAILABLE = False
    print("Warning: orchestrator.py not found. Running in demo mode.")

# Initialize orchestrator if available
orchestrator = None
if ORCHESTRATOR_AVAILABLE:
    try:
        orchestrator = LoanScreeningOrchestrator()
    except Exception as e:
        print(f"Error initializing orchestrator: {e}")
        ORCHESTRATOR_AVAILABLE = False

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'orchestrator_available': ORCHESTRATOR_AVAILABLE,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/evaluate', methods=['POST'])
def evaluate_application():
    """
    Evaluate a credit application
    """
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = [
            'applicant_name', 'loan_amount', 'annual_income', 
            'employment_status', 'credit_score'
        ]
        
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({
                'error': f'Missing required fields: {", ".join(missing_fields)}'
            }), 400
        
        # Process the application
        if ORCHESTRATOR_AVAILABLE and orchestrator:
            try:
                # Calculate DTI from form data
                annual_income = float(data.get("annual_income", 1))
                existing_debt = float(data.get("existing_debt", 0))
                loan_amount = float(data.get("loan_amount", 0))
                dti = (existing_debt / annual_income) * 100 if annual_income > 0 else 0
                
                # Map frontend data to agent pipeline expectations with complete fields
                agent_data = {
                    # Basic application info
                    "applicant_name": data.get("applicant_name"),
                    "loan_amount": loan_amount,
                    "annual_income": annual_income,
                    "loan_purpose": data.get("loan_purpose", "other"),
                    "id": f"web_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                    
                    # Employment info
                    "emp_title": data.get("employment_status", "Unknown"),
                    "emp_length": f"{data.get('years_employed', 0)} years",
                    "verified_income": "Verified" if data.get("employment_status") == "Employed" else "Not Verified",
                    
                    # Credit info (use credit_score to estimate credit behavior)
                    "credit_score": int(data.get("credit_score", 650)),
                    "debt_to_income": dti,
                    "state": "CA",  # Default state required by IngestionAgent
                    
                    # Default values for fields not in form
                    "earliest_credit_line": "2015-01-01",
                    "total_credit_lines": 8,
                    "open_credit_lines": 5,
                    "total_credit_limit": annual_income * 0.5,  # Estimate based on income
                    "total_credit_utilized": existing_debt,
                    "delinq_2y": 0,
                    "inquiries_last_12m": 2,
                    "num_historical_failed_to_pay": 0,
                    "months_since_90d_late": None,
                    "current_accounts_delinq": 0,
                    "total_collection_amount_ever": 0,
                    "accounts_opened_24m": 1,
                    "num_collections_last_12m": 0,
                    "months_since_last_credit_inquiry": 6,
                    "num_accounts_30d_past_due": 0,
                    "num_accounts_120d_past_due": 0,
                    "num_total_cc_accounts": 3,
                    "num_open_cc_accounts": 2,
                    "num_cc_carrying_balance": 1,
                    "num_mort_accounts": 0,
                    "current_installment_accounts": 1,
                    "term": 36,
                    "homeownership": "RENT",
                    "application_type": "Individual"
                }

                # Call the orchestrator to process the application
                result = orchestrator.screen_application(agent_data)
                
                # Extract risk score from result
                overall_risk = result.get("overall_risk_score", {})
                if isinstance(overall_risk, dict):
                    risk_score = overall_risk.get("overall_default_rate", 0.0)
                else:
                    risk_score = 0.0
                
                # Get decision (check both 'decision' and 'recommendation')
                decision = result.get("decision") or result.get("recommendation", "MANUAL_REVIEW")
                
                return jsonify({
                    'success': result.get("success", False),
                    'error': result.get("error"),  # Pass through pipeline error
                    'result': {
                        'decision': decision,
                        'risk_score': risk_score,
                        'confidence': result.get("confidence", 0),
                        'explanation': result.get("explanation", ""),
                        'similar_cases': result.get("similar_cases", []),
                        'anomalies': result.get("anomalies", []),
                        'mode': 'agent'
                    },
                    'timestamp': datetime.now().isoformat()
                })
            except Exception as e:
                import traceback
                print(traceback.format_exc())
                return jsonify({
                    'success': False,
                    'error': f'Server processing error: {str(e)}',
                    'timestamp': datetime.now().isoformat()
                }), 500
        else:
            # Demo mode response
            risk_score = calculate_demo_risk_score(data)
            decision = "Approved" if risk_score < 0.5 else "Rejected"
            
            return jsonify({
                'success': True,
                'result': {
                    'decision': decision,
                    'risk_score': risk_score,
                    'explanation': f"Demo mode: Based on preliminary analysis, the application is {decision.lower()}. "
                                 f"Credit score: {data['credit_score']}, "
                                 f"Income: ${data['annual_income']:,.2f}, "
                                 f"Loan amount: ${data['loan_amount']:,.2f}",
                    'similar_cases': [],
                    'mode': 'demo'
                },
                'timestamp': datetime.now().isoformat()
            })
            
    except Exception as e:
        return jsonify({
            'error': f'Server error: {str(e)}'
        }), 500

def calculate_demo_risk_score(data):
    """Calculate a simple demo risk score"""
    score = 0.5  # Base score
    
    # Adjust based on credit score
    credit_score = data.get('credit_score', 0)
    if credit_score >= 750:
        score -= 0.2
    elif credit_score < 600:
        score += 0.3
    
    # Adjust based on debt-to-income ratio
    loan_amount = data.get('loan_amount', 0)
    annual_income = data.get('annual_income', 1)
    existing_debt = data.get('existing_debt', 0)
    
    dti_ratio = (existing_debt + loan_amount) / (annual_income if annual_income > 0 else 1)
    if dti_ratio > 0.5:
        score += 0.2
    elif dti_ratio < 0.3:
        score -= 0.1
    
    # Adjust based on employment
    if data.get('employment_status') == 'Unemployed':
        score += 0.3
    
    return max(0, min(1, score))  # Clamp between 0 and 1

@app.route('/api/applications', methods=['GET'])
def get_applications():
    """Get recent application evaluations"""
    try:
        limit = request.args.get('limit', 50, type=int)
        
        if ORCHESTRATOR_AVAILABLE and orchestrator and orchestrator.pipeline.monitor:
            history = orchestrator.pipeline.monitor.get_decision_history(limit=limit)
            return jsonify({
                'success': True,
                'applications': history
            })
        else:
            return jsonify({
                'success': True,
                'applications': [],
                'message': 'No persistent storage available'
            })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/applications/<application_id>', methods=['GET'])
def get_application_detail(application_id):
    """Get detailed information for a specific application"""
    try:
        if ORCHESTRATOR_AVAILABLE and orchestrator and orchestrator.pipeline.monitor:
            # First search by original ID if it exists in metadata, or by decision_id
            history = orchestrator.pipeline.monitor.get_decision_history(limit=500)
            
            # Find the match
            case = None
            for entry in history:
                if entry.get('decision_id') == application_id:
                    case = entry
                    break
                    
            if not case:
                return jsonify({'success': False, 'error': 'Application not found'}), 404
                
            return jsonify({
                'success': True,
                'application': case
            })
        else:
            return jsonify({'success': False, 'error': 'No persistent storage available'}), 503
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/dashboard/stats', methods=['GET'])
def get_dashboard_stats():
    """Get aggregated statistics for the dashboard"""
    try:
        if ORCHESTRATOR_AVAILABLE and orchestrator and orchestrator.pipeline.monitor:
            history = orchestrator.pipeline.monitor.get_decision_history(limit=1000)
            
            if not history:
                return jsonify({
                    'success': True,
                    'stats': {
                        'total_count': 0,
                        'approval_rate': 0,
                        'avg_risk': 0,
                        'decision_dist': {'APPROVE': 0, 'REJECT': 0, 'MANUAL_REVIEW': 0},
                        'trends': []
                    }
                })
            
            # Aggregate stats
            decisions = [h['decision'] for h in history]
            risk_scores = [h['risk_score'] for h in history if h['risk_score'] is not None]
            
            counts = {
                'APPROVE': decisions.count('APPROVE'),
                'REJECT': decisions.count('REJECT'),
                'MANUAL_REVIEW': decisions.count('MANUAL_REVIEW')
            }
            
            total = len(decisions)
            approval_rate = (counts['APPROVE'] / total) * 100 if total > 0 else 0
            avg_risk = (sum(risk_scores) / len(risk_scores)) * 100 if risk_scores else 0
            
            # Daily trends (last 7 days)
            trends = []
            # Simplified trend grouping by date
            from collections import Counter
            date_counts = Counter(h['timestamp'].split('T')[0] for h in history)
            sorted_dates = sorted(date_counts.keys(), reverse=True)[:7]
            for date in reversed(sorted_dates):
                trends.append({
                    'date': date,
                    'count': date_counts[date]
                })
            
            return jsonify({
                'success': True,
                'stats': {
                    'total_count': total,
                    'approval_rate': round(approval_rate, 1),
                    'avg_risk': round(avg_risk, 1),
                    'decision_dist': counts,
                    'trends': trends,
                    'avg_latency': round(sum(h['latency_ms'] for h in history) / total, 0) if total > 0 else 0
                }
            })
        else:
            return jsonify({
                'success': False, 
                'error': 'No persistent storage available'
            }), 503
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
