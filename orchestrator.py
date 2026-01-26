"""
Credit Decision ADK Orchestrator

Standalone orchestrator script that manages the multi-agent pipeline for loan screening.
Provides a clean command-line interface and handles the complete workflow.

Usage:
    # Screen a new loan application
    python orchestrator.py --screen <application_data.json>
    
    # Load dataset into Qdrant
    python orchestrator.py --load-data <loans_full_schema.csv>
    
    # Run interactive mode
    python orchestrator.py --interactive
"""

import json
import sys
import os
import argparse
from typing import Dict, Any, Optional
from pathlib import Path
import time

# Fix Unicode encoding for Windows PowerShell
if os.name == 'nt':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from multi_agent_pipeline import MultiAgentPipeline


class LoanScreeningOrchestrator:
    """
    Orchestrates the loan screening process using the multi-agent pipeline.
    """
    
    def __init__(self):
        """Initialize the orchestrator with the multi-agent pipeline."""
        print("🚀 Initializing Credit Decision ADK Orchestrator...")
        self.pipeline = MultiAgentPipeline()
        print("✓ Pipeline initialized successfully\n")
    
    def screen_application(self, application_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Screen a single loan application.
        
        Args:
            application_data: Loan application data dictionary
            
        Returns:
            Screening result with decision and explanation
        """
        print("\n" + "="*70)
        print("📋 SCREENING NEW LOAN APPLICATION")
        print("="*70)
        
        print(f"\n📊 Application Details:")
        print(f"   • Loan Amount: ${application_data.get('loan_amount', 0):,.2f}")
        print(f"   • Annual Income: ${application_data.get('annual_income', 0):,.2f}")
        print(f"   • Employment: {application_data.get('emp_title', 'Unknown')}")
        print(f"   • State: {application_data.get('state', 'Unknown')}")
        
        # Run through pipeline
        print("\n⏳ Processing through agent pipeline...")
        start_time = time.time()
        
        result = self.pipeline.process_application(application_data)
        
        elapsed_time = time.time() - start_time
        
        # Display results
        print("\n" + "="*70)
        print("📈 SCREENING RESULTS")
        print("="*70)
        
        if result.get("success"):
            decision = result.get("decision", "UNKNOWN")
            risk_score = result.get("risk_score", 0)
            
            # Color-code decision
            decision_symbol = "✅" if decision == "APPROVE" else "❌" if decision == "REJECT" else "⚠️"
            
            print(f"\n{decision_symbol} DECISION: {decision}")
            print(f"📊 RISK SCORE: {risk_score:.2%}")
            
            # Similar cases
            similar_cases = result.get("similar_cases", [])
            if similar_cases:
                print(f"\n🔍 SIMILAR HISTORICAL CASES: {len(similar_cases)}")
                for i, case in enumerate(similar_cases[:3], 1):
                    print(f"   {i}. Loan ID: {case.get('id', 'N/A')} | "
                          f"Amount: ${case.get('amount', 0):,.0f} | "
                          f"Similarity: {case.get('similarity', 0):.2%}")
            
            # Explanation
            explanation = result.get("explanation", "")
            if explanation:
                print(f"\n💡 EXPLANATION:")
                print(f"   {explanation}")
            
            # Risk factors
            risk_factors = result.get("risk_factors", [])
            if risk_factors:
                print(f"\n⚠️  RISK FACTORS:")
                for factor in risk_factors[:5]:
                    print(f"   • {factor}")
            
            print(f"\n⏱️  Processing Time: {elapsed_time:.2f}s")
            print("="*70 + "\n")
            
            return result
        else:
            print(f"\n❌ SCREENING FAILED: {result.get('error', 'Unknown error')}")
            print(f"   Stage: {result.get('stage', 'Unknown')}")
            print("="*70 + "\n")
            return result
    
    def load_example_application(self) -> Dict[str, Any]:
        """Load an example loan application for testing."""
        return {
            "emp_title": "Software Engineer",
            "emp_length": "5 years",
            "annual_income": 85000,
            "verified_income": "Verified",
            "annual_income_joint": 0,
            "earliest_credit_line": "2010-01-15",
            "total_credit_lines": 8,
            "open_credit_lines": 5,
            "total_credit_limit": 45000,
            "total_credit_utilized": 12000,
            "delinq_2y": 0,
            "inquiries_last_12m": 2,
            "debt_to_income": 18.5,
            "num_historical_failed_to_pay": 0,
            "months_since_90d_late": None,
            "current_accounts_delinq": 0,
            "total_collection_amount_ever": 0,
            "accounts_opened_24m": 2,
            "num_collections_last_12m": 0,
            "months_since_last_credit_inquiry": 6,
            "num_accounts_30d_past_due": 0,
            "num_accounts_120d_past_due": 0,
            "num_total_cc_accounts": 4,
            "num_open_cc_accounts": 3,
            "num_cc_carrying_balance": 2,
            "num_mort_accounts": 1,
            "current_installment_accounts": 1,
            "loan_amount": 15000,
            "loan_purpose": "debt_consolidation",
            "application_type": "Individual",
            "term": 60,
            "homeownership": "MORTGAGE",
            "state": "CA",
            "id": "test_app_001"
        }
    
    def screen_from_json(self, json_file: str):
        """Screen an application from a JSON file."""
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            self.screen_application(data)
        except FileNotFoundError:
            print(f"❌ File not found: {json_file}")
            sys.exit(1)
        except json.JSONDecodeError:
            print(f"❌ Invalid JSON file: {json_file}")
            sys.exit(1)
    
    def interactive_mode(self):
        """Run interactive screening mode."""
        print("\n🎯 INTERACTIVE LOAN SCREENING MODE")
        print("="*70)
        print("Enter loan application data interactively.")
        print("(Type 'example' to load sample data, 'quit' to exit)\n")
        
        while True:
            choice = input("Load [e]xample data, [j]SON file, [q]uit? > ").strip().lower()
            
            if choice in ['q', 'quit']:
                print("\n✓ Exiting orchestrator.")
                break
            elif choice in ['e', 'example']:
                app_data = self.load_example_application()
                self.screen_application(app_data)
            elif choice in ['j', 'json']:
                json_file = input("Enter JSON file path: ").strip()
                self.screen_from_json(json_file)
            else:
                print("Invalid choice. Please enter 'e', 'j', or 'q'.")


def main():
    parser = argparse.ArgumentParser(
        description="Credit Decision ADK Orchestrator - Loan Application Screening",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python orchestrator.py --test
  python orchestrator.py --screen application.json
  python orchestrator.py --interactive
        """
    )
    
    parser.add_argument(
        "--test",
        action="store_true",
        help="Screen an example loan application"
    )
    parser.add_argument(
        "--screen",
        type=str,
        help="Screen a loan application from JSON file"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )
    
    args = parser.parse_args()
    
    try:
        orchestrator = LoanScreeningOrchestrator()
        
        if args.test:
            # Run with example data
            app_data = orchestrator.load_example_application()
            orchestrator.screen_application(app_data)
        
        elif args.screen:
            # Screen from JSON file
            orchestrator.screen_from_json(args.screen)
        
        elif args.interactive:
            # Interactive mode
            orchestrator.interactive_mode()
        
        else:
            # Default: test mode
            app_data = orchestrator.load_example_application()
            orchestrator.screen_application(app_data)
    
    except Exception as e:
        print(f"\n❌ Orchestrator Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
