"""
Ingestion & Validation Agent

Validates and normalizes loan application data before processing.
"""

import pandas as pd
from typing import Dict, Any
from agents.base_agent import BaseAgent, AgentMessage, AgentResponse


class IngestionAgent(BaseAgent):
    """
    Agent responsible for data ingestion, validation, and normalization.
    
    Input: Raw loan application data
    Output: Cleaned and validated data
    """
    
    def __init__(self):
        super().__init__("ingestion_agent", "Data Validation & Normalization")
        self.validation_rules = self._initialize_validation_rules()
    
    def _initialize_validation_rules(self) -> Dict[str, Any]:
        """Initialize data validation rules."""
        return {
            "required_fields": [
                "loan_amount", "annual_income",
                "debt_to_income", "emp_title", "state"
            ],
            "numeric_fields": [
                "loan_amount", "annual_income", "debt_to_income",
                "total_credit_lines", "open_credit_lines"
            ],
            "valid_states": [
                "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
                "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
                "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
                "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
                "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"
            ]
        }
    
    def process(self, message: AgentMessage) -> AgentResponse:
        """
        Process loan application data: validate and normalize.
        
        Args:
            message: Contains raw loan application data
            
        Returns:
            AgentResponse with cleaned data
        """
        import time
        start_time = time.time()
        
        try:
            raw_data = message.payload.get("data", {})
            
            # Validate required fields
            validation_errors = self._validate_data(raw_data)
            if validation_errors:
                return AgentResponse(
                    success=False,
                    error=f"Validation failed: {', '.join(validation_errors)}",
                    processing_time=time.time() - start_time
                )
            
            # Normalize data
            cleaned_data = self._normalize_data(raw_data)
            
            # Log processing
            self.log_message(message)
            self.update_state({
                "last_processed": time.time(),
                "total_processed": self.state.get("total_processed", 0) + 1
            })
            
            return AgentResponse(
                success=True,
                data={"cleaned_data": cleaned_data},
                metadata={
                    "validation_passed": True,
                    "fields_normalized": len(cleaned_data)
                },
                processing_time=time.time() - start_time
            )
            
        except Exception as e:
            return AgentResponse(
                success=False,
                error=f"Processing error: {str(e)}",
                processing_time=time.time() - start_time
            )
    
    def _validate_data(self, data: Dict[str, Any]) -> list[str]:
        """Validate data against rules."""
        errors = []
        
        # Check required fields
        for field in self.validation_rules["required_fields"]:
            if field not in data or data[field] is None:
                errors.append(f"Missing required field: {field}")
        
        # Validate numeric fields
        for field in self.validation_rules["numeric_fields"]:
            if field in data:
                try:
                    float(data[field])
                except (ValueError, TypeError):
                    errors.append(f"Invalid numeric value for {field}")
        
        # Validate state code
        if "state" in data:
            state = str(data["state"]).upper().strip()
            if state not in self.validation_rules["valid_states"]:
                errors.append(f"Invalid state code: {data['state']}")
        
        return errors
    
    def _normalize_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize data types and formats."""
        normalized = data.copy()
        
        # Normalize numeric fields
        for field in self.validation_rules["numeric_fields"]:
            if field in normalized:
                try:
                    normalized[field] = float(normalized[field]) if normalized[field] is not None else 0.0
                except (ValueError, TypeError):
                    normalized[field] = 0.0
        
        # Normalize state code
        if "state" in normalized:
            normalized["state"] = str(normalized["state"]).upper().strip()
        
        # Normalize text fields
        text_fields = ["emp_title", "loan_purpose", "application_type", "homeownership"]
        for field in text_fields:
            if field in normalized:
                normalized[field] = str(normalized[field]).strip() if normalized[field] else ""
        
        # Handle NaN values
        for key, value in normalized.items():
            if pd.isna(value) if isinstance(value, (int, float)) else False:
                normalized[key] = None
        
        return normalized
