"""
Explanation Agent

Generates human-readable explanations for credit decisions.
"""

import os
import requests
import json
from dotenv import load_dotenv
from agents.base_agent import BaseAgent, AgentMessage, AgentResponse
from typing import Dict, Any, List

load_dotenv()

try:
    from google.cloud import aiplatform
    from vertexai.preview.generative_models import GenerativeModel
    GOOGLE_AI_AVAILABLE = True
except ImportError:
    GOOGLE_AI_AVAILABLE = False


class ExplanationAgent(BaseAgent):
    """
    Agent responsible for generating explanations for decisions.
    
    Input: Decision and risk analysis
    Output: Human-readable explanation
    """
    
    def __init__(self, use_google_ai: bool = False, model_name: str = "gemini-pro"):
        """
        Initialize explanation agent.
        
        Args:
            use_google_ai: If True, use Google Gemini for explanations
            model_name: Google AI model name (default: gemini-pro)
        """
        super().__init__("explanation_agent", "Explanation Generation")
        self.use_google_ai = use_google_ai and GOOGLE_AI_AVAILABLE
        self.model_name = model_name
        
        # OpenRouter Configuration
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        self.use_openrouter = self.openrouter_api_key is not None
        
        if self.use_google_ai:
            try:
                self.ai_model = GenerativeModel(model_name)
            except Exception as e:
                print(f"Warning: Could not initialize Google AI model: {e}")
                self.use_google_ai = False
        
        if self.use_openrouter:
            print(f"✓ OpenRouter initialized for explanations")
        elif not self.use_google_ai:
             print("Warning: No AI provider (Google/OpenRouter) available. Using templates.")
    
    def process(self, message: AgentMessage) -> AgentResponse:
        """
        Generate explanation for the decision.
        
        Args:
            message: Contains decision and risk analysis
            
        Returns:
            AgentResponse with explanation text
        """
        import time
        start_time = time.time()
        
        try:
            decision_data = message.payload.get("decision", {})
            risk_data = message.payload.get("risk_analysis", {})
            per_dimension_risk = risk_data.get("per_dimension_risk", {})
            anomalies = risk_data.get("anomalies", [])
            
            if not decision_data:
                return AgentResponse(
                    success=False,
                    error="No decision data provided",
                    processing_time=time.time() - start_time
                )
            
            # Generate explanation (Priority: Google AI -> OpenRouter -> Template)
            method = "template"
            if self.use_google_ai:
                explanation = self._generate_ai_explanation(
                    decision_data, per_dimension_risk, anomalies
                )
                method = "google_ai"
            elif self.use_openrouter:
                explanation = self._generate_openrouter_explanation(
                    decision_data, per_dimension_risk, anomalies
                )
                method = "openrouter"
            else:
                explanation = self._generate_template_explanation(
                    decision_data, per_dimension_risk, anomalies
                )
            
            # Log processing
            self.log_message(message)
            self.update_state({
                "last_processed": time.time(),
                "total_explanations": self.state.get("total_explanations", 0) + 1
            })
            
            return AgentResponse(
                success=True,
                data={"explanation": explanation},
                metadata={
                    "method": method,
                    "length": len(explanation)
                },
                processing_time=time.time() - start_time
            )
            
        except Exception as e:
            return AgentResponse(
                success=False,
                error=f"Explanation error: {str(e)}",
                processing_time=time.time() - start_time
            )

    def _generate_openrouter_explanation(
        self,
        decision_data: Dict[str, Any],
        per_dimension_risk: Dict[str, Dict[str, Any]],
        anomalies: List[Dict[str, str]]
    ) -> str:
        """Generate explanation using OpenRouter API."""
        try:
            prompt = self._build_explanation_prompt(
                decision_data, per_dimension_risk, anomalies
            )
            
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.openrouter_api_key}",
                    "Content-Type": "application/json",
                },
                data=json.dumps({
                    "model": "nvidia/nemotron-3-nano-30b-a3b:free",  # Using a reliable free model
                    "messages": [
                        {"role": "system", "content": "You are a professional credit risk analyst providing justifications for loan screening decisions."},
                        {"role": "user", "content": prompt}
                    ]
                }),
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()['choices'][0]['message']['content']
            else:
                print(f"OpenRouter error: {response.text}. Falling back to template.")
                return self._generate_template_explanation(decision_data, per_dimension_risk, anomalies)
                
        except Exception as e:
            print(f"OpenRouter exception: {e}. Falling back to template.")
            return self._generate_template_explanation(decision_data, per_dimension_risk, anomalies)
    
    def _generate_ai_explanation(
        self,
        decision_data: Dict[str, Any],
        per_dimension_risk: Dict[str, Dict[str, Any]],
        anomalies: List[Dict[str, str]]
    ) -> str:
        """Generate explanation using Google Gemini AI."""
        try:
            prompt = self._build_explanation_prompt(
                decision_data, per_dimension_risk, anomalies
            )
            
            response = self.ai_model.generate_content(prompt)
            return response.text
            
        except Exception as e:
            print(f"AI explanation error: {e}. Falling back to template.")
            return self._generate_template_explanation(
                decision_data, per_dimension_risk, anomalies
            )
    
    def _generate_template_explanation(
        self,
        decision_data: Dict[str, Any],
        per_dimension_risk: Dict[str, Dict[str, Any]],
        anomalies: List[Dict[str, str]]
    ) -> str:
        """Generate explanation using template."""
        recommendation = decision_data.get("recommendation", "UNKNOWN")
        confidence = decision_data.get("confidence", 0.0)
        default_rate = decision_data.get("default_rate", 0.0)
        
        explanation = f"Decision: {recommendation} (Confidence: {confidence:.1%})\n\n"
        
        explanation += "Overall Analysis:\n"
        explanation += f"- Similar historical loans had a default rate of {default_rate:.1%}\n"
        explanation += f"- Found {len(anomalies)} potential risk factors\n\n"
        
        # Per-dimension summary
        explanation += "Per-Dimension Risk Assessment:\n"
        for chunk_type, analysis in per_dimension_risk.items():
            risk_level = analysis.get("risk_level", "UNKNOWN")
            default_rate_dim = analysis.get("default_rate", 0.0)
            avg_sim = analysis.get("avg_similarity", 0.0)
            status_summary = analysis.get("status_summary", "Unknown status breakdown")
            
            explanation += f"- {chunk_type.replace('_', ' ').title()}: {risk_level} risk\n"
            explanation += f"  (Default Rate: {default_rate_dim:.1%}, Similarity: {avg_sim:.3f})\n"
            explanation += f"  (Matches: {status_summary})\n"
        
        # Anomalies
        if anomalies:
            explanation += f"\nDetected Anomalies:\n"
            for i, anomaly in enumerate(anomalies, 1):
                explanation += f"{i}. [{anomaly.get('severity', 'UNKNOWN')}] {anomaly.get('description', '')}\n"
        
        # Rationale
        explanation += f"\nRationale:\n"
        if recommendation == "APPROVE":
            explanation += "The application shows strong similarity to historically successful loans "
            explanation += "with low default rates. No major risk factors were identified."
        elif recommendation == "REJECT":
            explanation += "The application shows high similarity to loans that defaulted, or multiple "
            explanation += "high-severity risk factors were detected. The risk of default is too high."
        else:
            explanation += "The application shows mixed signals. Some dimensions indicate low risk while "
            explanation += "others show concerns. Manual review by a credit analyst is recommended."
        
        return explanation
    
    def _build_explanation_prompt(
        self,
        decision_data: Dict[str, Any],
        per_dimension_risk: Dict[str, Dict[str, Any]],
        anomalies: List[Dict[str, str]]
    ) -> str:
        """Build prompt for AI explanation generation."""
        return f"""
Generate a clear, professional explanation for a credit decision.

Decision: {decision_data.get('recommendation')}
Confidence: {decision_data.get('confidence', 0.0):.1%}
Default Rate: {decision_data.get('default_rate', 0.0):.1%}

Risk Analysis by Dimension:
{self._format_risk_analysis(per_dimension_risk)}

Anomalies Detected:
{self._format_anomalies(anomalies)}

Provide a concise explanation (2-3 paragraphs) that:
1. Summarizes the decision and confidence level
2. Explains the key risk factors identified
3. Provides reasoning for the recommendation

Write in plain English suitable for a credit analyst or loan officer.
"""
    
    def _format_risk_analysis(self, per_dimension_risk: Dict[str, Dict[str, Any]]) -> str:
        """Format risk analysis for prompt."""
        lines = []
        for chunk_type, analysis in per_dimension_risk.items():
            status_summary = analysis.get("status_summary", "Unknown status breakdown")
            lines.append(
                f"- {chunk_type.replace('_', ' ').title()}: {analysis.get('risk_level', 'UNKNOWN')} risk "
                f"(default rate: {analysis.get('default_rate', 0.0):.1%})\n"
                f"  Historical Similarity Data: {status_summary}"
            )
        return "\n".join(lines)
    
    def _format_anomalies(self, anomalies: List[Dict[str, str]]) -> str:
        """Format anomalies for prompt."""
        if not anomalies:
            return "None"
        return "\n".join([
            f"- [{a.get('severity', 'UNKNOWN')}] {a.get('type', 'Unknown Type')}: {a.get('description', '')}"
            for a in anomalies
        ])
