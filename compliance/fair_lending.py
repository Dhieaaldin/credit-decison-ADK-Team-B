"""
Fair Lending Compliance Module

Implements fairness testing per ECOA, Fair Housing Act, and Regulation B requirements.
Tests for disparate impact across protected classes using the 80% rule (Griggs v. Duke Power).
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class DisparateImpactResult:
    """Result of disparate impact analysis for a protected group."""
    group: str
    approval_rate: float
    adverse_impact_ratio: float
    status: str  # 'PASS', 'FAIL', 'CONTROL'
    gap_from_control: float
    sample_size: int


class FairLendingTester:
    """
    Fairness testing implementation per regulatory requirements.
    Tests for disparate impact across protected classes.
    
    Key Regulations:
    - ECOA (Equal Credit Opportunity Act) - 15 U.S.C. §1691
    - Regulation B - 12 CFR §1002
    - Fair Housing Act - 42 U.S.C. §3601
    """
    
    PROTECTED_CLASSES = ['race', 'ethnicity', 'sex', 'age_group', 'national_origin']
    ADVERSE_IMPACT_THRESHOLD = 0.80  # 80% rule (Griggs v. Duke Power)
    
    def __init__(self, threshold: float = 0.80):
        """
        Initialize fair lending tester.
        
        Args:
            threshold: Adverse impact ratio threshold (default: 0.80 per 80% rule)
        """
        self.threshold = threshold
        self.test_results: List[Dict] = []
    
    def calculate_adverse_impact_ratio(
        self, 
        decisions: pd.DataFrame,
        protected_attribute: str,
        favorable_decision: str = "APPROVE"
    ) -> Dict[str, DisparateImpactResult]:
        """
        Calculate adverse impact ratio for each protected group.
        
        AIR = (Approval Rate for Protected Group) / (Approval Rate for Control Group)
        If AIR < 0.80, there is evidence of disparate impact.
        
        Args:
            decisions: DataFrame with 'decision' column and protected attribute column
            protected_attribute: Column name containing protected class information
            favorable_decision: The favorable outcome to measure (default: APPROVE)
            
        Returns:
            Dictionary mapping group names to DisparateImpactResult
        """
        if protected_attribute not in decisions.columns:
            raise ValueError(f"Protected attribute '{protected_attribute}' not in data")
        
        if 'decision' not in decisions.columns:
            raise ValueError("Data must contain 'decision' column")
        
        results = {}
        
        # Calculate approval rates per group
        approval_rates = decisions.groupby(protected_attribute).apply(
            lambda x: (x['decision'] == favorable_decision).mean()
        )
        
        sample_sizes = decisions.groupby(protected_attribute).size()
        
        # Identify control group (highest approval rate)
        control_group = approval_rates.idxmax()
        control_rate = approval_rates[control_group]
        
        for group in decisions[protected_attribute].unique():
            group_rate = approval_rates[group]
            sample_size = sample_sizes[group]
            
            if group == control_group:
                results[group] = DisparateImpactResult(
                    group=group,
                    approval_rate=group_rate,
                    adverse_impact_ratio=1.0,
                    status='CONTROL',
                    gap_from_control=0.0,
                    sample_size=sample_size
                )
            else:
                air = group_rate / control_rate if control_rate > 0 else 0
                status = 'PASS' if air >= self.threshold else 'FAIL'
                
                results[group] = DisparateImpactResult(
                    group=group,
                    approval_rate=group_rate,
                    adverse_impact_ratio=air,
                    status=status,
                    gap_from_control=control_rate - group_rate,
                    sample_size=sample_size
                )
        
        # Log results
        self._log_test_result(protected_attribute, results)
        
        return results
    
    def run_full_fair_lending_analysis(
        self,
        decisions: pd.DataFrame,
        protected_attributes: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Run comprehensive fair lending analysis across all protected classes.
        
        Args:
            decisions: DataFrame with decisions and protected attribute columns
            protected_attributes: List of columns to test (default: all available)
            
        Returns:
            Comprehensive analysis results with summary
        """
        if protected_attributes is None:
            protected_attributes = [
                col for col in self.PROTECTED_CLASSES 
                if col in decisions.columns
            ]
        
        all_results = {}
        violations = []
        
        for attr in protected_attributes:
            try:
                results = self.calculate_adverse_impact_ratio(decisions, attr)
                all_results[attr] = results
                
                # Check for violations
                for group, result in results.items():
                    if result.status == 'FAIL':
                        violations.append({
                            'attribute': attr,
                            'group': group,
                            'air': result.adverse_impact_ratio,
                            'gap': result.gap_from_control
                        })
                        
            except Exception as e:
                all_results[attr] = {'error': str(e)}
        
        return {
            'timestamp': datetime.now().isoformat(),
            'total_decisions': len(decisions),
            'attributes_tested': protected_attributes,
            'results': all_results,
            'violations': violations,
            'overall_status': 'FAIL' if violations else 'PASS',
            'summary': self._generate_summary(all_results, violations)
        }
    
    def marginal_effect_analysis(
        self,
        model,
        test_data: pd.DataFrame,
        feature: str,
        protected_attribute: str
    ) -> Dict[str, Any]:
        """
        Analyze if a feature has different effects across protected groups.
        Used to identify potentially discriminatory features.
        
        Args:
            model: Trained model with predict_proba method
            test_data: Test dataset
            feature: Feature to analyze
            protected_attribute: Protected class column
            
        Returns:
            Marginal effect analysis results per group
        """
        results = {}
        
        for group in test_data[protected_attribute].unique():
            group_data = test_data[test_data[protected_attribute] == group].copy()
            
            if len(group_data) < 50:
                results[group] = {'error': 'Insufficient sample size'}
                continue
            
            # Calculate partial dependence for this group
            feature_values = group_data[feature].quantile([0.1, 0.5, 0.9]).values
            effects = []
            
            for val in feature_values:
                modified = group_data.copy()
                modified[feature] = val
                
                try:
                    if hasattr(model, 'predict_proba'):
                        predictions = model.predict_proba(modified)[:, 1]
                    else:
                        predictions = model.predict(modified)
                    effects.append(predictions.mean())
                except Exception as e:
                    effects.append(None)
            
            # Filter out None values
            valid_effects = [e for e in effects if e is not None]
            
            if len(valid_effects) >= 2:
                results[group] = {
                    'feature_effect_range': max(valid_effects) - min(valid_effects),
                    'direction': 'positive' if valid_effects[-1] > valid_effects[0] else 'negative',
                    'sample_size': len(group_data)
                }
            else:
                results[group] = {'error': 'Could not calculate effects'}
        
        return results
    
    def check_proxy_discrimination(
        self,
        data: pd.DataFrame,
        features: List[str],
        protected_attribute: str,
        correlation_threshold: float = 0.3
    ) -> Dict[str, Any]:
        """
        Check if features are proxies for protected attributes.
        
        Features highly correlated with protected attributes may cause
        disparate impact even if the protected attribute is not used directly.
        
        Args:
            data: Dataset with features and protected attribute
            features: List of features to check
            protected_attribute: Protected class column
            correlation_threshold: Threshold for concerning correlation
            
        Returns:
            Proxy analysis results
        """
        results = {}
        
        # Encode protected attribute if categorical
        if data[protected_attribute].dtype == 'object':
            protected_encoded = pd.get_dummies(data[protected_attribute], drop_first=True)
        else:
            protected_encoded = data[[protected_attribute]]
        
        for feature in features:
            if feature not in data.columns:
                continue
                
            # Handle categorical features
            if data[feature].dtype == 'object':
                feature_encoded = pd.get_dummies(data[feature], drop_first=True)
            else:
                feature_encoded = data[[feature]]
            
            # Calculate correlations
            correlations = {}
            for prot_col in protected_encoded.columns:
                for feat_col in feature_encoded.columns:
                    try:
                        corr = np.corrcoef(
                            protected_encoded[prot_col].fillna(0),
                            feature_encoded[feat_col].fillna(0)
                        )[0, 1]
                        correlations[f"{feat_col}_vs_{prot_col}"] = corr
                    except:
                        pass
            
            max_correlation = max(abs(c) for c in correlations.values()) if correlations else 0
            
            results[feature] = {
                'max_correlation': max_correlation,
                'is_proxy': max_correlation > correlation_threshold,
                'correlations': correlations,
                'recommendation': 'REMOVE' if max_correlation > 0.5 else 
                                 'REVIEW' if max_correlation > correlation_threshold else 'OK'
            }
        
        return {
            'protected_attribute': protected_attribute,
            'features_analyzed': features,
            'threshold': correlation_threshold,
            'results': results,
            'potential_proxies': [f for f, r in results.items() if r.get('is_proxy', False)]
        }
    
    def _log_test_result(
        self,
        protected_attribute: str,
        results: Dict[str, DisparateImpactResult]
    ):
        """Log test results for audit trail."""
        self.test_results.append({
            'timestamp': datetime.now().isoformat(),
            'protected_attribute': protected_attribute,
            'results': {k: vars(v) for k, v in results.items()},
            'any_violations': any(r.status == 'FAIL' for r in results.values())
        })
    
    def _generate_summary(
        self,
        results: Dict[str, Any],
        violations: List[Dict]
    ) -> str:
        """Generate human-readable summary of fair lending analysis."""
        summary_lines = [
            f"Fair Lending Analysis Summary",
            f"=" * 40,
            f"Attributes Tested: {len(results)}",
            f"Violations Found: {len(violations)}",
        ]
        
        if violations:
            summary_lines.append("\nViolations requiring remediation:")
            for v in violations:
                summary_lines.append(
                    f"  - {v['attribute']}/{v['group']}: AIR={v['air']:.2%} "
                    f"(below 80% threshold)"
                )
        else:
            summary_lines.append("\n✓ No fair lending violations detected.")
        
        return "\n".join(summary_lines)
    
    def get_audit_report(self) -> List[Dict]:
        """Get audit trail of all tests performed."""
        return self.test_results.copy()
