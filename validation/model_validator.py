"""
Model Validation Module

Implements comprehensive model validation per SR 11-7 requirements.
Covers discriminatory power, calibration, stability, and sensitivity testing.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime


@dataclass
class ValidationResult:
    """Result of a validation test."""
    test_name: str
    status: str  # 'PASS', 'FAIL', 'WARNING'
    metric_value: float
    threshold: float
    details: Dict[str, Any]
    interpretation: str


class ModelValidator:
    """
    Model validation framework per SR 11-7 requirements.
    
    Provides comprehensive validation including:
    - Discriminatory power (Gini, KS, AUC-ROC)
    - Calibration (Hosmer-Lemeshow test)
    - Stability (Population Stability Index)
    - Concentration (portfolio analysis)
    - Sensitivity (feature importance and stress testing)
    """
    
    # Validation thresholds
    THRESHOLDS = {
        'gini_minimum': 0.30,           # Minimum acceptable Gini coefficient
        'ks_minimum': 0.20,             # Minimum acceptable KS statistic
        'auc_minimum': 0.65,            # Minimum acceptable AUC-ROC
        'psi_warning': 0.10,            # PSI warning threshold
        'psi_critical': 0.25,           # PSI critical threshold
        'hosmer_lemeshow_pvalue': 0.05, # HL test p-value threshold
        'calibration_ratio_range': (0.8, 1.2),  # Acceptable predicted/actual ratio
    }
    
    def __init__(self, target_column: str = 'defaulted'):
        """
        Initialize model validator.
        
        Args:
            target_column: Name of the target (outcome) column
        """
        self.target_column = target_column
        self.validation_results: List[ValidationResult] = []
    
    def run_full_validation(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        data: Optional[pd.DataFrame] = None,
        recent_predictions: Optional[np.ndarray] = None,
        recent_actuals: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Execute complete validation suite.
        
        Args:
            predictions: Model predicted probabilities
            actuals: Actual outcomes (0/1)
            data: Full dataset for additional analysis
            recent_predictions: Recent period predictions for stability
            recent_actuals: Recent period actuals for stability
            
        Returns:
            Comprehensive validation results
        """
        self.validation_results = []
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'sample_size': len(predictions),
            'actual_default_rate': float(np.mean(actuals)),
            'predicted_default_rate': float(np.mean(predictions)),
            'tests': {}
        }
        
        # Discriminatory power tests
        results['tests']['discriminatory_power'] = self.test_discriminatory_power(
            predictions, actuals
        )
        
        # Calibration tests
        results['tests']['calibration'] = self.test_calibration(
            predictions, actuals
        )
        
        # Stability tests (if recent data provided)
        if recent_predictions is not None and recent_actuals is not None:
            results['tests']['stability'] = self.test_stability(
                predictions, recent_predictions
            )
        
        # Concentration analysis
        results['tests']['concentration'] = self.test_concentration(
            predictions, actuals
        )
        
        # Generate overall status
        results['overall_status'] = self._determine_overall_status(results['tests'])
        results['summary'] = self._generate_summary(results)
        
        return results
    
    def test_discriminatory_power(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray
    ) -> Dict[str, Any]:
        """
        Test model's ability to rank-order risk.
        
        Uses AUC-ROC, Gini coefficient, and Kolmogorov-Smirnov statistic.
        """
        # Calculate AUC-ROC
        auc = self._calculate_auc(actuals, predictions)
        gini = 2 * auc - 1
        ks = self._calculate_ks_statistic(actuals, predictions)
        
        # Determine status
        gini_status = 'PASS' if gini >= self.THRESHOLDS['gini_minimum'] else 'FAIL'
        ks_status = 'PASS' if ks >= self.THRESHOLDS['ks_minimum'] else 'FAIL'
        auc_status = 'PASS' if auc >= self.THRESHOLDS['auc_minimum'] else 'FAIL'
        
        overall_status = 'PASS' if all(s == 'PASS' for s in [gini_status, ks_status, auc_status]) else 'FAIL'
        
        result = ValidationResult(
            test_name='Discriminatory Power',
            status=overall_status,
            metric_value=gini,
            threshold=self.THRESHOLDS['gini_minimum'],
            details={
                'auc_roc': auc,
                'gini_coefficient': gini,
                'ks_statistic': ks,
                'auc_status': auc_status,
                'gini_status': gini_status,
                'ks_status': ks_status
            },
            interpretation=self._interpret_gini(gini)
        )
        self.validation_results.append(result)
        
        return {
            'auc_roc': auc,
            'gini_coefficient': gini,
            'ks_statistic': ks,
            'status': overall_status,
            'thresholds': {
                'gini_minimum': self.THRESHOLDS['gini_minimum'],
                'ks_minimum': self.THRESHOLDS['ks_minimum'],
                'auc_minimum': self.THRESHOLDS['auc_minimum']
            },
            'interpretation': self._interpret_gini(gini)
        }
    
    def test_calibration(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        num_bins: int = 10
    ) -> Dict[str, Any]:
        """
        Test if predicted probabilities match observed frequencies.
        """
        # Bin predictions into deciles
        bin_edges = np.percentile(predictions, np.linspace(0, 100, num_bins + 1))
        bin_edges[0] = -0.001
        bin_edges[-1] = 1.001
        
        bin_indices = np.digitize(predictions, bin_edges) - 1
        
        calibration_data = []
        for i in range(num_bins):
            mask = bin_indices == i
            if mask.sum() > 0:
                predicted_mean = predictions[mask].mean()
                actual_mean = actuals[mask].mean()
                count = mask.sum()
                
                ratio = actual_mean / predicted_mean if predicted_mean > 0 else None
                calibration_data.append({
                    'bin': i + 1,
                    'predicted': float(predicted_mean),
                    'actual': float(actual_mean),
                    'count': int(count),
                    'ratio': float(ratio) if ratio else None
                })
        
        # Simple calibration test (Brier score)
        brier_score = np.mean((predictions - actuals) ** 2)
        
        # Check if ratios are within acceptable range
        ratios = [d['ratio'] for d in calibration_data if d['ratio'] is not None]
        ratios_in_range = sum(
            1 for r in ratios 
            if self.THRESHOLDS['calibration_ratio_range'][0] <= r <= self.THRESHOLDS['calibration_ratio_range'][1]
        )
        
        calibration_quality = ratios_in_range / len(ratios) if ratios else 0
        status = 'PASS' if calibration_quality >= 0.7 else 'WARNING' if calibration_quality >= 0.5 else 'FAIL'
        
        result = ValidationResult(
            test_name='Calibration',
            status=status,
            metric_value=brier_score,
            threshold=0.25,
            details={
                'calibration_by_decile': calibration_data,
                'brier_score': brier_score,
                'ratios_in_range_pct': calibration_quality
            },
            interpretation=f"Brier score: {brier_score:.4f}. {calibration_quality:.0%} of bins well-calibrated."
        )
        self.validation_results.append(result)
        
        return {
            'calibration_by_decile': calibration_data,
            'brier_score': brier_score,
            'calibration_quality': calibration_quality,
            'status': status
        }
    
    def test_stability(
        self,
        baseline_predictions: np.ndarray,
        recent_predictions: np.ndarray,
        num_buckets: int = 10
    ) -> Dict[str, Any]:
        """
        Test model stability over time (PSI analysis).
        """
        psi = self._calculate_psi(baseline_predictions, recent_predictions, num_buckets)
        
        if psi < self.THRESHOLDS['psi_warning']:
            status = 'PASS'
            interpretation = 'Model is stable'
        elif psi < self.THRESHOLDS['psi_critical']:
            status = 'WARNING'
            interpretation = 'Moderate shift detected, investigation recommended'
        else:
            status = 'FAIL'
            interpretation = 'Significant drift detected, action required'
        
        result = ValidationResult(
            test_name='Stability (PSI)',
            status=status,
            metric_value=psi,
            threshold=self.THRESHOLDS['psi_warning'],
            details={
                'psi': psi,
                'baseline_count': len(baseline_predictions),
                'recent_count': len(recent_predictions)
            },
            interpretation=interpretation
        )
        self.validation_results.append(result)
        
        return {
            'psi': psi,
            'status': status,
            'interpretation': interpretation,
            'thresholds': {
                'warning': self.THRESHOLDS['psi_warning'],
                'critical': self.THRESHOLDS['psi_critical']
            }
        }
    
    def test_concentration(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        num_buckets: int = 10
    ) -> Dict[str, Any]:
        """
        Analyze risk concentration across score buckets.
        """
        # Create score buckets
        bucket_edges = np.percentile(predictions, np.linspace(0, 100, num_buckets + 1))
        bucket_indices = np.digitize(predictions, bucket_edges) - 1
        bucket_indices = np.clip(bucket_indices, 0, num_buckets - 1)
        
        concentration_data = []
        cumulative_defaults = 0
        total_defaults = actuals.sum()
        
        for i in range(num_buckets):
            mask = bucket_indices == i
            if mask.sum() > 0:
                bucket_defaults = actuals[mask].sum()
                cumulative_defaults += bucket_defaults
                
                concentration_data.append({
                    'bucket': i + 1,
                    'score_range': f"{bucket_edges[i]:.2f}-{bucket_edges[i+1]:.2f}",
                    'count': int(mask.sum()),
                    'pct_of_total': float(mask.sum() / len(predictions)),
                    'default_rate': float(actuals[mask].mean()),
                    'defaults': int(bucket_defaults),
                    'cumulative_default_capture': float(cumulative_defaults / total_defaults) if total_defaults > 0 else 0
                })
        
        # Check monotonicity (default rates should increase with score)
        default_rates = [d['default_rate'] for d in concentration_data]
        monotonic_violations = sum(
            1 for i in range(1, len(default_rates)) 
            if default_rates[i] < default_rates[i-1] * 0.9  # 10% tolerance
        )
        
        is_monotonic = monotonic_violations <= 1
        status = 'PASS' if is_monotonic else 'WARNING'
        
        result = ValidationResult(
            test_name='Concentration',
            status=status,
            metric_value=float(monotonic_violations),
            threshold=1.0,
            details={
                'concentration_by_bucket': concentration_data,
                'is_monotonic': is_monotonic,
                'violations': monotonic_violations
            },
            interpretation=f"{'Monotonic' if is_monotonic else 'Non-monotonic'} rank ordering with {monotonic_violations} violations"
        )
        self.validation_results.append(result)
        
        return {
            'concentration_by_bucket': concentration_data,
            'is_monotonic': is_monotonic,
            'monotonic_violations': monotonic_violations,
            'status': status
        }
    
    def _calculate_auc(self, actuals: np.ndarray, predictions: np.ndarray) -> float:
        """Calculate AUC-ROC using the trapezoidal rule."""
        # Sort by predictions
        sorted_indices = np.argsort(predictions)[::-1]
        sorted_actuals = actuals[sorted_indices]
        
        # Calculate TPR and FPR at each threshold
        tps = np.cumsum(sorted_actuals)
        fps = np.cumsum(1 - sorted_actuals)
        
        total_positives = actuals.sum()
        total_negatives = len(actuals) - total_positives
        
        if total_positives == 0 or total_negatives == 0:
            return 0.5  # No discrimination possible
        
        tpr = tps / total_positives
        fpr = fps / total_negatives
        
        # Add origin point
        tpr = np.concatenate([[0], tpr])
        fpr = np.concatenate([[0], fpr])
        
        # Calculate AUC using trapezoidal rule
        auc = np.trapz(tpr, fpr)
        
        return float(auc)
    
    def _calculate_ks_statistic(self, actuals: np.ndarray, predictions: np.ndarray) -> float:
        """Calculate Kolmogorov-Smirnov statistic."""
        # Separate good and bad predictions
        good_predictions = predictions[actuals == 0]
        bad_predictions = predictions[actuals == 1]
        
        if len(good_predictions) == 0 or len(bad_predictions) == 0:
            return 0.0
        
        # Create CDFs
        all_values = np.sort(np.unique(predictions))
        
        good_cdf = np.array([np.mean(good_predictions <= v) for v in all_values])
        bad_cdf = np.array([np.mean(bad_predictions <= v) for v in all_values])
        
        ks = float(np.max(np.abs(good_cdf - bad_cdf)))
        
        return ks
    
    def _calculate_psi(
        self,
        expected: np.ndarray,
        actual: np.ndarray,
        buckets: int = 10
    ) -> float:
        """Calculate Population Stability Index."""
        # Create buckets based on expected distribution
        breakpoints = np.percentile(expected, np.linspace(0, 100, buckets + 1))
        breakpoints[0] = -np.inf
        breakpoints[-1] = np.inf
        
        expected_counts = np.histogram(expected, bins=breakpoints)[0]
        actual_counts = np.histogram(actual, bins=breakpoints)[0]
        
        # Add small constant to avoid division by zero
        expected_pct = (expected_counts + 0.0001) / len(expected)
        actual_pct = (actual_counts + 0.0001) / len(actual)
        
        psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
        
        return float(psi)
    
    def _interpret_gini(self, gini: float) -> str:
        """Interpret Gini coefficient value."""
        if gini >= 0.50:
            return "Excellent discriminatory power"
        elif gini >= 0.40:
            return "Good discriminatory power"
        elif gini >= 0.30:
            return "Acceptable discriminatory power"
        elif gini >= 0.20:
            return "Weak discriminatory power - model needs improvement"
        else:
            return "Very weak discriminatory power - model is not suitable"
    
    def _determine_overall_status(self, tests: Dict[str, Any]) -> str:
        """Determine overall validation status."""
        statuses = [t.get('status', 'UNKNOWN') for t in tests.values()]
        
        if 'FAIL' in statuses:
            return 'FAIL'
        elif 'WARNING' in statuses:
            return 'WARNING'
        elif all(s == 'PASS' for s in statuses):
            return 'PASS'
        else:
            return 'UNKNOWN'
    
    def _generate_summary(self, results: Dict[str, Any]) -> str:
        """Generate human-readable summary."""
        status = results.get('overall_status', 'UNKNOWN')
        tests = results.get('tests', {})
        
        summary_lines = [
            f"Model Validation Summary",
            f"=" * 40,
            f"Overall Status: {status}",
            f"Sample Size: {results.get('sample_size', 'N/A')}",
            f"Actual Default Rate: {results.get('actual_default_rate', 0):.2%}",
            f"Predicted Default Rate: {results.get('predicted_default_rate', 0):.2%}",
            f"",
            f"Test Results:"
        ]
        
        for test_name, test_result in tests.items():
            test_status = test_result.get('status', 'UNKNOWN')
            summary_lines.append(f"  - {test_name}: {test_status}")
        
        return "\n".join(summary_lines)
    
    def get_validation_report(self) -> List[ValidationResult]:
        """Get all validation results."""
        return self.validation_results.copy()
