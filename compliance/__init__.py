"""
Compliance Module for Credit Decision ADK

This module provides fair lending compliance testing, adverse action notice
generation, and regulatory compliance utilities.
"""

from compliance.fair_lending import FairLendingTester
from compliance.adverse_action import AdverseActionGenerator

__all__ = ['FairLendingTester', 'AdverseActionGenerator']
