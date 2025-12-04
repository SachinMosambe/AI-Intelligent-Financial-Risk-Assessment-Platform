"""
Classification Feature Engineering for EMI Eligibility Prediction
8 Essential Features Only - Direct from Assignment Requirements
"""

from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd


class ClassificationFeatureEngineer(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        def safe_div(a, b):
            return a / (b + 1e-6)

        # 1. Debt-to-Income Ratio
        X['debt_to_income_ratio'] = safe_div(
            X['current_emi_amount'], 
            X['monthly_salary']
        )

        # 2. Total Monthly Expenses
        X['total_monthly_expenses'] = (
            X['monthly_rent'] +
            X['school_fees'] +
            X['college_fees'] +
            X['travel_expenses'] +
            X['groceries_utilities'] +
            X['other_monthly_expenses']
        )

        # 3. Expense-to-Income Ratio
        X['expense_to_income_ratio'] = safe_div(
            X['total_monthly_expenses'], 
            X['monthly_salary']
        )

        # 4. Affordability Ratio
        X['affordability_ratio'] = safe_div(
            X['monthly_salary'] - X['total_monthly_expenses'] - X['current_emi_amount'],
            X['requested_amount'] / X['requested_tenure']
        )

        # 5. Credit Score (Normalized 0-1)
        X['credit_score_normalized'] = safe_div(
            X['credit_score'] - 300,
            550  # Range: 300-850
        )

        # 6. Emergency Fund Coverage
        X['emergency_fund_coverage'] = safe_div(
            X['emergency_fund'],
            X['total_monthly_expenses']
        )

        # 7. Loan-to-Income Ratio
        X['loan_to_income_ratio'] = safe_div(
            X['requested_amount'],
            X['monthly_salary'] * X['requested_tenure']
        )

        # 8. Dependent Burden
        X['dependent_burden'] = safe_div(
            X['dependents'],
            X['family_size']
        )

        return X