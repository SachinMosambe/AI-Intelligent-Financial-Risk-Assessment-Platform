"""
Regression Feature Engineering for Maximum EMI Amount Prediction
Essential features only - Clean and Justifiable
"""

from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd


class RegressionFeatureEngineer(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        def safe_div(a, b):
            return a / (b + 1e-6)

        # 1. Total Monthly Expenses
        X['total_monthly_expenses'] = (
            X['monthly_rent'] +
            X['school_fees'] +
            X['college_fees'] +
            X['travel_expenses'] +
            X['groceries_utilities'] +
            X['other_monthly_expenses'] +
            X['current_emi_amount']
        )

        # 2. Net Disposable Income (Key for EMI calculation)
        X['net_disposable_income'] = (
            X['monthly_salary'] - X['total_monthly_expenses']
        ).clip(lower=0)

        # 3. Expense-to-Income Ratio
        X['expense_to_income'] = safe_div(
            X['total_monthly_expenses'], 
            X['monthly_salary']
        )

        # 4. Safe EMI Limit (40% of disposable income)
        X['safe_emi_limit'] = X['net_disposable_income'] * 0.40

        # 5. Available EMI Capacity (after existing EMIs)
        X['available_emi_capacity'] = (
            X['safe_emi_limit'] - X['current_emi_amount']
        ).clip(lower=0)

        # 6. Credit Score Factor (0 to 1)
        X['credit_score_factor'] = safe_div(
            X['credit_score'] - 300,
            850 - 300
        )

        # 7. Total Liquid Assets
        X['total_liquid_assets'] = (
            X['bank_balance'] + X['emergency_fund']
        )

        # 8. Emergency Fund Coverage (in months)
        X['emergency_months_coverage'] = safe_div(
            X['emergency_fund'],
            X['total_monthly_expenses']
        )

        # 9. Employment Stability Factor
        X['employment_stability'] = np.where(
            X['years_of_employment'] >= 5, 
            1.0,
            X['years_of_employment'] / 5.0
        )

        # 10. Age Factor (optimal age: 30-50)
        X['age_factor'] = np.where(
            (X['age'] >= 30) & (X['age'] <= 50),
            1.0,
            np.where(X['age'] < 30, 0.85, 0.75)
        )

        # 11. Per Capita Income
        X['per_capita_income'] = safe_div(
            X['monthly_salary'],
            X['family_size']
        )

        # 12. Loan-to-Annual-Income Ratio
        X['loan_to_annual_income'] = safe_div(
            X['requested_amount'],
            X['monthly_salary'] * 12
        )

        # 13. Credit-Adjusted EMI Capacity
        X['credit_adjusted_emi'] = (
            X['safe_emi_limit'] * (0.8 + X['credit_score_factor'] * 0.4)
        )

        # 14. Fully Adjusted EMI (with all risk factors)
        X['fully_adjusted_emi'] = (
            X['credit_adjusted_emi'] * 
            X['age_factor'] * 
            X['employment_stability']
        )

        return X