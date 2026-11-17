"""Shared definitions used by BOTH the notebook and app.py."""
import pandas as pd

CATEGORICAL = ["Department", "JobRole", "BusinessTravel", "OverTime"]

NUMERIC = [
    "JobLevel", "MonthlyIncome", "DistanceFromHome",
    "JobSatisfaction", "EnvironmentSatisfaction", "WorkLifeBalance",
    "JobInvolvement", "StockOptionLevel", "TotalWorkingYears",
    "NumCompaniesWorked", "YearsAtCompany", "YearsSinceLastPromotion",
    "YearsWithCurrManager", "TrainingTimesLastYear",
]

INPUT_COLUMNS = CATEGORICAL + NUMERIC

ENGINEERED = ["AvgSatisfaction", "TenureRatio", "PromotionGap",
              "IncomePerLevel", "JobHopping"]

NUMERIC_ALL = NUMERIC + ENGINEERED


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create HR-meaningful ratios from the raw inputs."""
    df = df.copy()
    df["AvgSatisfaction"] = df[["JobSatisfaction", "EnvironmentSatisfaction",
                                "WorkLifeBalance"]].mean(axis=1)
    df["TenureRatio"] = df["YearsAtCompany"] / (df["TotalWorkingYears"] + 1)
    df["PromotionGap"] = df["YearsSinceLastPromotion"] / (df["YearsAtCompany"] + 1)
    df["IncomePerLevel"] = df["MonthlyIncome"] / df["JobLevel"].clip(lower=1)
    df["JobHopping"] = df["NumCompaniesWorked"] / (df["TotalWorkingYears"] + 1)
    return df
