import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def perform_eda(df: pd.DataFrame, target_col: str, output_dir: str = "eda_outputs"):
    """Performs full Exploratory Data Analysis."""
    os.makedirs(output_dir, exist_ok=True)

    # Check Missing Values
    missing = df.isnull().sum()
    missing.to_csv(f"{output_dir}/missing_values.csv")

    # Numeric Distributions
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        plt.figure()
        sns.histplot(df[col].dropna(), kde=True)
        plt.title(f'Distribution of {col}')
        plt.savefig(f"{output_dir}/dist_{col}.png")
        plt.close()

    # Correlations (avoid target leakage check)
    corr = df[numeric_cols].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    plt.title("Feature Correlation")
    plt.savefig(f"{output_dir}/correlation_matrix.png")
    plt.close()

    # Outliers (Boxplots)
    for col in numeric_cols:
        plt.figure()
        sns.boxplot(x=df[col])
        plt.title(f'Outliers in {col}')
        plt.savefig(f"{output_dir}/boxplot_{col}.png")
        plt.close()

    # Demographic / Fairness preliminary look
    cat_cols = ['education', 'self_employed']  # present in this dataset
    for col in cat_cols:
        if col in df.columns:
            plt.figure()
            sns.boxplot(data=df, x=col, y=target_col)
            plt.title(f'{target_col} by {col}')
            plt.savefig(f"{output_dir}/bias_check_{col}.png")
            plt.close()

    print(f"EDA completed. Outputs saved to {output_dir}")