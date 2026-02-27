import pandas as pd
from sklearn.metrics import mean_absolute_error


def check_fairness(df_eval: pd.DataFrame, target_col: str, pred_col: str, group_cols: list):
    print("\n" + "=" * 50)
    print(" ⚖️  FAIRNESS & BIAS DIAGNOSTICS")
    print("=" * 50)

    diagnostics = {}
    for group in group_cols:
        if group in df_eval.columns:
            # Clean up the group name for display (e.g., "education_ Not Graduate" -> "Education (Not Graduate)")
            display_group = group.replace('_', ' ').title()
            print(f"\n➤ Demographic Proxy: {display_group}")
            print("-" * 50)
            print(f"{'Group Value':<15} | {'Count':<8} | {'Mean Absolute Error (MAE)':<20}")
            print("-" * 50)

            group_metrics = []
            for val in df_eval[group].unique():
                subset = df_eval[df_eval[group] == val]
                if len(subset) > 0:
                    mae = mean_absolute_error(subset[target_col], subset[pred_col])
                    count = len(subset)
                    group_metrics.append({'group_val': val, 'MAE': mae, 'count': count})
                    # Format output with commas for currency-like readability
                    print(f"{str(val):<15} | {count:<8} | ${mae:,.2f}")

            res_df = pd.DataFrame(group_metrics)

            # Flag severe imbalance
            max_mae = res_df['MAE'].max()
            min_mae = res_df['MAE'].min()
            variance = (max_mae - min_mae) / min_mae

            if variance > 0.2:
                print(
                    f"\n   ⚠️ WARNING: Substantial bias detected! MAE variance is {variance * 100:.1f}% (> 20% threshold).")
            else:
                print(
                    f"\n   ✅ PASS: Model performs equitably across this demographic (Variance: {variance * 100:.1f}%).")

            diagnostics[group] = res_df
            print("=" * 50)

    return diagnostics