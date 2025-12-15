import pandas as pd
import numpy as np

# ---------------------------------------------------------
# 1. PREPROCESSING & KNOWLEDGE UPLIFT
# ---------------------------------------------------------


def load_and_clean_data(file_path):

    df = pd.read_csv(file_path)

    # Standardize column names for consistency
    time_col = [c for c in df.columns if 'time' in c.lower()][0]
    case_col = [c for c in df.columns if 'stay' in c.lower()][0]
    act_col = [c for c in df.columns if 'activity' in c.lower()][0]

    df[time_col] = pd.to_datetime(df[time_col])

    df = df.sort_values([case_col, time_col])

    return df, case_col, time_col, act_col

# ---------------------------------------------------------
# 2. PERFORMANCE ANALYSIS
# ---------------------------------------------------------


def analyze_performance(df, case_col, time_col):
    # Calculate Lead Time per Case (End Time - Start Time)
    case_durations = df.groupby(case_col)[time_col].agg(
        lambda x: x.max() - x.min()).dt.total_seconds() / 3600

    print("\n--- Performance Metrics ---")
    print(f"Average Lead Time: {case_durations.mean():.2f} hours")
    print(f"Median Lead Time:  {case_durations.median():.2f} hours")
    print(f"Max Lead Time:     {case_durations.max():.2f} hours")

    return case_durations

# ---------------------------------------------------------
# 3. BOTTLENECK ANALYSIS (Transition Times)
# ---------------------------------------------------------


def find_bottlenecks(df, case_col, time_col, act_col):

    df['next_time'] = df.groupby(case_col)[time_col].shift(-1)
    df['next_act'] = df.groupby(case_col)[act_col].shift(-1)

    # Calculate duration between activities (Transition Time)
    df['duration_to_next'] = (
        df['next_time'] - df[time_col]).dt.total_seconds() / 60  # Minutes

    # Group by transition pair (From -> To) and calculate average duration
    bottlenecks = df.groupby([act_col, 'next_act'])[
        'duration_to_next'].mean().sort_values(ascending=False).head(5)

    print("\n--- Top 5 Bottlenecks (Avg Transition Time in Minutes) ---")
    print(bottlenecks)

# ---------------------------------------------------------
# 4. PATTERN-BASED VARIANT ANALYSIS (Exam Question 5)
# ---------------------------------------------------------


def pattern_based_variant_analysis(df, case_col, time_col):
    # Aggregate data per case to find max temp, min O2, etc.
    case_stats = df.groupby(case_col).agg({
        'temperature': 'max',
        'acuity': 'min',
        'o2sat': 'min',
        time_col: lambda x: (x.max() - x.min()).total_seconds() / 3600
    }).rename(columns={time_col: 'Lead_Time_Hours'})

    # Create Binary Features
    case_stats['P1_Fever'] = (case_stats['temperature'] >= 100).astype(int)
    case_stats['P2_HighAcuity'] = (case_stats['acuity'] <= 3).astype(int)
    case_stats['P3_Hypoxemia'] = (case_stats['o2sat'] < 90).astype(int)

    # Create Variant Vector String (e.g., "101")
    case_stats['Variant_Vector'] = (
        case_stats['P1_Fever'].astype(str) +
        case_stats['P2_HighAcuity'].astype(str) +
        case_stats['P3_Hypoxemia'].astype(str)
    )

    print("\n--- Pattern-Based Variant Analysis ---")
    print("Counts per Variant Vector:")
    print(case_stats['Variant_Vector'].value_counts())

    print("\nAverage Lead Time per Variant Vector:")
    print(case_stats.groupby('Variant_Vector')['Lead_Time_Hours'].mean())


df, case_col, time_col, act_col = load_and_clean_data('dataset_for_exam.csv')
analyze_performance(df, case_col, time_col)
find_bottlenecks(df, case_col, time_col, act_col)
pattern_based_variant_analysis(df, case_col, time_col)
