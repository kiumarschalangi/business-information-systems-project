import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------
# 1. PREPROCESSING & KNOWLEDGE UPLIFT
# ---------------------------------------------------------


def load_and_clean_data(file_path):

    df = pd.read_csv("/content/dataset_for_exam.csv")

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
    # Shift to find the next activity and time for the same case
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

    case_stats = df.groupby(case_col).agg({
        'temperature': 'max',
        'acuity': 'min',
        'o2sat': 'min',
        time_col: lambda x: (x.max() - x.min()).total_seconds() / 3600
    }).rename(columns={time_col: 'Lead_Time_Hours'})

    # Binary Features
    case_stats['P1_Fever'] = (case_stats['temperature'] >= 100).astype(int)
    case_stats['P2_HighAcuity'] = (case_stats['acuity'] <= 3).astype(int)
    case_stats['P3_Hypoxemia'] = (case_stats['o2sat'] < 90).astype(int)

    # Variant Vector String (e.g., "101")
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
# ---------------------------------------------------------
# 5. CONFORMANCE CHECKING (Rule-Based)
# ---------------------------------------------------------


def check_conformance_rules(df, case_col, time_col, act_col):
    print("\n--- Conformance Checking Results ---")

    skip_errors = 0
    insert_errors = 0
    total_cases = df[case_col].nunique()

    # We loop through every single patient case
    for case_id, group in df.groupby(case_col):
        # Get the list of activities for this patient in order
        activities = group[act_col].tolist()

        # --- RULE 1: SKIP ERROR (Triage -> Discharge) ---
        # Logic: IF "Triage" is immediately followed by "Discharge" THEN it is a Skip Error
        if 'Triage in the ED' in activities:
            # Find where Triage happened
            triage_indices = [i for i, x in enumerate(
                activities) if x == 'Triage in the ED']
            for idx in triage_indices:
                # Check if the very next step is Discharge
                if idx + 1 < len(activities):
                    if activities[idx+1] == 'Discharge from the ED':
                        skip_errors += 1
                        break  # Count only once per patient

        # --- RULE 2: INSERT ERROR (Meds before Order) ---
        # Logic: IF "Meds Dispensed" time < "Meds Reconciliation" time THEN it is an Insert Error
        if 'Medicine dispensations' in activities and 'Medicine reconciliation' in activities:
            # Get the timestamp of the first medication given
            first_meds_time = group[group[act_col] ==
                                    'Medicine dispensations'][time_col].min()
            # Get the timestamp of the doctor's reconciliation (order)
            first_recon_time = group[group[act_col] ==
                                     'Medicine reconciliation'][time_col].min()

            # The Critical "If" Statement
            if first_meds_time < first_recon_time:
                insert_errors += 1

    print(f"Total Cases Checked: {total_cases}")
    print(f"Skip Deviations Found: {skip_errors}")
    print(f"Insert Deviations Found: {insert_errors}")

    return total_cases, skip_errors, insert_errors


def plot_deviations(total, skips, inserts):
    labels = ['Total Cases',
              'Insert Errors\n(Early Meds)', 'Skip Errors\n(Triage->Discharge)']
    values = [total, inserts, skips]
    # Grey for total, Red/Orange for errors
    colors = ['lightgray', '#ff9999', '#ffcc99']

    plt.figure(figsize=(8, 6))
    bars = plt.bar(labels, values, color=colors, edgecolor='black')

    plt.ylabel('Number of Cases', fontsize=12, fontweight='bold')
    plt.title('Conformance Checking: Frequency of Deviations',
              fontsize=14, fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Add labels on top of bars
    for bar in bars:
        height = bar.get_height()
        pct = (height / total) * 100
        label = f"{height}\n({pct:.1f}%)" if height < total else f"{height}"
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 label,
                 ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig('deviation_chart.png', dpi=300)
    plt.show()


df, case_col, time_col, act_col = load_and_clean_data('dataset_for_exam.csv')
analyze_performance(df, case_col, time_col)
find_bottlenecks(df, case_col, time_col, act_col)
pattern_based_variant_analysis(df, case_col, time_col)
total, skips, inserts = check_conformance_rules(
    df, case_col, time_col, act_col)
plot_deviations(total, skips, inserts)
