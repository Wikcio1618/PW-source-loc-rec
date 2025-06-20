import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_distance_distributions(
    modify_type, graph_type, loc_type, r, beta, 
    dj_values, data_dir="data", dist_column="dist"
):
    records = []
    for dj in dj_values:
        fname = f"mod_{modify_type}_dj{int(round(dj*100))}_{graph_type}_{loc_type}_r{int(round(r*100))}_beta{int(round(beta*100))}.csv"
        path = os.path.join("../" + data_dir, fname)
        if not os.path.isfile(path):
            print(f"File not found: {path}")
            continue
        df = pd.read_csv(path, skiprows=1)
        if dist_column not in df.columns:
            print(f"Column '{dist_column}' not in file: {path}")
            continue
        distances = df[dist_column]
        distances = distances[(distances != -1) & distances.notna()]
        if distances.empty:
            continue

        plt.figure(figsize=(5, 3))
        plt.hist(distances, bins=range(int(distances.max()) + 2), edgecolor='black', alpha=0.7)
        plt.title(f"Distance distribution for dj={dj}")
        plt.xlabel("Distance to true source")
        plt.ylabel("Frequency")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()

        # Store for stats if needed later
        for val in distances:
            records.append({"dj": dj, "distance": val})

    # Summary stats
    if records:
        full_df = pd.DataFrame(records)
        summary = full_df.groupby("dj")["distance"].describe()
        print("Summary statistics:")
        print(summary)
    else:
        print("No valid data to analyze.")

    
