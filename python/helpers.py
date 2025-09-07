import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math

def get_localisation_data_from_file(path:str):
    with open(path, "r") as file:
        lines = file.readlines()[2:]
        prec = [float(line.split(',')[1]) for line in lines]
        precs.append(np.mean(prec))
        errs.append(np.std(prec) / math.sqrt(len(prec)))

        precs = np.array(precs)
        errs = np.array(errs)
    
    return precs, errs

def reconstruction_plots(
        graph_type, 
        reconstruct_type, 
        loc_type, 
        r, 
        beta, 
        dj_list, 
        labels
    ):
    
    plt.style.use('ggplot')  # or 'classic', 'bmh', 'fivethirtyeight'

    plt.rcParams.update({
        'font.size': 16,
        'axes.labelsize': 18,
        'axes.titlesize': 20,
        'legend.fontsize': 14,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'axes.edgecolor': 'black',
        'axes.linewidth': 1.2,
        'grid.linestyle': '--',
        'grid.linewidth': 0.7,
        'grid.alpha': 0.6,
        'axes.grid': True,
        'lines.linewidth': 2.5,
        'lines.markersize': 8,
    })

    k_groups = {}
    for dj in dj_list:
        filename =f"rec_{reconstruct_type}_dj{int(dj*100)}_{graph_type}_{loc_type}_r{int(r*100)}_beta{int(beta*100)}.csv"
        path = os.path.join("../data", filename)
        df = pd.read_csv(path, engine="python", skiprows=1)
        for i, (k, prec, err) in enumerate(zip(df["k"], df["prec"], df["prec_err"])):
            k_groups.setdefault(i, {"dj_vals": [], "prec": [], "prec_err": []})
            k_groups[i]["dj_vals"].append(dj)
            k_groups[i]["prec"].append(prec)
            k_groups[i]["prec_err"].append(err)

    colors = plt.cm.plasma(np.linspace(0, 1, len(labels)))

    fig, ax = plt.subplots(figsize=(8, 5))
    plot_idx = 0

    for i, (group_id, data) in enumerate(k_groups.items()):
        dj_vals = np.array(data["dj_vals"])
        precs = np.array(data["prec"])
        errs = np.array(data["prec_err"])

        ax.plot(dj_vals, precs, color=colors[plot_idx], label=labels[plot_idx])
        ax.fill_between(dj_vals, precs - errs, precs + errs, color=colors[plot_idx], alpha=0.25)

        plot_idx += 1

    ax.set_xlabel("dj (odległość Jaccarda)")
    ax.set_ylabel("Precyzja")
    ax.set_title(f"Precyzja {reconstruct_type.upper()} + {loc_type} + {graph_type}")
    ax.legend(title="Ilość zrekonstruowanych linków", loc="upper right", frameon=False)
    ax.grid(True, linestyle='--', linewidth=0.7, alpha=0.6)

    plt.tight_layout()
    plt.show()


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
