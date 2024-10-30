import marimo

__generated_with = "0.5.2"
app = marimo.App(width="full")


@app.cell
def __():
    import marimo as mo
    from helpers.experiment_scripts import load_json
    import numpy as np
    from math import nan, isnan
    import scikit_posthocs as sp
    from scipy import stats
    import matplotlib.pyplot as plt
    import seaborn as sns
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    return (
        isnan,
        load_json,
        mo,
        nan,
        np,
        pairwise_tukeyhsd,
        plt,
        sns,
        sp,
        stats,
    )


@app.cell
def __():
    import os
    import pickle

    # Directory containing pickle files
    directory = "./results/"

    # List to store loaded dictionaries
    d = []

    # Iterate over all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".pkl"):
            file_path = os.path.join(directory, filename)
            
            # Load each pickle file and append the data to the list
            with open(file_path, "rb") as file:
                exp_dictionary= pickle.load(file)
                d.append(exp_dictionary)

    # d = [e for e in d if "Multi_Spec" in e]
    len(d)
    return (
        d,
        directory,
        exp_dictionary,
        file,
        file_path,
        filename,
        os,
        pickle,
    )


@app.cell
def __():
    lfn_names = ['DTW_Onset','L1_Spec' ,'SIMSE_Spec', 'JTFS']
    program_num = 0
    performance_measure = "MSS"
    # performance_measure = "P-Loss"
    return lfn_names, performance_measure, program_num


@app.cell
def __(d, isnan, lfn_names, np, performance_measure, program_num):
    def get_p_error(e):
        """calculate p-loss given an experiment dictionary"""
        p1 = np.array(list(e["true_params"]["params"].values()))
        p2 = np.array(list(e["norm_params"].values()))[:,-1]
        return np.sqrt(np.sum((p1-p2)**2))

    def filter_experiments(d,loss_fn_name,prog_num):
        return [x for x in d if x["loss"]==loss_fn_name and x["program_id"]==prog_num]

    if performance_measure == "MSS":
        g = [[x["Multi_Spec"] for x in filter_experiments(d,lfn_name,program_num)] for lfn_name in lfn_names]
    else:
        g = [[get_p_error(x) for x in filter_experiments(d,lfn_name,program_num)] for lfn_name in lfn_names]
        g = [[2 if isnan(i) else i for i in j ] for j in g]

    g = [[float(element) for element in sublist] for sublist in g] # to remove jax types from floats

    [len(e) for e in g]
    return filter_experiments, g, get_p_error


@app.cell
def __(g, lfn_names, np, pairwise_tukeyhsd, plt, stats):
    data = g

    # Perform One-Way ANOVA to test if the means are different
    # F, p_value = stats.f_oneway(*g)
    F, p_value = stats.kruskal(*g)

    print(f"Critical value: {F}, p-value: {p_value}")

    # Check if the p-value is less than 0.05 (95% confidence)
    if p_value < 0.05:
        print("There is a significant difference between the arrays.")

        # Flatten the arrays for pairwise comparison (Tukey's HSD test)
        combined_data = np.concatenate(g)
        group_labels = [lfn_names[i] for i in range(len(lfn_names)) for _ in range(len(g[i]))]

        # Perform Tukey's HSD post-hoc test to find which array is different
        tukey_result = pairwise_tukeyhsd(combined_data, group_labels, alpha=0.05)
        print(tukey_result)

        # Plot the confidence intervals
        tukey_result.plot_simultaneous()
        plt.show()

    else:
        print("No significant difference between the arrays.")
    return F, combined_data, data, group_labels, p_value, tukey_result


@app.cell
def __(g, lfn_names, np, plt, sp):
    # Perform Dunn's test
    dunn_results = sp.posthoc_dunn(g, p_adjust='bonferroni')

    print(dunn_results)

    # Calculate the group means
    group_means = [np.mean(group) for group in g]

    # Plot pairwise comparisons with mean differences
    comparisons = []
    mean_diffs = []
    for i in range(len(g)):
        for j in range(i+1, len(g)):
            comparisons.append(f"{lfn_names[i]} vs {lfn_names[j]}")
            mean_diffs.append(group_means[i] - group_means[j])

    # Create a plot with error bars for each comparison
    plt.figure(figsize=(8, 6))
    plt.errorbar(comparisons, mean_diffs, yerr=0.1, fmt='o', capsize=5, color='blue')
    plt.axhline(0, linestyle='--', color='gray')
    plt.title("Dunn's Test Pairwise Comparisons (Mean Differences)")
    plt.ylabel("Mean Difference")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    return comparisons, dunn_results, group_means, i, j, mean_diffs


@app.cell
def __(g, lfn_names, np, performance_measure, plt, program_num, sns):
    import pandas as pd
    from itertools import combinations
    from matplotlib.colors import LinearSegmentedColormap
    # Create DataFrame from `g` and `lfn_names`
    performance_data = {'Category': [], 'Score': []}
    for category, scores in zip(lfn_names, g):
        performance_data['Category'].extend([category] * len(scores))
        performance_data['Score'].extend(scores)

    df = pd.DataFrame(performance_data)
    # Bootstrapping function
    def bootstrap_means(scores, n_iterations=1000):
        boot_means = []
        for _ in range(n_iterations):
            sample = np.random.choice(scores, size=len(scores)//4, replace=True)
            boot_means.append(np.mean(sample))
        return boot_means

    # Perform bootstrapping for each category
    bootstrapped_data = []
    percentiles = {}
    for category in df['Category'].unique():
        category_scores = df[df['Category'] == category]['Score'].values
        boot_means = bootstrap_means(category_scores)
        bootstrapped_data.extend([(category, mean) for mean in boot_means])
        
        # Calculate 25th and 75th percentiles
        p25, p75 = np.percentile(boot_means, [25, 75])
        percentiles[category] = (p25, p75)

    # Convert to DataFrame for easy plotting
    boot_df = pd.DataFrame(bootstrapped_data, columns=['Category', 'Bootstrapped Mean'])

    # Determine overlapping groups
    overlapping_groups = {}
    for (cat1, (p25_1, p75_1)), (cat2, (p25_2, p75_2)) in combinations(percentiles.items(), 2):
        if (p25_1 <= p75_2 and p75_1 >= p25_2):  # Check for overlap
            overlapping_groups.setdefault(cat1, set()).add(cat2)
            overlapping_groups.setdefault(cat2, set()).add(cat1)

    # Assign colors to groups based on overlap
    distinct_colors = [
        "#A50000",  # Dark Red
        "#005EB8",  # Dark Blue
        "#007A33",  # Dark Green
        "#D95F0E"   # Dark Orange
    ]

    color_map = {}
    assigned_colors = {}
    current_color_idx = 0

    # Sort by performance (lower is better)
    sorted_categories = sorted(percentiles, key=lambda x: percentiles[x][0])

    # Assign colors based on rank
    for category in sorted_categories:
        if category not in assigned_colors:
            # Assign a new color to this category
            color_map[category] = distinct_colors[current_color_idx]
            assigned_colors[category] = distinct_colors[current_color_idx]
            
            # Assign the same color to all overlapping groups
            if category in overlapping_groups:
                for overlapping_category in overlapping_groups[category]:
                    color_map[overlapping_category] = distinct_colors[current_color_idx]
                    assigned_colors[overlapping_category] = distinct_colors[current_color_idx]
            
            current_color_idx += 1  # Move to the next color

    # Calculate rankings (with ties)
    rankings = {}
    rank_map = {}  # Map to store ranks and handle ties
    rank = 1
    last_p25 = None

    # Adjust ranks for categories that share the same color
    for category in sorted_categories:
        p25 = percentiles[category][0]
        if last_p25 is None or p25 > last_p25:
            rankings[category] = rank
            rank_map[rank] = [category]  # Initialize a list for categories sharing this rank
        else:
            rankings[category] = rank  # Assign same rank for ties
            rank_map[rank].append(category)  # Append to the list of shared rank
        last_p25 = p25
        rank += 1

    # Adjust ranks to ensure overlapping groups have the same rank
    for group in overlapping_groups:
        # Get the lowest rank for this group
        min_rank = min(rankings[category] for category in overlapping_groups[group] | {group})
        for overlapping_category in overlapping_groups[group] | {group}:
            rankings[overlapping_category] = min_rank

    # Plot
    plt.figure(figsize=(6, 6))

    for category in df['Category'].unique():
        # Create a half violin by multiplying the bootstrapped means by 1 (for right side)
        sns.violinplot(
            y=[category] * len(boot_df[boot_df['Category'] == category]),
            x=boot_df[boot_df['Category'] == category]['Bootstrapped Mean'],
            color=color_map[category],
            orient='h',  # Horizontal orientation
            inner='quart',
            split="false",
            scale='area'  # Ensure the area is correctly scaled
        )

    # sns.boxplot(y='Category', x='Bootstrapped Mean', data=boot_df, width=0.1, color="black")

    # Highlight the 25th and 75th percentiles
    for category in df['Category'].unique():
        category_means = boot_df[boot_df['Category'] == category]['Bootstrapped Mean']
        p25, p75 = np.percentile(category_means, [25, 75])
        plt.scatter([p25, p75], [category] * 2, color='red', marker='o', label='25th/75th Percentile' if category == 'A' else "")

    # Create a legend mapping colors to performance
    handles = []
    for category, color in color_map.items():
        handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10))
    plt.legend(handles, [f"{category} (Rank {rankings[category]})" for category in color_map.keys()], title="Performance Rank")

    # plt.title("Bootstrapped Means")
    plt.xlabel("Bootstrapped Mean")  # Add x-label for clarity
    # plt.ylabel("Loss Function")  # Add y-label for clarity
    plt.tight_layout()
    plt.savefig("./plots/p%d_%s.png"%(program_num,performance_measure),bbox_inches='tight', pad_inches=0, transparent=True)
    plt.show()
    return (
        LinearSegmentedColormap,
        assigned_colors,
        boot_df,
        boot_means,
        bootstrap_means,
        bootstrapped_data,
        cat1,
        cat2,
        category,
        category_means,
        category_scores,
        color,
        color_map,
        combinations,
        current_color_idx,
        df,
        distinct_colors,
        group,
        handles,
        last_p25,
        min_rank,
        overlapping_category,
        overlapping_groups,
        p25,
        p25_1,
        p25_2,
        p75,
        p75_1,
        p75_2,
        pd,
        percentiles,
        performance_data,
        rank,
        rank_map,
        rankings,
        scores,
        sorted_categories,
    )


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
