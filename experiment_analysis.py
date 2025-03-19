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
    import pandas as pd
    from itertools import combinations
    return (
        combinations,
        isnan,
        load_json,
        mo,
        nan,
        np,
        pairwise_tukeyhsd,
        pd,
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
                # Remove sounds
                exp_dictionary.pop("target_sound", None)
                exp_dictionary.pop("output_sound", None)
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
    for ii in range(len(g)):
        for j in range(ii+1, len(g)):
            comparisons.append(f"{lfn_names[ii]} vs {lfn_names[j]}")
            mean_diffs.append(group_means[ii] - group_means[j])

    # Create a plot with error bars for each comparison
    plt.figure(figsize=(8, 6))
    plt.errorbar(comparisons, mean_diffs, yerr=0.1, fmt='o', capsize=5, color='blue')
    plt.axhline(0, linestyle='--', color='gray')
    plt.title("Dunn's Test Pairwise Comparisons (Mean Differences)")
    plt.ylabel("Mean Difference")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    return comparisons, dunn_results, group_means, ii, j, mean_diffs


@app.cell
def __(g, lfn_names, np, pd, performance_measure, plt, sns):
    # Convert data and create a DataFrame from `g` and `lfn_names`
    performance_data = {'Category': [], 'Score': []}
    for category, scores in zip(lfn_names, g):
        performance_data['Category'].extend([category] * len(scores))
        performance_data['Score'].extend(scores)

    df = pd.DataFrame(performance_data)

    # Bootstrapping function to compute means and confidence intervals
    def bootstrap_means(scores, n_iterations=1000):
        boot_means = [np.mean(np.random.choice(scores, size=len(scores) // 2, replace=True)) for _ in range(n_iterations)]
        return boot_means

    # Perform bootstrapping for each category
    bootstrapped_data = []
    percentiles = {}
    for category in df['Category'].unique():
        category_scores = df[df['Category'] == category]['Score'].values
        boot_means = bootstrap_means(category_scores)
        bootstrapped_data.extend([(category, mean) for mean in boot_means])

        # Calculate 95% confidence interval
        ci_lower, ci_upper = np.percentile(boot_means, [2.5, 97.5])
        percentiles[category] = (ci_lower, ci_upper)

    boot_df = pd.DataFrame(bootstrapped_data, columns=['Category', 'Bootstrapped Mean'])

    # Initialize variables to find the best performer
    best_performer = None
    lowest_ci_upper = float('inf')

    # Identify the lowest upper confidence interval
    for category, (ci_lower, ci_upper) in percentiles.items():
        if ci_upper < lowest_ci_upper:
            lowest_ci_upper = ci_upper
            best_performer = category

    # Gather best performers, starting with the identified best performer
    best_performers = [best_performer]

    # Check for overlaps with other categories
    for category, (ci_lower, ci_upper) in percentiles.items():
        if category != best_performer:  # Avoid comparing with itself
            best_performer_ci_lower = percentiles[best_performer][0]
            best_performer_ci_upper = percentiles[best_performer][1]
            # Check if the confidence intervals overlap
            if ci_lower <= best_performer_ci_upper and ci_upper >= best_performer_ci_lower:
                best_performers.append(category)

    # Output the results
    print("Best Performers:", best_performers)

    # Define colors for the violin plots
    color_map = {category: 'blue' if category in best_performers else 'white' for category in df['Category'].unique()}

    # Plot
    plt.figure(figsize=(6, 6))
    for category in df['Category'].unique():
        sns.violinplot(
            y=[category] * len(boot_df[boot_df['Category'] == category]),
            x=boot_df[boot_df['Category'] == category]['Bootstrapped Mean'],
            color=color_map[category],
            orient='h',
            inner=None,
            split=True
        )

        # Add a star for best performers in the middle
        if category in best_performers:
            mean_value = boot_df[boot_df['Category'] == category]['Bootstrapped Mean'].mean()
            plt.text(mean_value, category, '*', fontsize=30, color='yellow', ha='center', va='center')  # Make star bigger and yellow


    # Plotting the 95% confidence intervals
    for idx, category in enumerate(df['Category'].unique()):
        ci_lower, ci_upper = percentiles[category]

        # Plot dashed vertical lines for the confidence interval
        plt.vlines([ci_lower, ci_upper], ymin=idx - 0.4, ymax=idx + 0.4, color='red', linestyle='--', linewidth=2)

    plt.xlabel("Bootstrapped Mean %s" % performance_measure)
    plt.yticks(rotation=90)
    plt.tight_layout()
    # plt.savefig("./plots/p%d_%s.png" % (program_num, performance_measure), bbox_inches='tight', pad_inches=0, transparent=True)

    plt.show()
    return (
        best_performer,
        best_performer_ci_lower,
        best_performer_ci_upper,
        best_performers,
        boot_df,
        boot_means,
        bootstrap_means,
        bootstrapped_data,
        category,
        category_scores,
        ci_lower,
        ci_upper,
        color_map,
        df,
        idx,
        lowest_ci_upper,
        mean_value,
        percentiles,
        performance_data,
        scores,
    )


if __name__ == "__main__":
    app.run()
