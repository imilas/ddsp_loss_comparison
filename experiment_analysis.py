import marimo

__generated_with = "0.5.2"
app = marimo.App(width="full")


@app.cell
def __():
    import marimo as mo
    from helpers.experiment_scripts import load_json
    import numpy as np
    return load_json, mo, np


@app.cell
def __(load_json):
    d = load_json("results/experiments.json")
    return d,


@app.cell
def __():
    lfn_names = ['DTW_Onset','L1_Spec' ,'SIMSE_Spec', 'JTFS']
    return lfn_names,


@app.cell
def __(d, lfn_names, np):
    from math import nan, isnan
    def get_p_error(e):
        p1 = np.array(list(e["true_params"]["params"].values()))
        p2 = np.array(list(e["norm_params"].values()))[:,-1]
        return np.sqrt(np.sum((p1-p2)**2))
    def filter_experiments(d,loss_fn_name,prog_num):
        return [x for x in d if x["loss"]==loss_fn_name and x["program_id"]==prog_num]

    g = [[get_p_error(x) for x in filter_experiments(d,lfn_name,0)] for lfn_name in lfn_names]
    g = [[2 if isnan(i) else i for i in j ] for j in g]
    return filter_experiments, g, get_p_error, isnan, nan


@app.cell
def __(g, lfn_names, np):
    from scipy import stats
    import matplotlib.pyplot as plt
    import seaborn as sns
    from statsmodels.stats.multicomp import pairwise_tukeyhsd

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
    return (
        F,
        combined_data,
        data,
        group_labels,
        p_value,
        pairwise_tukeyhsd,
        plt,
        sns,
        stats,
        tukey_result,
    )


@app.cell
def __(g):
    import scikit_posthocs as sp

    # Perform Dunn's test for multiple comparisons
    dunn_result = sp.posthoc_dunn(g, p_adjust='bonferroni')

    print(dunn_result)
    return dunn_result, sp


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


if __name__ == "__main__":
    app.run()
