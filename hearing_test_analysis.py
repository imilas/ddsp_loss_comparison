import marimo

__generated_with = "0.5.2"
app = marimo.App(width="full")


@app.cell
def __():
    import json
    import pandas as pd
    import numpy as np
    import os
    import scikit_posthocs as sp
    import matplotlib.pyplot as plt
    from itertools import combinations
    from scipy.stats import mannwhitneyu
    import seaborn as sns


    # List of similarity rating JSON files
    JSON_FILES = ["survey_results/similarity_ratings_a1.json", "survey_results/similarity_ratings_a2.json"]

    def load_ratings_from_files(json_files):
        """Load similarity ratings from multiple JSON files and combine them into a DataFrame."""
        data = []

        for responder_id, file in enumerate(json_files, start=1):
            if os.path.exists(file):
                with open(file, "r") as f:
                    ratings = json.load(f)

                # Process each entry in the JSON data
                for key, score in ratings.items():
                    parts = key.split("_")
                    program_id = int(parts[-2])  # Extract program number
                    function_name = "_".join(parts[:-2])  # Keep full function name

                    data.append([function_name, program_id, score, responder_id,key])

        # Create DataFrame
        df = pd.DataFrame(data, columns=["Function", "Program", "Score", "Responder","sound_file"])
        return df

    # Load data
    df = load_ratings_from_files(JSON_FILES)

    # Print the resulting DataFrame
    print(df)

    # Bootstrapping function to compute means and confidence intervals
    def bootstrap_means(scores, n_iterations=100):
        boot_means = [np.mean(np.random.choice(scores, size=len(scores), replace=True)) for _ in range(n_iterations)]
        return boot_means

    return (
        JSON_FILES,
        bootstrap_means,
        combinations,
        df,
        json,
        load_ratings_from_files,
        mannwhitneyu,
        np,
        os,
        pd,
        plt,
        sns,
        sp,
    )


@app.cell
def __(bootstrap_means, df, np, pd):
    # Perform bootstrapping for each function
    bootstrapped_data = []
    percentiles = {}
    for f in df['Function'].unique():
        function_scores = df[df['Function'] == f]['Score'].values
        boot_means = bootstrap_means(function_scores)
        bootstrapped_data.extend([(i,f, mean) for i,mean in enumerate(boot_means)])

        # Calculate 80% confidence interval (10th and 90th percentiles)
        ci_lower, ci_upper = np.percentile(boot_means, [10, 90])
        percentiles[f] = (ci_lower, ci_upper)
    boot_df = pd.DataFrame(bootstrapped_data, columns=["cv",'Function', 'Score'])
    print(boot_df)

    return (
        boot_df,
        boot_means,
        bootstrapped_data,
        ci_lower,
        ci_upper,
        f,
        function_scores,
        percentiles,
    )


@app.cell
def __(df, plt, sp):
    # data = boot_df
    data = df
    data = data[data["Program"] == 0]
    data["cv"] = data.groupby("Function").cumcount()
    avg_rank = data.groupby('cv').Score.rank(pct=True).groupby(data.Function).mean()
    print(avg_rank)
    test_results = sp.posthoc_conover_friedman(
        data,
        melted=True,
        block_col='cv',
        group_col='Function',
        y_col='Score',
    )
    print(test_results)
    # sp.sign_plot(test_results)
    plt.figure(figsize=(10, 2), dpi=100)
    plt.title('Critical difference diagram of average score ranks')
    sp.critical_difference_diagram(avg_rank, test_results)
    plt.show()
    return avg_rank, data, test_results


@app.cell
def __(boot_df, combinations, df, mannwhitneyu, percentiles, plt, sns):
    # Identify the best performer based on the highest **upper** confidence bound
    best_performer = max(percentiles, key=lambda f: percentiles[f][1])  # Highest upper bound
    best_performers = [best_performer]

    # Check for overlaps with other functions
    best_performer_ci = percentiles[best_performer]
    for function, ci in percentiles.items():
        if function != best_performer:
            if ci[0] <= best_performer_ci[1] and ci[1] >= best_performer_ci[0]:
                best_performers.append(function)

    # Output results
    print("Best Performers:", best_performers)

    # Mann-Whitney U-test: Check if one function is significantly better than another
    alpha = 0.05  # Significance level
    significant_differences = []

    for f1, f2 in combinations(df["Function"].unique(), 2):
        dist1 = boot_df[boot_df["Function"] == f1]["Score"]
        dist2 = boot_df[boot_df["Function"] == f2]["Score"]

        stat, p = mannwhitneyu(dist1, dist2, alternative='greater')  # Test if f1 > f2

        if p < alpha:
            significant_differences.append((f1, f2, p))
            print(f"{f1} is significantly better than {f2} (p={p:.5f})")

    if not significant_differences:
        print("No statistically significant differences found.")

    # Define colors for violin plots
    color_map = {func: 'blue' if func in best_performers else 'white' for func in df['Function'].unique()}

    # Plot
    plt.figure(figsize=(6, 6))
    for function in df['Function'].unique():
        sns.violinplot(
            y=[function] * len(boot_df[boot_df['Function'] == function]),
            x=boot_df[boot_df['Function'] == function]['Score'],
            color=color_map[function],
            orient='h',
            inner=None,
            split=True
        )

        # Add a star for best performers
        if function in best_performers:
            mean_value = boot_df[boot_df['Function'] == function]['Score'].mean()
            plt.text(mean_value, function, '*', fontsize=30, color='yellow', ha='center', va='center')

    # # Plot 80% confidence intervals
    # for idx, function in enumerate(df['Function'].unique()):
    #     ci_lower, ci_upper = percentiles[function]
    #     plt.vlines([ci_lower, ci_upper], ymin=idx - 0.4, ymax=idx + 0.4, color='red', linestyle='--', linewidth=2)

    plt.xlabel("Score Likert Score")
    plt.yticks(rotation=90)
    plt.tight_layout()
    plt.show()
    return (
        alpha,
        best_performer,
        best_performer_ci,
        best_performers,
        ci,
        color_map,
        dist1,
        dist2,
        f1,
        f2,
        function,
        mean_value,
        p,
        significant_differences,
        stat,
    )


if __name__ == "__main__":
    app.run()
