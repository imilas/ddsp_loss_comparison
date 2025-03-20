import marimo

__generated_with = "0.5.2"
app = marimo.App(width="full")


@app.cell
def __():
    import json
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np

    # Load similarity ratings JSON
    SAVE_FILE = "similarity_ratings.json"
    with open(SAVE_FILE, "r") as f:
        ratings = json.load(f)

    def select_by_program_and_create_df(ratings, program_number):
        # Create a list to store function names and Likert scores
        data = []

        # Iterate over the ratings dictionary
        for key, score in ratings.items():
            # Split the key to extract the program number
            parts = key.split("_")
            program_id = int(parts[-2])  # The second to last element is the program number

            # If the program number matches the input, process the key
            if program_id == program_number:
                # Extract the function name
                base_name = "_".join(parts[:2])

                # Remove numbers from the end of function names like "JTFS_1"
                if base_name.startswith("JTFS"):
                    base_name = "JTFS"

                # Append the function name and score to the data list
                data.append([base_name, score])
        
        # Create a DataFrame from the data list
        df = pd.DataFrame(data, columns=["Function", "Score"])
        
        return df

    # Example usage: Select ratings for program 1 and create the DataFrame
    program_number = 0
    df = select_by_program_and_create_df(ratings, program_number)

    # Print the resulting DataFrame
    print(df)

    return (
        SAVE_FILE,
        df,
        f,
        json,
        np,
        pd,
        plt,
        program_number,
        ratings,
        select_by_program_and_create_df,
        sns,
    )


@app.cell
def __(df, np, pd, plt, sns):

    from scipy.stats import mannwhitneyu
    from itertools import combinations

    # Bootstrapping function to compute means and confidence intervals
    def bootstrap_means(scores, n_iterations=1000):
        boot_means = [np.mean(np.random.choice(scores, size=len(scores) // 1, replace=True)) for _ in range(n_iterations)]
        return boot_means

    # Perform bootstrapping for each function
    bootstrapped_data = []
    percentiles = {}
    for function in df['Function'].unique():
        function_scores = df[df['Function'] == function]['Score'].values
        boot_means = bootstrap_means(function_scores)
        bootstrapped_data.extend([(function, mean) for mean in boot_means])

        # Calculate 80% confidence interval (10th and 90th percentiles)
        ci_lower, ci_upper = np.percentile(boot_means, [10, 90])
        percentiles[function] = (ci_lower, ci_upper)

    boot_df = pd.DataFrame(bootstrapped_data, columns=['Function', 'Bootstrapped Mean'])

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
        dist1 = boot_df[boot_df["Function"] == f1]["Bootstrapped Mean"]
        dist2 = boot_df[boot_df["Function"] == f2]["Bootstrapped Mean"]

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
            x=boot_df[boot_df['Function'] == function]['Bootstrapped Mean'],
            color=color_map[function],
            orient='h',
            inner=None,
            split=True
        )

        # Add a star for best performers
        if function in best_performers:
            mean_value = boot_df[boot_df['Function'] == function]['Bootstrapped Mean'].mean()
            plt.text(mean_value, function, '*', fontsize=30, color='yellow', ha='center', va='center')

    # Plot 80% confidence intervals
    for idx, function in enumerate(df['Function'].unique()):
        ci_lower, ci_upper = percentiles[function]
        plt.vlines([ci_lower, ci_upper], ymin=idx - 0.4, ymax=idx + 0.4, color='red', linestyle='--', linewidth=2)

    plt.xlabel("Bootstrapped Mean Likert Score")
    plt.yticks(rotation=90)
    plt.tight_layout()
    plt.show()

    return (
        alpha,
        best_performer,
        best_performer_ci,
        best_performers,
        boot_df,
        boot_means,
        bootstrap_means,
        bootstrapped_data,
        ci,
        ci_lower,
        ci_upper,
        color_map,
        combinations,
        dist1,
        dist2,
        f1,
        f2,
        function,
        function_scores,
        idx,
        mannwhitneyu,
        mean_value,
        p,
        percentiles,
        significant_differences,
        stat,
    )


if __name__ == "__main__":
    app.run()
