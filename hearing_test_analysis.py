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

    program_num = 3
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
        program_num,
        sns,
        sp,
    )


@app.cell
def __(df, plt, program_num, sp):
    # data = boot_df
    data = df
    data = data[data["Program"] == program_num]
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
    # plt.title('Critical difference diagram of average score ranks')
    sp.critical_difference_diagram(avg_rank, test_results)
    plt.tight_layout()
    plt.savefig("./plots/critical_diff_%d.png" % (program_num), bbox_inches='tight', pad_inches=0, transparent=True)
    plt.show()
    return avg_rank, data, test_results


if __name__ == "__main__":
    app.run()
