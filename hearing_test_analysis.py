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
    import marimo as mo
    import scipy as scipy
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

    return (
        JSON_FILES,
        combinations,
        df,
        json,
        load_ratings_from_files,
        mannwhitneyu,
        mo,
        np,
        os,
        pd,
        plt,
        scipy,
        sns,
        sp,
    )


@app.cell
def __(mo):
    mo.md("# D Responsers Agree? We calculate the spearman coeff")

    return


@app.cell
def __(df, scipy):
    from itertools import product
    matched_responses = [df[df.Responder == x].sort_values("sound_file")["Score"].values for x in df.Responder.unique()]
    print([scipy.stats.spearmanr(x,y) for x,y in product(matched_responses, repeat=2)][1]) # this line is written in case there's more responders
    return matched_responses, product


@app.cell
def __():
    # program_num = 1
    # data = df
    # data = data[data["Program"] == program_num]
    # data["cv"] = data.groupby("Function").cumcount()
    # avg_rank = data.groupby('cv').Score.rank(pct=True).groupby(data.Function).mean()
    # print(avg_rank)
    # test_results = sp.posthoc_conover_friedman(
    #     data,
    #     melted=True,
    #     block_col='cv',
    #     group_col='Function',
    #     y_col='Score',
    # )
    # print(test_results)
    # # sp.sign_plot(test_results)
    # plt.figure(figsize=(10, 2), dpi=100)
    # # plt.title('Critical difference diagram of average score ranks')
    # sp.critical_difference_diagram(avg_rank, test_results)
    # plt.tight_layout()
    # plt.savefig("./plots/critical_diff_%d.png" % (program_num), bbox_inches='tight', pad_inches=0, transparent=True)
    # plt.show()
    return


@app.cell
def __(df, np, pd):
    # Bootstrapping function to compute means and confidence intervals
    def bootstrap_means(scores, n_iterations=1000):
        boot_means = [np.mean(np.random.choice(scores, size=len(scores)//2, replace=True)) for _ in range(n_iterations)]
        return boot_means

    # Perform bootstrapping for each category
    bootstrapped_data = []
    percentiles = {}
    program_num = 2

    for category in df['Function'].unique():
        category_scores = df[ (df["Program"]==program_num) &  (df['Function'] == category) ]['Score'].values
        boot_means = bootstrap_means(category_scores)
        bootstrapped_data.extend([(category, mean, i) for i,mean in enumerate(boot_means)])
        
    boot_df = pd.DataFrame(bootstrapped_data, columns=['Category', 'Bootstrapped Mean',"cv"])


    boot_df
    return (
        boot_df,
        boot_means,
        bootstrap_means,
        bootstrapped_data,
        category,
        category_scores,
        percentiles,
        program_num,
    )


@app.cell
def __(boot_df, np, pd, plt, sns):
    from rpy2.robjects.packages import importr
    from rpy2.robjects import pandas2ri

    # Activate automatic conversion between pandas and R
    pandas2ri.activate()

    # Import ScottKnottESD from R
    sk = importr("ScottKnottESD")

    np.random.seed(42)

    model_performance_df = boot_df.pivot(index='cv', columns='Category', values='Bootstrapped Mean')
    model_performance_df.to_csv("model_performance.csv",index=False)
    # Convert DataFrame to R object and run Scott-Knott ESD
    r_data = pandas2ri.py2rpy(model_performance_df)
    sk_results = sk.sk_esd(r_data)

    # Extract rankings
    sk_ranks = pd.DataFrame({
        "Model": sk_results.rx2("nms")[[x-1 for x in list(sk_results.rx2("ord"))]],
        "Rank": [ str(rank) for rank in list(sk_results.rx2("groups"))]
    })

    # Convert DataFrame to long format for Seaborn
    plot_data = model_performance_df.melt(var_name="Model", value_name="Score")

    # Merge rankings
    plot_data = plot_data.merge(sk_ranks, on="Model")
    plot_data = plot_data.sort_values(["Model"])


    fp = sns.FacetGrid(plot_data,col="Rank",sharey=True,sharex=False,height=4,aspect=0.5,)
    fp.map_dataframe(
        sns.boxplot,
        x="Model",
        y="Score",
    )

    fp.set(xlabel=None,)

    plt.tight_layout()

    plt.show()
    return (
        fp,
        importr,
        model_performance_df,
        pandas2ri,
        plot_data,
        r_data,
        sk,
        sk_ranks,
        sk_results,
    )


@app.cell
def __():
    return


@app.cell
def __(plot_data, plt, program_num, sk_ranks, sns):
    import matplotlib.patches as mpatches

    # Set up the color palette based on unique ranks

    num_ranks = len(plot_data["Rank"].unique())
    colors = ["#00FF00", "#555555",  "#999999",  "#BBBBBB"][0:num_ranks]
    ranks = ["1.0","2.0","3.0","4.0"][0:num_ranks]
    # # Set up the color palette based on unique ranks
    rank_palette = dict(zip(ranks,colors))


    # Create the plot
    plt.figure(figsize=(8, 6))
    ax = sns.boxplot(
        data=plot_data,
        x="Model",
        y="Score",
        hue="Rank",
        palette=rank_palette,
        hue_order=["1.0", "2.0", "3.0", "4.0"][0:num_ranks]  # Ensure legend order
    )

    # Define hatches
    hatches = [["*", "o", "+", "x"][int(float(x))-1] for x in sk_ranks.sort_values("Rank",ascending=True)["Rank"]]
    for hatch, patch in zip(hatches, ax.patches):
        patch.set_hatch(hatch)

    # Create proxy artists for the legend
    legend_handles = []
    for i, (hatch, color) in enumerate(zip(hatches, rank_palette.values())):
        patch = mpatches.Patch(facecolor='none', edgecolor=color, hatch=hatch, label=f'Rank {list(rank_palette.keys())[i]}')
        legend_handles.append(patch)

    # Add custom legend with larger symbols
    plt.xlabel("Model")
    plt.ylabel("Bootstrapped Mean Likert Score")
    plt.xticks(rotation=0)
    plt.legend(handles=legend_handles, title="Rank", loc='upper right', markerscale=2, fontsize=12)
    plt.grid(axis='y')

    plt.tight_layout()

    plt.savefig("./plots/npsk_likert_%d.png" % (program_num), bbox_inches='tight', pad_inches=0, transparent=True)
    plt.show()



    return (
        ax,
        color,
        colors,
        hatch,
        hatches,
        i,
        legend_handles,
        mpatches,
        num_ranks,
        patch,
        rank_palette,
        ranks,
    )


@app.cell
def __(hatches):
    hatches 
    return


if __name__ == "__main__":
    app.run()
