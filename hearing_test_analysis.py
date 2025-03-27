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
    # # data = boot_df
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
def __(df, pd):

    def bootstrap_means(df, N=1000):
        bootstrapped_dfs = []
        
        # Group by 'Program'
        for program, group in df.groupby("Program"):
            boot_means = [
                group["Score"].sample(frac=1, replace=True).mean()
                for _ in range(N)
            ]
            
            boot_df = pd.DataFrame({
                "Program": program,
                "Bootstrapped Mean": boot_means
            })
            
            bootstrapped_dfs.append(boot_df)

        return pd.concat(bootstrapped_dfs, ignore_index=True)

    # Example usage
    bootstrapped_df = bootstrap_means(df, N=1000)
    print(bootstrapped_df)

    return bootstrap_means, bootstrapped_df


@app.cell
def __(data, np, pd, plt, program_num, sns):
    from rpy2.robjects.packages import importr
    from rpy2.robjects import pandas2ri

    # Activate automatic conversion between pandas and R
    pandas2ri.activate()

    # Import ScottKnottESD from R
    sk = importr("ScottKnottESD")

    np.random.seed(42)

    model_performance_df = data.pivot(index="cv",columns="Function",values="Score")
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
    plot_data = model_performance_df.melt(var_name="Model", value_name="Likert Score")

    # Merge rankings
    plot_data = plot_data.merge(sk_ranks, on="Model")
    plot_data = plot_data.sort_values(["Rank","Model"])
    # Set color palette for ranks
    unique_ranks = sorted(plot_data["Rank"].unique())
    rank_palette = dict(zip(unique_ranks, sns.color_palette("Blues", len(unique_ranks))))

    # Create the boxplot
    # plt.figure(figsize=(10, 6))

    fp = sns.FacetGrid(plot_data,col="Rank",sharey=True,sharex=False,height=4,aspect=0.5,)
    fp.map_dataframe(
        sns.boxplot,
        x="Model",
        y="Likert Score",
        # order=["Rank-1", "Rank-2", "Rank-3", "Rank-4"],  # Adjust to match your data
        palette="Blues"
        
      
    )

    fp.set(xlabel=None,)

    plt.tight_layout()
    plt.savefig("./plots/npsk_hearing_%d.png" % (program_num), bbox_inches='tight', pad_inches=0, transparent=True)
    plt.show()
    return (
        fp,
        importr,
        model_performance_df,
        pandas2ri,
        plot_data,
        r_data,
        rank_palette,
        sk,
        sk_ranks,
        sk_results,
        unique_ranks,
    )


@app.cell
def __(plot_data, plt, sns):
    # Define colors: Rank 1 gets blue, others get gray
    palette = {model: "blue" if rank == "1.0" else "lightgray"
               for model, rank in zip(plot_data["Model"], plot_data["Rank"])}

    # Create the violin plot
    sns.boxplot(
        data=plot_data,
        hue="Model",
        y="Likert Score",
        palette=palette,  
        legend=False,
    )


    # Rotate x-axis labels for readability
    plt.xticks(rotation=45)
    plt.xlabel("Model")
    plt.ylabel("Inverse Loss")
    plt.title("Violin Plot with Rank-1 Models Highlighted")

    plt.tight_layout()
    plt.show()
    return palette,


@app.cell
def __():
    return


@app.cell
def __(palette):
    palette
    return


@app.cell
def __(plot_data):
    plot_data
    return


@app.cell
def __(plot_data):
    plot_data
    return


if __name__ == "__main__":
    app.run()
