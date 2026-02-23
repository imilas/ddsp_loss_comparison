import marimo

__generated_with = "0.13.6"
app = marimo.App(width="full")


@app.cell
def _():
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
    # JSON_FILES = ["survey_results/in_domain/similarity_ratings_a1.json", "survey_results/in_domain/similarity_ratings_a2.json"]
    JSON_FILES = ["survey_results/out_of_domain/amir_similarity_ratings.json","survey_results/out_of_domain/abram_similarity_ratings.json"]

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
    return df, mo, np, pd, plt, scipy, sns


@app.cell
def _(mo):
    mo.md("""# D Responsers Agree? We calculate the spearman coeff""")
    return


@app.cell
def _(df, scipy):
    # could do spearman or kendalltau
    def match_and_spearman(df,group_name="all"):
        matched_responses = [df[df.Responder == x].sort_values("sound_file")["Score"].values for x in df.Responder.unique()]
        print("spearman for %s"%group_name, scipy.stats.spearmanr(matched_responses[1],matched_responses[0]))
    
    match_and_spearman(df)
    # per program spearman r
    df.groupby(["Program"]).apply(lambda x: match_and_spearman(x,x.name))
    return


@app.cell
def _(mo):
    mo.md("""# Kruskal wallis""")
    return


@app.cell
def _(df):
    df
    return


@app.cell
def _(df):
    from scipy.stats import kruskal
    def kruskal_by_loss_group(df, value_column):
        """
        Perform Kruskal-Wallis test on `value_column` for each group in the 'loss' column.

        Parameters:
            df (pd.DataFrame): The input DataFrame with at least 'loss' and `value_column`.
            value_column (str): The name of the column on which to apply the test.

        Returns:
            H-statistic, p-value
        """
        grouped_values = [
            group[value_column].values
            for _, group in df.groupby("Function")
        ]

        stat, p_value = kruskal(*grouped_values)
        return stat, p_value
    for pid in df.Program.unique():
        print("pid %d"%pid,kruskal_by_loss_group(df[df.Program==pid],"Score"))
    return


@app.cell
def _():
    # # program_num = 1
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
def _(df, np, pd):
    # Bootstrapping function to compute means and confidence intervals


    def bootstrap_means(scores, n_iterations=1000):
        boot_means = [np.mean(np.random.choice(scores, size=len(scores), replace=True)) for _ in range(n_iterations)]
        return boot_means

    # Perform bootstrapping for each category
    bootstrapped_data = []
    percentiles = {}
    # 0: am_non-overlapping freqs
    # 1: am_sine_target_saw_imitate
    # 2: am_saw_imitate_sine_target
    # 3: bp_noise_target_saw_imitate
    # 4: chirp_delayed_pitchbend
    # 5: chirp pulsating
    # 6: chirp normal

    program_num = 6
    for category in df['Function'].unique():
        category_scores = df[ (df["Program"]==program_num) &  (df['Function'] == category) ]['Score'].values
        boot_means = bootstrap_means(category_scores)
        bootstrapped_data.extend([(category, mean, i) for i,mean in enumerate(boot_means)])

    boot_df = pd.DataFrame(bootstrapped_data, columns=['Category', 'Bootstrapped Mean',"cv"])


    boot_df
    return boot_df, program_num


@app.cell
def _(boot_df, np, pd, plt, sns):
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
        "Rank": [ int(rank) for rank in list(sk_results.rx2("groups"))]
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
    return (plot_data,)


@app.cell
def _(plot_data):
    plot_data
    return


@app.cell
def _(plot_data, program_num):
    import plotly.graph_objects as go
    import plotly.io as pio
    pio.templates.default = "plotly_white"
    # Sort models alphabetically
    model_order = sorted(plot_data["Model"].unique())

    # Rank-to-color mapping
    rank_palette = {
        1: "#70FF70",
        2: "#858585",
        3: "#454545",
        4: "#000000"
    }

    # Create the figure manually, one half-violin per model
    fig = go.Figure()

    for model in model_order:
        sub_df = plot_data[plot_data["Model"] == model]
        rank = int(sub_df["Rank"].iloc[0])
        color = rank_palette[rank]

        fig.add_trace(go.Violin(
            x=sub_df["Score"],
            y=[model] * len(sub_df),
            orientation="h",
            name=model,
            line_color=color,
            fillcolor=color,
            box_visible=False,
            meanline_visible=False,
            side="positive",  # <-- half violin
            points="outliers",
            marker=dict(color=color, outliercolor=color, line=dict(color=color)),
            width=0.6
        ))

    # Minimal layout
    fig.update_layout(
        xaxis=dict(
            side="top",  # <- move title to the top
            tickfont=dict(
                family="JetBrainsMono Nerd Font Mono",  # Bold font
                size=16,
                color="black"
            )

        ),
            yaxis=dict(
            showticklabels=False  # hides the y-axis tick labels (model names)
        ),
        showlegend=False,
        yaxis_title=None,
        title=None,

        margin=dict(l=0, r=0, t=0, b=0), 
        width=250,
        height=175
    )

    # Save the figure as a PDF
    fig.write_image("./plots/npsk_ood_likert_%d.png" % (program_num), engine="kaleido",scale=5)
    fig.show()
    return


@app.cell
def _():
    # from PIL import Image
    # # import matplotlib.pyplot as plt

    # # Open the image
    # img = Image.open('plots/npsk_likert_2.png')

    # # Convert to greyscale
    # gray_img = img.convert('L')

    # img.show()
    # gray_img.show()
    return


if __name__ == "__main__":
    app.run()
