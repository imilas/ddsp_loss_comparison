import marimo

__generated_with = "0.13.6"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    from helper_funcs.experiment_scripts import load_json
    import numpy as np
    from math import nan, isnan
    import scikit_posthocs as sp
    from scipy import stats
    import matplotlib.pyplot as plt
    import seaborn as sns
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    import pandas as pd
    from itertools import combinations
    return isnan, np, pd, plt, sns


@app.cell
def _():
    import os
    import pickle

    # Directory containing pickle files
    directory = "./results/in_domain"

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
    return (d,)


@app.cell
def _():
    lfn_names = ['DTW_Onset','L1_Spec' ,'SIMSE_Spec', 'JTFS']
    program_num = 1
    performance_measure = "MSS"
    # performance_measure = "P-Loss"
    return lfn_names, performance_measure, program_num


@app.cell
def _():
    return


@app.cell
def _(d, isnan, lfn_names, np, performance_measure, program_num):
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
    return g, get_p_error


@app.cell
def _(d, get_p_error, pd):
    # Kruskall wallic per program
    from operator import itemgetter
    columns = ['program_id', 'loss', 'Multi_Spec']
    def get_mss_ploss(x):
        return *itemgetter(*columns)(x),get_p_error(x)
    all_results_array = [get_mss_ploss(x) for x in d]
    evals_df = pd.DataFrame(all_results_array,columns=columns+["P_Loss"])
    evals_df
    return (evals_df,)


@app.cell
def _(d):
    d[0]
    return


@app.cell
def _(evals_df):
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
            for _, group in df.groupby("loss")
        ]

        stat, p_value = kruskal(*grouped_values)
        return stat, p_value

    for pid in evals_df["program_id"].unique():
        for eval_method in ["Multi_Spec","P_Loss"]:
            print("program %d evaluation method %s"%(pid,eval_method),kruskal_by_loss_group(evals_df[evals_df["program_id"]==pid],eval_method))
    return


@app.cell
def _(g, lfn_names, np, pd):
    # Convert data and create a DataFrame from `g` and `lfn_names`
    performance_data = {'Category': [], 'Score': []}
    for category, scores in zip(lfn_names, g):
        performance_data['Category'].extend([category] * len(scores))
        performance_data['Score'].extend(scores)

    df = pd.DataFrame(performance_data)

    # Bootstrapping function to compute means and confidence intervals
    def bootstrap_means(scores, n_iterations=1000):
        boot_means = [1/np.mean(np.random.choice(scores, size=len(scores), replace=True)) for _ in range(n_iterations)]
        return boot_means

    # Perform bootstrapping for each category
    bootstrapped_data = []
    percentiles = {}
    for category in df['Category'].unique():
        category_scores = df[df['Category'] == category]['Score'].values
        boot_means = bootstrap_means(category_scores)
        bootstrapped_data.extend([(category, mean, i) for i,mean in enumerate(boot_means)])


        # Calculate 95% confidence interval
        ci_lower, ci_upper = np.percentile(boot_means, [5, 95])
        percentiles[category] = (ci_lower, ci_upper)

    boot_df = pd.DataFrame(bootstrapped_data, columns=['Category', 'Bootstrapped Mean',"cv"])
    return (boot_df,)


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
    plot_data = model_performance_df.melt(var_name="Model", value_name="Inverse Loss")

    # Merge rankings
    plot_data = plot_data.merge(sk_ranks, on="Model")
    plot_data = plot_data.sort_values(["Model"])


    fp = sns.FacetGrid(plot_data,col="Rank",sharey=True,sharex=False,height=4,aspect=0.5,)
    fp.map_dataframe(
        sns.boxplot,
        x="Model",
        y="Inverse Loss",
    )

    fp.set(xlabel=None,)

    plt.tight_layout()
    # plt.savefig("./plots/npsk_%s_%d.png" % (performance_measure,program_num), bbox_inches='tight', pad_inches=0, transparent=True)
    plt.show()
    return (plot_data,)


@app.cell
def _(plot_data):
    import plotly.graph_objects as go

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
            x=sub_df["Inverse Loss"],
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

        margin=dict(l=0.1, r=0.1, t=0, b=0), 
        # template="seaborn",
        width=250,
        height=175
    )
    # Save the figure as a PDF
    # fig.write_image("./plots/npsk_%s_%d.png" % (performance_measure,program_num), engine="kaleido",scale=5)
    fig.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
