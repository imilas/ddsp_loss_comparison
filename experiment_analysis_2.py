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
    program_num = 3
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
def __(g, lfn_names, np, pd):
    # Convert data and create a DataFrame from `g` and `lfn_names`
    performance_data = {'Category': [], 'Score': []}
    for category, scores in zip(lfn_names, g):
        performance_data['Category'].extend([category] * len(scores))
        performance_data['Score'].extend(scores)

    df = pd.DataFrame(performance_data)

    # Bootstrapping function to compute means and confidence intervals
    def bootstrap_means(scores, n_iterations=1000):
        boot_means = [1/np.mean(np.random.choice(scores, size=len(scores) , replace=True)) for _ in range(n_iterations)]
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
    return (
        boot_df,
        boot_means,
        bootstrap_means,
        bootstrapped_data,
        category,
        category_scores,
        ci_lower,
        ci_upper,
        df,
        percentiles,
        performance_data,
        scores,
    )


@app.cell
def __(boot_df, np, pd, performance_measure, plt, program_num, sns):
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
    plot_data = model_performance_df.melt(var_name="Model", value_name="Inverse Loss")

    # Merge rankings
    plot_data = plot_data.merge(sk_ranks, on="Model")
    plot_data = plot_data.sort_values(["Rank","Model"])
    # Set color palette for ranks
    unique_ranks = sorted(plot_data["Rank"].unique())
    rank_palette = dict(zip(unique_ranks, sns.color_palette("Blues", len(unique_ranks))))

    # Create the boxplot
    # plt.figure(figsize=(10, 6))


    fp = sns.FacetGrid(plot_data,col="Rank",sharey=True,sharex=False,height=4,aspect=0.75,)
    fp.map_dataframe(
        sns.boxplot,
        x="Model",
        y="Inverse Loss",
        # order=["Rank-1", "Rank-2", "Rank-3", "Rank-4"],  # Adjust to match your data
        palette="Blues"
        
      
    )

    # Adjustments
    # plt.xticks(rotation=45)
    # plt.ylim(0, 1)
    fp.set(xlabel=None,)
    # plt.xlabel("Model")
    # plt.ylabel("Inverse Loss")
    # plt.title("Scott-Knott ESD Rankings (Placeholder Data)")
    # plt.legend(title="Rank", loc="upper right")
    # plt.grid(True)
    plt.tight_layout()
    plt.savefig("./plots/npsk_%s_%d.png" % (performance_measure,program_num), bbox_inches='tight', pad_inches=0, transparent=True)
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


if __name__ == "__main__":
    app.run()
