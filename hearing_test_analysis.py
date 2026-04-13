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

    from rpy2.robjects.packages import importr
    from rpy2.robjects import pandas2ri
    import plotly.graph_objects as go
    import plotly.io as pio

    # List of similarity rating JSON files
    # JSON_FILES = ["survey_results/in_domain/similarity_ratings_a1.json", "survey_results/in_domain/similarity_ratings_a2.json"]
    JSON_FILES = ["survey_results/out_of_domain/amir_similarity_ratings.json","survey_results/out_of_domain/abram_similarity_ratings.json","survey_results/out_of_domain/kalvin_similarity_ratings.json",
                 "survey_results/out_of_domain/daniel_similarity_ratings.json"]


    PROGRAM_NAMES = {
        -1:"all",
        0: "am_non-overlapping freqs",
        1: "am_sine_target_saw_imitate",
        2: "am_saw_imitate_sine_target",
        3: "bp_noise_target_saw_imitate",
        4: "chirp_delayed_pitchbend",
        5: "chirp pulsating",
        6: "chirp normal",
    }

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
        PROGRAM_NAMES,
        combinations,
        df,
        go,
        importr,
        mo,
        np,
        pandas2ri,
        pd,
        pio,
        scipy,
    )


@app.cell
def _():
    # # sets per responder
    # r1 = set(df[df["Responder"] == 1]["sound_file"])
    # r2 = set(df[df["Responder"] == 2]["sound_file"])
    # r3 = set(df[df["Responder"] == 3]["sound_file"])

    # # files shared by 1 and 2
    # common_12 = r1 & r2

    # # missing in responder 3
    # missing = common_12 - r3

    # missing
    return


@app.cell
def _(mo):
    mo.md("""# D Responsers Agree? We calculate the spearman coeff""")
    return


@app.cell
def _(PROGRAM_NAMES, combinations, df, np, pd, scipy, stats):
    import pingouin as pg
    def kendalls_w(ratings: list[list]) -> dict:
        """
        Compute Kendall's W (coefficient of concordance) for multiple raters.
    
        Args:
            ratings: list of lists, one list per rater
        Returns:
            dict with W, chi2, and p_value
        """
        matrix = np.vstack([np.array(r) for r in ratings])
        ranked = np.apply_along_axis(scipy.stats.rankdata, 1, matrix)
    
        n_raters, n_items = ranked.shape
        rank_sums = ranked.sum(axis=0)
        S = np.sum((rank_sums - rank_sums.mean()) ** 2)
        W = 12 * S / (n_raters ** 2 * (n_items ** 3 - n_items))
    
        chi2 = n_raters * (n_items - 1) * W
        p_value = 1 - scipy.stats.chi2.cdf(chi2, df=n_items - 1)
    
        return {"W": round(W, 4), "chi2": round(chi2, 4), "p_value": round(p_value, 4)}


    def icc(ratings: list[list], icc_type: str = "ICC3") -> pd.DataFrame:
        """
        Compute Intraclass Correlation Coefficient across multiple raters.
        ICC3 = two-way mixed, consistency (ignores mean differences between raters).
        ICC2 = two-way random, absolute agreement (penalises generous raters).

        Args:
            ratings:  list of lists, one list per rater
            icc_type: which ICC row to return, e.g. "ICC3" or "ICC2" (default "ICC3")
        Returns:
            DataFrame row for the requested ICC type
        """
        n_items = len(ratings[0])
        n_raters = len(ratings)

        df = pd.DataFrame({
            "item":   list(range(n_items)) * n_raters,
            "rater":  [f"r{i}" for i in range(1, n_raters + 1) for _ in range(n_items)],
            "rating": np.concatenate([np.array(r) for r in ratings])
        })

        result = pg.intraclass_corr(data=df, targets="item", raters="rater", ratings="rating")
        return result[result["Type"] == icc_type][["Type", "ICC", "CI95%", "pval"]]


    def pairwise_spearman(ratings: list[list], names: list[str] = None) -> pd.DataFrame:
        """
        Compute all pairwise Spearman correlations between raters.
    
        Args:
            ratings: list of lists, one list per rater
            names:   optional list of rater names; defaults to R1, R2, ...
        Returns:
            DataFrame with columns rater_1, rater_2, rho, p_value
        """
        if names is None:
            names = [f"R{i+1}" for i in range(len(ratings))]

        rows = []
        for (i, r1), (j, r2) in combinations(enumerate(ratings), 2):
            rho, p = stats.spearmanr(r1, r2)
            rows.append({"rater_1": names[i], "rater_2": names[j],
                         "rho": round(rho, 4), "p_value": round(p, 4)})

        return pd.DataFrame(rows)

    def analyze_group(df,group_name=-1):
        matched_responses = [df[df.Responder == x].sort_values("sound_file")["Score"].values for x in df.Responder.unique()]
        print("kendall for group %s"%PROGRAM_NAMES[group_name], kendalls_w(matched_responses))
        print("icc3 ratings for group %s -> "%PROGRAM_NAMES[group_name], icc(matched_responses,icc_type="ICC3"))

    
    analyze_group(df)
    # per program spearman r
    df.groupby(["Program"]).apply(lambda x: analyze_group(x,x.name))
    return


@app.cell
def _():

    # def match_and_spearman(df,group_name="all"):
    #     matched_responses = [df[df.Responder == x].sort_values("sound_file")["Score"].values for x in df.Responder.unique()]
    #     print("spearman for %s"%group_name, scipy.stats.spearmanr(matched_responses[1],matched_responses[0]))
    #     # print("spearman for %s"%group_name, scipy.stats.spearmanr(matched_responses[2],matched_responses[0]))
    #     print("spearman for %s"%group_name, scipy.stats.spearmanr(matched_responses[1],matched_responses[2]))
    # match_and_spearman(df)
    # # per program spearman r
    # df.groupby(["Program"]).apply(lambda x: match_and_spearman(x,x.name))
    return


@app.cell
def _():

    # def get_matched_responses(df):
    #     """
    #     Returns a list of arrays, one per responder, aligned by sound_file.
    #     Shape conceptually: raters x items
    #     """
    #     responders = df["Responder"].unique()
    #     matched_responses = [
    #         df[df.Responder == r].sort_values("sound_file")["Score"].to_numpy()
    #         for r in responders
    #     ]

    #     lengths = [len(x) for x in matched_responses]
    #     if len(set(lengths)) != 1:
    #         raise ValueError(f"Raters do not have the same number of scored items: {lengths}")

    #     return matched_responses, responders


    # def match_and_spearman(df, group_name="all"):
    #     matched_responses, responders = get_matched_responses(df)

    #     n = len(matched_responses)
    #     print(f"\nSpearman for {group_name}")
    #     for i in range(n):
    #         for j in range(i + 1, n):
    #             rho, p = scipy.stats.spearmanr(matched_responses[i], matched_responses[j])
    #             print(f"  {responders[i]} vs {responders[j]}: rho={rho:.4f}, p={p:.4g}")


    # def kendalls_w_from_ratings(matched_responses):
    #     """
    #     Kendall's W for raters x items data.
    #     Uses average ranks, so ties are handled reasonably via rankdata(method='average').
    #     """
    #     data = np.asarray(matched_responses, dtype=float)  # shape: m raters x n items
    #     if data.ndim != 2:
    #         raise ValueError("matched_responses must be 2D: raters x items")

    #     m, n = data.shape
    #     if m < 2:
    #         raise ValueError("Need at least 2 raters")
    #     if n < 2:
    #         raise ValueError("Need at least 2 items")

    #     ranked = np.array([
    #         scipy.stats.rankdata(row, method="average")
    #         for row in data
    #     ])  # m x n

    #     R = np.sum(ranked, axis=0)
    #     R_bar = np.mean(R)
    #     S = np.sum((R - R_bar) ** 2)

    #     # Tie correction
    #     T = 0.0
    #     for row in data:
    #         _, counts = np.unique(row, return_counts=True)
    #         T += np.sum(counts**3 - counts)

    #     denominator = m**2 * (n**3 - n) - m * T
    #     if denominator == 0:
    #         return np.nan

    #     W = 12 * S / denominator
    #     return W


    # def kendalls_w_test_from_ratings(matched_responses):
    #     """
    #     Returns Kendall's W plus chi-square approximation and p-value.
    #     """
    #     data = np.asarray(matched_responses, dtype=float)
    #     m, n = data.shape

    #     W = kendalls_w_from_ratings(data)
    #     chi2_stat = m * (n - 1) * W
    #     p_value = scipy.stats.chi2.sf(chi2_stat, df=n - 1)

    #     return {
    #         "kendalls_w": W,
    #         "chi2": chi2_stat,
    #         "df": n - 1,
    #         "p_value": p_value,
    #     }


    # def match_and_kendalls_w(df, group_name="all"):
    #     matched_responses, responders = get_matched_responses(df)
    #     result = kendalls_w_test_from_ratings(matched_responses)

    #     print(f"\nKendall's W for {group_name}")
    #     print(f"  raters: {list(responders)}")
    #     print(f"  W={result['kendalls_w']:.4f}, chi2={result['chi2']:.4f}, p={result['p_value']:.4g}")

    #     return result


    # match_and_spearman(df)
    # match_and_kendalls_w(df)

    # # per program
    # df.groupby("Program").apply(lambda x: match_and_kendalls_w(x, x.name))

    return


@app.cell
def _(df, pd):
    # import pandas as pd
    from scipy.stats import spearmanr

    def spearman_table(df, group_name="all"):
        responders = sorted(df["Responder"].unique())

        # align by sound_file within each responder
        matched = {
            r: df[df["Responder"] == r]
                .sort_values("sound_file")["Score"]
                .to_numpy()
            for r in responders
        }

        rows = []
        for i in range(len(responders)):
            for j in range(i + 1, len(responders)):
                r1, r2 = responders[i], responders[j]
                rho, p = spearmanr(matched[r1], matched[r2])
                rows.append({
                    "Group": group_name,
                    "Pair": f"{r1} vs {r2}",
                    "Spearman_rho": round(rho, 3),
                    "p_value": f"{p:.2e}",
                    "n": len(matched[r1]),
                })

        return pd.DataFrame(rows)

    # all data
    all_results = spearman_table(df)

    # per program
    program_results = (
        df.groupby("Program", group_keys=False)
          .apply(lambda x: spearman_table(x, group_name=x.name))
          .reset_index(drop=True)
    )

    results = pd.concat([all_results, program_results], ignore_index=True)
    pretty = results.pivot(index="Group", columns="Pair", values="Spearman_rho")
    print(pretty.round(3).to_string())
    return


@app.cell
def _(np):
    # Bootstrapping function to compute means and confidence intervals


    def bootstrap_means(scores, n_iterations=1000):
        boot_means = [np.mean(np.random.choice(scores, size=len(scores), replace=True)) for _ in range(n_iterations)]
        return boot_means

    # Perform bootstrapping for each category

    # 0: am_non-overlapping freqs
    # 1: am_sine_target_saw_imitate
    # 2: am_saw_imitate_sine_target
    # 3: bp_noise_target_saw_imitate
    # 4: chirp_delayed_pitchbend
    # 5: chirp pulsating
    # 6: chirp normal
    return (bootstrap_means,)


@app.cell
def _():
    return


@app.cell
def _():
    # def bootstrap_means(scores, n_iterations=1000):
    #     boot_means = [np.mean(np.random.choice(scores, size=len(scores), replace=True)) for _ in range(n_iterations)]
    #     return boot_means

    # bootstrapped_data = []
    # percentiles = {}
    # program_num = 1

    # for category in df['Function'].unique():
    #     category_scores = df[ (df["Program"]==program_num) &  (df['Function'] == category) ]['Score'].values
    #     boot_means = bootstrap_means(category_scores)
    #     bootstrapped_data.extend([(category, mean, i) for i,mean in enumerate(boot_means)])

    # boot_df = pd.DataFrame(bootstrapped_data, columns=['Category', 'Bootstrapped Mean',"cv"])

    # # Activate automatic conversion between pandas and R
    # pandas2ri.activate()

    # # Import ScottKnottESD from R
    # sk = importr("ScottKnottESD")

    # np.random.seed(42)

    # model_performance_df = boot_df.pivot(index='cv', columns='Category', values='Bootstrapped Mean')
    # model_performance_df.to_csv("model_performance.csv",index=False)
    # # Convert DataFrame to R object and run Scott-Knott ESD
    # r_data = pandas2ri.py2rpy(model_performance_df)
    # sk_results = sk.sk_esd(r_data)

    # # Extract rankings
    # sk_ranks = pd.DataFrame({
    #     "Model": sk_results.rx2("nms")[[x-1 for x in list(sk_results.rx2("ord"))]],
    #     "Rank": [ int(rank) for rank in list(sk_results.rx2("groups"))]
    # })

    # # Convert DataFrame to long format for Seaborn
    # plot_data = model_performance_df.melt(var_name="Model", value_name="Score")

    # # Merge rankings
    # plot_data = plot_data.merge(sk_ranks, on="Model")
    # plot_data = plot_data.sort_values(["Model"])

    # pio.templates.default = "plotly_white"
    # # Sort models alphabetically
    # model_order = sorted(plot_data["Model"].unique())

    # # Rank-to-color mapping
    # rank_palette = {
    #     1: "#70FF70",
    #     2: "#858585",
    #     3: "#454545",
    #     4: "#000000"
    # }

    # # Create the figure manually, one half-violin per model
    # fig = go.Figure()

    # for model in model_order:
    #     sub_df = plot_data[plot_data["Model"] == model]
    #     rank = int(sub_df["Rank"].iloc[0])
    #     color = rank_palette[rank]

    #     fig.add_trace(go.Violin(
    #         x=sub_df["Score"],
    #         y=[model] * len(sub_df),
    #         orientation="h",
    #         name=model,
    #         line_color=color,
    #         fillcolor=color,
    #         box_visible=False,
    #         meanline_visible=False,
    #         side="positive",  # <-- half violin
    #         points="outliers",
    #         marker=dict(color=color, outliercolor=color, line=dict(color=color)),
    #         width=0.6
    #     ))

    # # Minimal layout
    # fig.update_layout(
    #     xaxis=dict(
    #         side="top",  # <- move title to the top
    #         tickfont=dict(
    #             family="JetBrainsMono Nerd Font Mono",  # Bold font
    #             size=16,
    #             color="black"
    #         )

    #     ),
    #         yaxis=dict(
    #         showticklabels=False  # hides the y-axis tick labels (model names)
    #     ),
    #     showlegend=False,
    #     yaxis_title=None,
    #     title=None,

    #     margin=dict(l=0, r=0, t=0, b=0), 
    #     width=250,
    #     height=175
    # )

    # # Save the figure as a PDF
    # fig.write_image("./plots/npsk_ood_likert_%d.png" % (program_num), engine="kaleido",scale=5)
    # fig.show()

    return


@app.cell
def _(PROGRAM_NAMES, bootstrap_means, df, go, importr, np, pandas2ri, pd, pio):


    program_nums = [0,1,2,3,4,5,6]
    all_sk_ranks = {}  # program_num → sk_ranks DataFrame

    for program_num in program_nums:

        program_name = PROGRAM_NAMES[program_num]

        # fig.write_image(f"./plots/npsk_ood_likert_{program_name}.png", engine="kaleido", scale=5)

        bootstrapped_data = []
        for category in df['Function'].unique():
            category_scores = df[
                (df["Program"] == program_num) &
                (df['Function'] == category)
            ]['Score'].values
            boot_means = bootstrap_means(category_scores)
            bootstrapped_data.extend([(category, mean, i) for i, mean in enumerate(boot_means)])

        boot_df = pd.DataFrame(bootstrapped_data, columns=['Category', 'Bootstrapped Mean', 'cv'])

        pandas2ri.activate()
        sk = importr("ScottKnottESD")
        np.random.seed(42)

        model_performance_df = boot_df.pivot(index='cv', columns='Category', values='Bootstrapped Mean')

        r_data = pandas2ri.py2rpy(model_performance_df)
        sk_results = sk.sk_esd(r_data)

        sk_ranks = pd.DataFrame({
            "Function": sk_results.rx2("nms")[[x - 1 for x in list(sk_results.rx2("ord"))]],
            "Rank": [int(rank) for rank in list(sk_results.rx2("groups"))]
        })
        sk_ranks["Program"] = program_name

        all_sk_ranks[program_num] = sk_ranks

        # Plot
        plot_data = model_performance_df.melt(var_name="Model", value_name="Score")
        plot_data = plot_data.merge(sk_ranks.rename(columns={"Function": "Model"}), on="Model")
        plot_data = plot_data.sort_values("Model")

        pio.templates.default = "plotly_white"
        model_order = sorted(plot_data["Model"].unique())
        rank_palette = {1: "#70FF70", 2: "#858585", 3: "#454545", 4: "#000000"}

        fig = go.Figure()
        for model in model_order:
            sub_df = plot_data[plot_data["Model"] == model]
            rank = int(sub_df["Rank"].iloc[0])
            color = rank_palette.get(rank, "#000000")
            fig.add_trace(go.Violin(
                x=sub_df["Score"],
                y=[model] * len(sub_df),
                orientation="h",
                name=model,
                line_color=color,
                fillcolor=color,
                box_visible=False,
                meanline_visible=False,
                side="positive",
                points="outliers",
                marker=dict(color=color, outliercolor=color, line=dict(color=color)),
                width=0.6
            ))
        fig.update_layout(
            xaxis=dict(side="top", tickfont=dict(family="JetBrainsMono Nerd Font Mono", size=16, color="black")),
            yaxis=dict(showticklabels=False),
            showlegend=False,
            yaxis_title=None,
            title=None,
            margin=dict(l=0, r=0, t=0, b=0),
            width=250,
            height=175
        )
        # fig.write_image(f"./plots/npsk_ood_likert_{program_num}.png", engine="kaleido", scale=5)
        fig.write_image(f"./plots/npsk_ood_likert_{program_name}.png", engine="kaleido", scale=5)
        fig.show()
    return (all_sk_ranks,)


@app.cell
def _(all_sk_ranks, pd):
    summary_df = (
        pd.concat(all_sk_ranks.values(), ignore_index=True)
          .pivot(index="Function", columns="Program", values="Rank")
          # .rename(columns=lambda p: p)
          .sort_index()
    )

    summary_df
    return


if __name__ == "__main__":
    app.run()
