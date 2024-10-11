import marimo

__generated_with = "0.5.2"
app = marimo.App()


@app.cell
def __():
    import marimo as mo
    from helpers.experiment_scripts import load_json
    import numpy as np
    return load_json, mo, np


@app.cell
def __(load_json):
    d = load_json("results/experiments.json")
    d[0]
    return d,


@app.cell
def __():
    lfn_names = ['L1_Spec' , 'DTW_Onset', 'SIMSE_Spec', 'JTFS']
    return lfn_names,


@app.cell
def __(d, lfn_names, np):
    def get_p_error(e):
        p1 = np.array(list(e["true_params"]["params"].values()))
        p2 = np.array(list(e["norm_params"].values()))[:,-1]
        return np.sqrt(np.sum((p1-p2)**2))
    def filter_experiments(d,loss_fn_name,prog_num):
        return [x for x in d if x["loss"]==loss_fn_name and x["program_id"]==prog_num]

    [[get_p_error(x) for x in filter_experiments(d,lfn_name,1)] for lfn_name in lfn_names]
    return filter_experiments, get_p_error


if __name__ == "__main__":
    app.run()
