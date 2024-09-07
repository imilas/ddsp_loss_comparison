import marimo

__generated_with = "0.5.2"
app = marimo.App()


@app.cell
def __():
    import marimo as mo
    import jax
    import jax.numpy as jnp
    from jax.scipy.optimize import minimize
    import matplotlib.pyplot as plt
    import numpy as np
    from helpers import softdtw_jax
    return jax, jnp, minimize, mo, np, plt, softdtw_jax


@app.cell
def __(jax, softdtw_jax):
    dtw_jax = softdtw_jax.SoftDTW(gamma=0.8)
    dtw_jit = jax.jit(dtw_jax)
    return dtw_jax, dtw_jit


@app.cell
def __(np, plt):
    ## A noisy sine wave as query
    len_signal = 100
    idx = np.linspace(0,3.14,num=len_signal)
    query = np.sin(idx) + np.random.uniform(size=100)/10.0
    template = np.cos(idx) + np.random.uniform(size=100)/10.0
    plt.plot(template)
    return idx, len_signal, query, template


@app.cell
def __(dtw_jit, len_signal, np, plt, template):
    shifts = np.arange(0,len_signal,1) # how much to offset the signal
    dtw_loss = [dtw_jit(template,np.roll(template,x)) for x in shifts]
    plt.plot(shifts,dtw_loss)
    return dtw_loss, shifts


if __name__ == "__main__":
    app.run()
