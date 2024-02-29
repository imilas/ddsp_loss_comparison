import marimo

__generated_with = "0.2.8"
app = marimo.App()


@app.cell
def __():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    return mo, np, plt


@app.cell
def __(np, plt):
    def plot_sine_wave( amplitude,frequency,sr=44000):
        x = np.linspace(0, 2*np.pi, num=sr)
        plt.figure(figsize=(6.7, 2.5))
        values = amplitude*np.sin(frequency*x)
        plt.plot(x, values)
        plt.xlabel('$x$')
        plt.xlim(0, 2*np.pi)
        plt.ylim(-2, 2)
        plt.tight_layout()
        return plt.gca(),values
    return plot_sine_wave,


@app.cell
def __(mo):
    amplitude = mo.ui.slider(start=1, stop=2, step=0.1, label="amplitude")
    frequency = mo.ui.slider(start = 1, stop = 100, step = 1 , label = "frequency")
    [amplitude,frequency]
    return amplitude, frequency


@app.cell
def __(amplitude, frequency, plot_sine_wave):
    p, v = plot_sine_wave(amplitude.value,frequency.value)
    p
    return p, v


if __name__ == "__main__":
    app.run()
