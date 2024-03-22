import marimo

__generated_with = "0.2.12"
app = marimo.App()


@app.cell
def __():
    import marimo as mo
    import numpy as np
    from scipy import signal # needed for sawtooth
    import itertools # needed for generators
    import matplotlib.pyplot as plt
    return itertools, mo, np, plt, signal


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
    plot_sine_wave(amplitude.value,frequency.value)
    return


@app.cell
def __(np, plt, signal):
    # make a signal (3 falling ramps followed by silence)
    sr = 100
    t = np.linspace(0, 1, sr,endpoint=False)
    ramp = signal.sawtooth(2 * np.pi * 1 * t,width=0)*0.5 + 0.5
    s = np.concatenate([ramp,ramp,ramp,[0]*sr])
    plt.plot(s)
    return ramp, s, sr, t


@app.cell
def __(itertools, s):
    # using itertools, we can repeat the signal endlessly
    signal_gen = itertools.cycle(s)
    print(next(signal_gen))
    print(next(signal_gen))
    return signal_gen,


if __name__ == "__main__":
    app.run()
