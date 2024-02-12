import marimo

__generated_with = "0.2.2"
app = marimo.App(width="full")


@app.cell
def __():
    import marimo as mo
    import librosa
    import helper 
    import matplotlib.pyplot as plt
    import numpy as np

    import importlib as imp
    imp.reload(helper)
    return helper, imp, librosa, mo, np, plt


@app.cell
def __(mo):
    url0 = "https://upload.wikimedia.org/wikipedia/commons/b/b2/Snare_drum_unmuffled.ogg"
    url1 = "https://upload.wikimedia.org/wikipedia/commons/c/c4/Drum_Roll_Intro.ogg"
    url2 = "https://upload.wikimedia.org/wikipedia/commons/b/b2/Phaser_on_drums_%28slow_phaser%29.ogg"
    mo.vstack(
        [
            mo.audio(src=url0),
            mo.audio(src=url1),
            mo.audio(src=url2),
        ]
    )
    return url0, url1, url2


@app.cell
def __(helper, url0):
    y, sr = helper.load_url(url0)
    return sr, y


@app.cell
def __(librosa, np, plt, y):
    S = np.abs(librosa.stft(y, hop_length=16))
    fig, ax = plt.subplots()
    img = librosa.display.specshow(
        librosa.amplitude_to_db(S, ref=np.max), y_axis="log", x_axis="time", ax=ax
    )
    ax.set_title("Power spectrogram")
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax
    return S, ax, fig, img


if __name__ == "__main__":
    app.run()
