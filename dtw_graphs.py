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
    dtw_jax = softdtw_jax.SoftDTW(gamma=0.1)
    dtw_jit = jax.jit(dtw_jax)
    return dtw_jax, dtw_jit


@app.cell
def __(np, plt):
    ## A noisy sine wave as query
    len_signal = 100
    idx = np.linspace(0,3.14*2,num=len_signal)
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


@app.cell
def __(mo):
    mo.md("#ADSR function")
    return


@app.cell
def __(mo, np):
    # ADSR function
    def generate_adsr(attack_time, decay_time, sustain_level, sustain_time, release_time, sample_rate):
        # Calculate the number of samples for each phase
        attack_samples = int(attack_time * sample_rate)
        decay_samples = int(decay_time * sample_rate)
        sustain_samples = int(sustain_time * sample_rate)
        release_samples = int(release_time * sample_rate)

        # Attack phase: linear increase from 0 to 1
        attack = np.linspace(0, 1, attack_samples)

        # Decay phase: linear decrease from 1 to sustain level
        decay = np.linspace(1, sustain_level, decay_samples)

        # Sustain phase: constant at sustain level
        sustain = np.full(sustain_samples, sustain_level)

        # Release phase: linear decrease from sustain level to 0
        release = np.linspace(sustain_level, 0, release_samples)

        # Concatenate all phases to create the ADSR envelope
        adsr_envelope = np.concatenate([attack, decay, sustain, release])

        return adsr_envelope


    sample_rate = 30  # samples per second (Hz)

    # Parameters for the ADSR envelope
    attack_time =  mo.ui.slider(0, 1,0.1,0.1) # seconds
    decay_time = mo.ui.slider(0, 1,0.1,0.2)    # seconds
    sustain_level = mo.ui.slider(0, 1,0.1,0.7)  # amplitude (0 to 1)
    sustain_time = mo.ui.slider(0, 1,0.1,0.5)   # seconds
    release_time = mo.ui.slider(0, 1,0.1,0.3)   # seconds
    return (
        attack_time,
        decay_time,
        generate_adsr,
        release_time,
        sample_rate,
        sustain_level,
        sustain_time,
    )


@app.cell
def __(attack_time, decay_time, release_time, sustain_level, sustain_time):
    attack_time,decay_time,sustain_level,sustain_time,release_time
    return


@app.cell
def __():
    return


@app.cell
def __(
    attack_time,
    decay_time,
    generate_adsr,
    np,
    plt,
    release_time,
    sample_rate,
    sustain_level,
    sustain_time,
):
    # Generate the ADSR envelope
    adsr_envelope = generate_adsr(attack_time.value, decay_time.value, sustain_level.value, sustain_time.value, release_time.value, sample_rate)
    adsr_shifts = np.arange(0,len(adsr_envelope),1) # how much to offset the signal
    # Plot the ADSR envelope
    time = np.linspace(0, len(adsr_envelope) / sample_rate, len(adsr_envelope))
    plt.plot(time, adsr_envelope)
    plt.title("ADSR Envelope")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.show()
    return adsr_envelope, adsr_shifts, time


@app.cell
def __(adsr_envelope, adsr_shifts, dtw_jit, np, plt):
    dtw_loss_adsr = [dtw_jit(adsr_envelope,np.roll(adsr_envelope,x)) for x in adsr_shifts]
    plt.plot(adsr_shifts,dtw_loss_adsr)
    plt.xlabel("Shift Steps")
    plt.ylabel("Distance")
    plt.tight_layout()
    plt.show()
    return dtw_loss_adsr,


@app.cell
def __(dtw_jit, idx, len_signal, np, plt):
    dtw_loss_cos= [dtw_jit(np.cos(idx),np.cos(np.linspace(0,3.14*x,num=len_signal))) for x in np.linspace(0,35,141,endpoint=True)]
    plt.plot(np.linspace(0,35,141,endpoint=True),dtw_loss_cos,label="Distance")
    plt.axvline(
        2,
        color="#FF0000",
        linestyle="-",
        label="Correct Frequency",
    )
    for i in range(8):
        f = (i+2)*2
        plt.axvline(
        f,
        color =  (1, 0, 0,.7-i/10),
        linestyle=":",)


    plt.xticks(range(0, 30 + 1, 2 ))  
    plt.legend()
    plt.xlabel("Cosine Frequency")
    plt.ylabel("Distance")
    plt.tight_layout()
    plt.show()
    return dtw_loss_cos, f, i


@app.cell
def __(mo):
    mo.md("#ADSR alignment")
    return


@app.cell
def __(adsr_envelope, generate_adsr, plt, sample_rate):
    import dtw 
    # adsr_envelope_1 = generate_adsr(0.25, 0.1, 1, 0.35, 0.4, sample_rate)
    adsr_envelope_1 = adsr_envelope
    adsr_envelope_2 = generate_adsr(0.1, 0.3, 1, 0.3, 0.4, sample_rate)
    plt.plot(adsr_envelope_1)
    plt.plot(adsr_envelope_2)
    return adsr_envelope_1, adsr_envelope_2, dtw


@app.cell
def __(adsr_envelope_1, adsr_envelope_2, dtw):
    alignment = dtw.dtw(adsr_envelope_1, adsr_envelope_2, keep_internals=True)
    ## Display the warping curve, i.e. the alignment curve
    alignment.plot(type="threeway")
    return alignment,


@app.cell
def __(dtw, plt):
    def draw_alignments(s1,s2):
        ## Align and plot with the Rabiner-Juang type VI-c unsmoothed recursion
        ax = dtw.dtw(s1,s2, keep_internals=True)\
            .plot(type="twoway",offset=1)
        ax2 = ax.get_shared_x_axes().get_siblings(ax)[1]
        ax2.set_ylabel('Reference Value', color='b')  # This sets the label for the second y-axis
        
        # plt.title("ADSR Alignment")
        plt.tight_layout()
        return plt
    return draw_alignments,


@app.cell
def __(adsr_envelope_1, draw_alignments, np):
    p1 = draw_alignments(adsr_envelope_1,np.roll(adsr_envelope_1,1))
    p1.show()
    return p1,


@app.cell
def __(draw_alignments, idx, np, template):
    p2 = draw_alignments(template,np.cos(idx))
    p2.show()
    return p2,


@app.cell
def __(np, plt):

    import librosa
    import librosa.display

    # Define the sample rate and time vector
    fs = 2048  # Sample rate (Hz)
    t = np.linspace(0, 2, 2 * fs, endpoint=False)  # Time vector for 2 seconds
    # Generate the signal
    signal = np.concatenate([
        np.sin(2 * np.pi * 15 * t[:fs]),  # 4 Hz for the first second
        np.sin(2 * np.pi * 60 * t[fs:])+0.1*np.sin(2 * np.pi * 300 * t[fs:]),
    ])

    # Plot Time Domain
    plt.figure(figsize=(6, 8))

    plt.subplot(3, 1, 1)
    plt.plot(t, signal)
    plt.title('Time Domain')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')

    # Plot Frequency Domain
    plt.subplot(3, 1, 2)
    plt.psd(signal, NFFT=fs, Fs=fs)
    plt.title('Frequency Domain')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Power/Frequency [dB/Hz]')

    # Compute and Plot Mel Spectrogram
    plt.subplot(3, 1, 3)
    # Compute the mel spectrogram
    S = librosa.feature.melspectrogram(y=signal, sr=fs, n_mels=512, fmax=fs//2,hop_length=32,win_length=256*2, fmin=0)
    # Convert to dB scale
    S_dB = librosa.power_to_db(S,ref=np.max)

    # Plot Mel Spectrogram
    librosa.display.specshow(S_dB, sr=fs, x_axis='time', y_axis='mel',hop_length=512,fmax=256)
    plt.title('Spectrogram')
    # plt.colorbar(format='%+2.0f dB')
    plt.xlabel('Time [s]')
    plt.ylabel('Frequency [Hz]')

    plt.tight_layout()
    plt.show()

    return S, S_dB, fs, librosa, signal, t


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
