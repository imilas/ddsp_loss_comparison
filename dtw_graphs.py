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
    return dtw_loss_adsr,


@app.cell
def __(mo):
    mo.md("#ADSR alignment")
    return


@app.cell
def __(generate_adsr, plt, sample_rate):
    import dtw 
    adsr_envelope_1 = generate_adsr(0.1, 0.1, 0.5, 0.8, 0.4, sample_rate)
    adsr_envelope_2 = generate_adsr(0.2, 0.1, 1, 0.3, 0.4, sample_rate)
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
def __(adsr_envelope_1, adsr_envelope_2, dtw):
    ## Align and plot with the Rabiner-Juang type VI-c unsmoothed recursion
    dtw.dtw(adsr_envelope_1,adsr_envelope_2, keep_internals=True, 
        step_pattern=dtw.rabinerJuangStepPattern(6, "c"))\
        .plot(type="twoway",offset=-2)
    return


if __name__ == "__main__":
    app.run()
