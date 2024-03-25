import marimo

__generated_with = "0.2.12"
app = marimo.App(width="full")


@app.cell
def __():
    import marimo as mo
    import numpy as np
    import jax.numpy as jnp
    import jax 
    import matplotlib.pyplot as plt
    import pandas as pd 
    from scipy import signal # needed for sawtooth
    import itertools # needed for generators
    return itertools, jax, jnp, mo, np, pd, plt, signal


@app.cell
def __(itertools, np, plt, signal):
    # make a signal (3 falling ramps followed by silence)
    # this signal repeats indefinitely 
    sr = 100
    t = np.linspace(0, 1, sr,endpoint=False)
    ramp = signal.sawtooth(2 * np.pi * 1 * t,width=0)*0.5 + 0.5
    s = np.concatenate([ramp,ramp,ramp,[0]*sr])
    signal_gen = itertools.cycle(s)
    plt.plot([next(signal_gen) for i in range(3000)])
    return ramp, s, signal_gen, sr, t


@app.cell
def __(np):
    class agent:
        def __init__(self, nback, learning_rate,discount_factor):
            """ nback: size of past values/state size""" 
            self.lr = learning_rate
            self.gamma = discount_factor
            self.nback = nback
            self.w = np.random.rand(nback)*0.001
            self.s = np.zeros(nback)
            self.error = []
        def update(self,reward):  
            s_prime = np.roll(self.s,1)
            # s_prime = self.s.at[0].set[reward]
            s_prime[0] = reward
            td_error = (reward
                       + self.gamma*np.dot(self.w,s_prime) 
                       - np.dot(self.w,self.s))
            self.w = self.w + self.lr * td_error
            self.error.append(td_error)
            self.s = s_prime
        def log_values(self):
            print("state:",self.s)
            print("w:",self.w)


    agent = agent(10,0.1,0.1)

    test_signal = []
    return agent, test_signal


@app.cell
def __(agent, np):
    # some tests
    agent.__init__
    agent.w = np.zeros(10)

    return


@app.cell
def __(agent, plt, signal_gen, test_signal):
    # is there a way to prevent td errors that we know are coming?
    for i in range(100):    
        r = next(signal_gen)
        test_signal.append(r)
        agent.update(r)
    plt.plot(agent.error,label="td error")
    plt.plot(test_signal,label="signal")
    return i, r


if __name__ == "__main__":
    app.run()
