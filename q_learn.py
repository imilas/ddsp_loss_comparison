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
    import plotly.express as px
    import plotly.graph_objects as go

    return go, itertools, jax, jnp, mo, np, pd, plt, px, signal


@app.cell
def __(itertools, mo, np, plt, signal):
    # make a signal (2 falling ramps followed by silence)
    # this signal repeats indefinitely 
    sr = 10
    t = np.linspace(0, 1, sr,endpoint=False)
    ramp = signal.sawtooth(2 * np.pi * 1 * t,width=0)*0.5 + 0.5
    s = np.concatenate([ramp,ramp,[0]*sr])
    signal_gen = itertools.cycle(s)
    sig_fig = plt.plot([next(signal_gen) for i in range(40)])

    mo.md(f"""
        We want to predict fluctuations in complex signals. This can be thought of as a form of 1-dimensional "nexting", where the past steps in a signal can be used to predict the values we might expect in the future of the same signal. We want to make this predictions in real-time, i.e, a signal is observed 1 time-step at a time and the observer (which we call agent) will make predictions about the future based on what values it has observed in the past. 
        
        To test the simplest case of making linear predictions, we create a signal that is a combination of sawtooth waves followed by a period of silence. This signal is the periodic: it consists of a saw-wave, followed by a saw-wave, followed by a stream of zeros (followed by 2 saw-waves, zeros, and so on). All 3 components take 10 time steps to complete, meaning that the signal in its entirety is 30 time-steps. 
         {mo.as_html(sig_fig)}
        """
         )
    return ramp, s, sig_fig, signal_gen, sr, t


@app.cell
def __(mo, np):
    def ideal_return(signal,gamma,steps):
        gs = np.array([gamma**i for i in range(steps)]) 
        return np.sum([x*y for x,y in zip(gs,signal)])
    mo.md(
        r'''
        ideal return:
        The ideal discounted return that the linear, discounted agent will approximate is calculated by:
        
        $v_{t} =\sum_{k=0}^{\infty}\gamma^{k}R_{t+k+1}=G_t$
        '''
    )
    return ideal_return,


@app.cell
def __(np):
    class agent:
        def __init__(self, nback, learning_rate,discount_factor,trace_decay=0.1,eps_decay=1,eps_min=1e-4):
            """ nback: size of past values/state size""" 
            self.lr = learning_rate
            self.gamma = discount_factor
            self.nback = nback
            self.w = np.random.rand(self.nback)*0.001
            self.s = np.zeros(self.nback)
            self.error = []
            self.trace_decay = trace_decay
            self.z = np.zeros(self.nback)
        def update(self,reward):  
            s_prime = np.roll(self.s,1)
            # s_prime = self.s.at[0].set[reward]
            s_prime[0] = reward
            td_error = (reward
                       + self.gamma*np.dot(self.w,s_prime) 
                       - np.dot(self.w,self.s))
            self.z = self.gamma * self.trace_decay * self.z + self.s
            self.w = self.w + self.lr * td_error*self.z
            self.error.append(td_error)
            self.s = s_prime
        def multi_update(self,signal_slice,log=False):
            """multiple update steps given a slice of signal"""
            for v in signal_slice:
                self.update(v)
                if log:
                    self.log_values()
        def reset(self):
            self.w = np.random.rand(self.nback)*0.001
            self.s = np.zeros(self.nback)
            self.error = []
        def log_values(self):
            print("state:",self.s)
            print("w:",self.w)
            print("error",self.error[-1])

    return agent,


@app.cell
def __(agent, np):
    # A test of update value
    agent_1 = agent(3,0.1,0.4,trace_decay=0)
    agent_1.w = np.zeros(3)
    agent_1.multi_update([1,1],log=False)
    np.testing.assert_array_equal(agent_1.w, np.array([0.1,0,0]),err_msg="wrong update")
    return agent_1,


@app.cell
def __(agent, mo, np, pd, px, signal_gen):
    # is there a way to prevent td errors that we know are coming?
    test_signal = []
    agent_2 = agent(29,0.01,0.4,trace_decay=0.0,eps_decay=0.999,eps_min=0.001)
    for i in range(100000):    
        r = next(signal_gen)
        test_signal.append(r)
        agent_2.update(r)
    df_2 = pd.DataFrame({"error":np.array(agent_2.error)**2,"test_signal":test_signal})
    mo.ui.plotly( px.line(df_2["error"]))
    return agent_2, df_2, i, r, test_signal


@app.cell
def __(agent_2, go, np):
    # learned weights after training
    weights_values_figure = go.Figure(go.Bar(x=agent_2.w,y=np.arange(len(agent_2.w)),orientation="h"))
    weights_values_figure.update_layout(title="weight values",height=300)
    return weights_values_figure,


@app.cell
def __(ideal_return, itertools, mo, np, pd, px, signal_gen):
    # ideal return plots. Zoom in for a better view. 
    # to do: use plotly for better interactivity
    signal_clone = itertools.tee(signal_gen,1)[0] # clone the signal_generator
    future_values = [next(signal_clone) for i in range(40)]
    windows = np.lib.stride_tricks.sliding_window_view(future_values,10)
    returns = [ideal_return(w,0.4,5) for w in windows]
    df_1 = pd.DataFrame({"ideal returns":returns,"signal":future_values[0:len(returns)]})
    fig = px.line(df_1,title="ideal returns vs signal")
    mo.ui.plotly(fig)
    return df_1, fig, future_values, returns, signal_clone, windows


@app.cell
def __(agent_2, ideal_return, mo, np, pd, px, signal_clone):
    # plot ideal returns vs prediction
    s_copy = [next(signal_clone) for i in range(160)]
    predictions, ideal_r = [],[]
    for i2 in range(100):
        offset = agent_2.nback + i2 
        past = s_copy[i2:offset]
        future = s_copy[offset:offset+10]
        predictions.append(np.dot(agent_2.w,past[::-1]))
        ideal_r.append(ideal_return(future,0.4,10))
        
    df_3 = pd.DataFrame({"predictions":predictions,"ideal returns":ideal_r,"signal":s_copy[0:len(ideal_r)]})
    mo.ui.plotly(px.line(df_3,title="ideal returns vs signal"))
    return df_3, future, i2, ideal_r, offset, past, predictions, s_copy


if __name__ == "__main__":
    app.run()
