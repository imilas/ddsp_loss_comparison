# Sound-Matching Experiments

## Requirements

Please ensure the following libraries and tools are installed:

- `jax`, `flax`, `optax`
- `dawdreamer`
- `kiyamoto`
- `jaxwt`
- `scikit-image` (`skimage`)
- `audax`
- `sbx-rl` — for reinforcement learning experiments
- `Qdax`
- `transformers` — from Hugging Face
- `py-dtw` — for Dynamic Time Warping
- `PIX` — for SSIM with JAX
- `streamlit` — for running the survey interface
- `scikit-posthocs` — for confidence interval diagrams
- `kaleido` — for figure export

---

## How to Run Experiments

Install [**Marimo**](https://github.com/marimo-team/marimo) to view and run the experiment notebooks interactively.

---

## Important Notebooks

- **`random_experiment.py`**  
  Runs a random experiment with a given `loss_fn`, `learning_rate`, and `program_id`.  
  Used by `run_experiments.sh`.

- **`program_design.py`**  
  Explore and tweak synthesizer programs.

- **`loss_landscape_navigation.py`**  
  Visualize and experiment with different loss functions.

---

## Key Code Components

- **`./helpers/`**  
  Contains:
  - Loss function definitions
  - Plotting utilities
  - DSP helper functions

- **`programs.json`**  
  Contains all FAUST synthesizer programs.

---

## Running the Hearing Tests

1. Extract `hearing_test.tar` into the `hearing_test/` directory.
2. Run the survey via:

   ```bash
   python hearing_survey.py
