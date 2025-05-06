requirements:

    - jax,flax,optax
    - dawdreamer
    - kiyamoto
    - jaxwt
    - skimage
    - audax
    - sbx-rl: for rl experiments
    - Qdax
    - hugging face transformers
    - dynamic time warping (py-dtw)
    - PIX (for ssim with jax)
    - streamlit (for survey)
    - scikit-posthocs (for CI digrams)
    - kaleido
# How to run experiments
Please install marimo to view and run the code.

# Important Notebooks:
random_experiment.py: runs a random experiment given a loss_fn, learning_rate, and program_id. It's used by run_experiments.sh 

program_design.py: use to play around with programs

loss_landscape_navigation: use to play around with losses

# Important Code  

./helpers/: Contains loss function definitions, plotting tools, and other useful DSP functions

programs.json: contains the FAUST programs

# Running the hearing tests:
unpack hearing_test.tar into the hearing_test directory, then run earing_survey.py



