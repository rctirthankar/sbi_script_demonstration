# Parameter Inference for 21-cm Cosmology

Welcome to the computational module on parameter estimation for 21-cm cosmology. In this repository, we explore how to extract the mean ionization fraction of the intergalactic medium ($Q_{\mathrm{HII}}$) from noisy, forward-modelled mock observations.

Students will transition from basic descriptive statistics of cosmic variance through traditional sampling techniques, ultimately training a Neural Posterior Estimator (NPE) to perform Simulation-Based Inference (SBI).

## Getting Started

**1. Clone the Repository**

Pull this repository to your local machine:

```bash
git clone https://github.com/rctirthankar/sbi_script_demonstration.git
cd sbi_script_demonstration
```

**2. Environment Setup**

This repository relies on a dedicated Conda environment to cleanly manage complex scientific dependencies like `torch` and `sbi`. Execute the provided shell script to build the environment:

```bash
bash setup_env.sh
```

*Note:* The environment setup may take several minutes, and in some cases considerably longer, depending on your internet connection, operating system, and Conda configuration. Installing and resolving scientific Python packages such as PyTorch and `sbi` can be relatively time-consuming. **Do not interrupt the script if it appears to be making progress.**

Once the script finishes executing, ensure you activate the newly created Conda environment before running any `.py` scripts or launching Jupyter to open the notebooks.

If the environment has already been created, you do not need to run `setup_env.sh` again. You can activate it directly using:

```bash
conda activate sbi_script_env
```

**Disk space:** Allow approximately **10 GB of free disk space** for the Conda environment and its scientific Python dependencies. Additional space will be required for the simulation data described below. If you generate a large number of realizations yourself, the `.npz` files can become the dominant contribution to disk usage.

**3. Handling the Data Files**

To train the neural networks and run Approximate Bayesian Computation (ABC), you need a bank of simulated power spectra. Generating these from scratch can be computationally intensive.

If you want to skip the simulation generation step and jump straight into the inference notebooks, simply copy the pre-computed `.npz` arrays from the `PRE_RUN_DATA_FILES` directory into your main working directory:

```bash
cp PRE_RUN_DATA_FILES/*.npz ./
```

This is the recommended option if you are primarily interested in understanding the inference methods rather than generating the simulation bank yourself.

If you prefer to run the forward models yourself, execute the relevant cells within the notebooks, which will write fresh `.npz` files directly to your root folder.

*Note:* Before generating a large simulation bank, check the number of realizations requested in `config.yaml`. Increasing the number of realizations improves the sampling available to ABC and SBI, but also increases both the computational time and disk-space requirements. You may therefore want to start with a smaller number of realizations when testing the pipeline.

## Repository Contents

### Configuration & Environment

* **`setup_env.sh`**: Shell script to build the required Conda environment. The setup can take several minutes or longer because of the number of scientific dependencies that need to be installed.
* **`config.yaml`**: The shared configuration file controlling the simulation parameters, prior bounds ($Q_{\mathrm{HII}}$ limits), and target mock observation settings.

### Python Scripts

* **`sim_utils.py`**: The core simulation utilities. This contains the functions responsible for generating the toy 21-cm power spectrum realizations and handling the random seeds.
* **`run_mcmc.py`**: This runs a MCMC sampler assuming a Gaussian likelihood.
* **`run_realizations.py`**: This runs multiple realizations for the ionization field, either to understand cosmic variance or to generate samples for the SBI.

### Jupyter Notebooks

* **`analyze_cosmic_variance.ipynb`**: Establishes our theoretical noise floor. You will fix the ionization fraction and vary the initial density seeds to quantify the irreducible sample variance in our $P_{21}(k)$ bins.
* **`analyze_abc_mcmc_sbi.ipynb`**: The main inference pipeline. This notebook compares three distinct methodologies:

  * **Approximate Bayesian Computation (ABC):** Rejection sampling based on Euclidean distance thresholds.
  * **MCMC:** Traditional likelihood-based inference used here as a baseline for posterior shape and convergence.
  * **Simulation-Based Inference (SBI):** Training a Neural Spline Flow (NSF) to model the complex posterior $P(\theta \mid d)$ directly from the simulation bank.

## Recommended Workflow

If you are new to the repository, the following order is recommended:

1. **Set up the Conda environment** using `setup_env.sh`.
2. **Copy the pre-computed simulation files** from `PRE_RUN_DATA_FILES` if you want to avoid generating the simulations yourself.
3. Run **`analyze_cosmic_variance.ipynb`** to understand the impact of cosmic/sample variance on the simulated $P_{21}(k)$.
4. Run **`analyze_abc_mcmc_sbi.ipynb`** to compare ABC, MCMC, and SBI.
5. Once you are comfortable with the pipeline, experiment with the simulation parameters and prior ranges in `config.yaml`.

## Computational Considerations

The repository is designed primarily as a teaching and demonstration module rather than as a production-scale inference pipeline.

The computational cost depends strongly on the number of simulated realizations. In general:

* **Using the pre-computed `.npz` files** is the fastest way to get started.
* **Generating new realizations** can take substantially longer than loading the pre-computed data.
* **ABC** can require many simulations because only a fraction of proposed samples are accepted.
* **MCMC** requires repeated likelihood evaluations and may take some time depending on the number of samples and model settings.
* **SBI/NPE** requires an initial training stage. Training time depends on the size of the simulation bank and the neural-network configuration.

A typical laptop or desktop should be sufficient for the demonstration, although generating a large simulation bank or substantially increasing the neural-network training set may require more CPU time, RAM, and disk space.

## Reproducibility

The simulation utilities use explicit random seeds to control the generation of realizations. This allows the examples to be reproduced when the same configuration and random seeds are used.

If you modify the number of realizations, parameter ranges, or random seeds in `config.yaml`, your results may differ from those obtained using the pre-computed data.

## Troubleshooting

**Conda command not found**

If the `conda` command is not recognized, make sure that Conda/Miniconda/Anaconda is installed and initialized in your shell before running `setup_env.sh`.

**The environment setup appears to be taking a long time**

This is not necessarily an error. Conda may spend considerable time resolving dependencies and downloading packages. Give the process some time before interrupting it.

**Jupyter cannot find the required Python packages**

Make sure the dedicated Conda environment is activated before launching Jupyter:

```bash
conda activate sbi_script_env
jupyter notebook
```

If Jupyter is being launched from another environment, the required packages may not be available to the notebook kernel.

**The notebooks cannot find the `.npz` files**

Make sure the pre-computed files have been copied into the repository's main working directory:

```bash
cp PRE_RUN_DATA_FILES/*.npz ./
```

Also make sure that Jupyter was launched from the repository directory.

**Running the simulations uses too much disk space**

Check the number of realizations specified in `config.yaml`. Reduce this number when testing the pipeline, or remove old `.npz` files that are no longer required. It is a good idea to check available disk space before generating a large simulation bank.

## Notes for Students

The three inference approaches in this repository are intended to illustrate different philosophies of parameter inference:

* **MCMC** relies on an explicitly specified likelihood.
* **ABC** avoids evaluating the likelihood directly but requires a distance metric and an acceptance criterion.
* **SBI/NPE** learns the posterior from simulated examples and can be particularly useful when the likelihood is difficult or impractical to evaluate explicitly.

The goal is not simply to determine which method produces the tightest constraint on $Q_{\mathrm{HII}}$, but to understand **why the methods differ, how assumptions about the data enter the inference, and how simulation uncertainty and cosmic variance affect the resulting posterior.**
