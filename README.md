# Parameter Inference for 21-cm Cosmology

Welcome to the computational module on parameter estimation for 21-cm cosmology. In this repository, we explore how to extract the mean ionization fraction of the intergalactic medium ($Q_{\mathrm{HII}}$) from noisy, forward-modeled mock observations. 

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
bash setenv.sh
```
*Note:* Once the script finishes executing, ensure you activate the newly created Conda environment before running any `.py` scripts or launching Jupyter to open the notebooks.

**3. Handling the Data Files**
To train the neural networks and run Approximate Bayesian Computation (ABC), you need a bank of simulated power spectra. Generating these from scratch can be computationally intensive. 

If you want to skip the simulation generation step and jump straight into the inference notebooks, simply copy the pre-computed `.npz` arrays from the `PRE_RUN_DATA_FILES` directory into your main working directory:
```bash
cp PRE_RUN_DATA_FILES/*.npz ./
```
If you prefer to run the forward models yourself, execute the relevant cells within the notebooks, which will write fresh `.npz` files directly to your root folder.

## Repository Contents

### Configuration & Environment
* **`setenv.sh`**: Shell script to build the required Conda environment.
* **`config.yaml`**: The shared configuration file controlling the simulation parameters, prior bounds ($Q_{\mathrm{HII}}$ limits), and target mock observation settings.

### Python Scripts
* **`sim_utils.py`**: The core simulation utilities. This contains the functions responsible for generating the toy 21-cm power spectrum realizations and handling the random seeds. 

### Jupyter Notebooks
* **`analyze_cosmic_variance.ipynb`**: Establishes our theoretical noise floor. You will fix the ionization fraction and vary the initial density seeds to quantify the irreducible sample variance in our $P_{21}(k)$ bins.
* **`analyze_abc_mcmc_sbi_2.ipynb`**: The main inference pipeline. This notebook compares three distinct methodologies:
  * **Approximate Bayesian Computation (ABC):** Rejection sampling based on Euclidean distance thresholds.
  * **MCMC:** Traditional likelihood-based inference used here as a baseline for posterior shape and convergence.
  * **Simulation-Based Inference (SBI):** Training a Neural Spline Flow (NSF) to model the complex posterior $P(\theta \mid d)$ directly from the simulation bank.
