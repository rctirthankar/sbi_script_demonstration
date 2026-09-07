import numpy as np
import multiprocessing as mp
import contextlib
import io
import time
import os
import shutil
import argparse
import emcee
import sim_utils

def log_prior(theta, q_min, q_max):
    qhii = theta[0]
    if q_min < qhii < q_max:
        return 0.0
    return -np.inf

def log_likelihood(theta, target_P21):
    qhii = theta[0]
    model_P21, model_kount = sim_utils.fast_eval_qhii(qhii)
    
    # Vectorized computation supports N-dimensional k-bins
    sigma = target_P21 / np.sqrt(model_kount)
    log_l = -0.5 * np.sum(((target_P21 - model_P21) / sigma) ** 2)
    return log_l

def log_probability(theta, target_P21, q_min, q_max):
    lp = log_prior(theta, q_min, q_max)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, target_P21)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MCMC Inference for P21.")
    parser.add_argument("-c", "--config", type=str, default="config.yaml")
    args = parser.parse_args()
    cfg = sim_utils.load_config(args.config)
    
    target_seed = cfg["target_seed"]
    target_QHII = cfg["target_QHII"]
    qhii_min = cfg.get("QHII_min", 0.0)
    qhii_max = cfg.get("QHII_max", 1.0)
    model_seed = cfg.get("model_seed", 111222333)
    
    print(f"Generating Target Observation for QHII = {target_QHII} (Seed: {target_seed})...")
    
    sim_utils.init_worker(cfg)
    with contextlib.redirect_stdout(io.StringIO()):
        target_result = sim_utils.run_realization((target_seed, target_QHII))
        
    target_P_k = target_result["P21"] # Retained as a full array
    print(f"Target P21 calculated: {target_P_k}")
    
    print(f"Pre-generating shared MUSIC snapshot for the MCMC model (Seed: {model_seed})...")
    shared_snap_path = os.path.join(cfg["base_dir"], f"SBI_MCMC_shared_snap_{model_seed}")
    
    with contextlib.redirect_stdout(io.StringIO()):
        sim_utils.generate_shared_music_snapshot(model_seed, shared_snap_path)

    ndim = 1
    nwalkers = cfg.get("mcmc_walkers", 10)
    nsteps = cfg.get("mcmc_steps", 150)
    max_workers = cfg.get("max_workers", 10)
    
    rng = np.random.default_rng(42)
    p0 = rng.uniform(qhii_min, qhii_max, size=(nwalkers, ndim))
    print(f"Starting MCMC with {nwalkers} walkers for {nsteps} steps...")
    t_start = time.time()
    
    with mp.Pool(processes=max_workers, initializer=sim_utils.init_mcmc_worker, initargs=(cfg, shared_snap_path)) as pool:
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, log_probability, 
            args=(target_P_k, qhii_min, qhii_max), 
            pool=pool
        )
        sampler.run_mcmc(p0, nsteps, progress=True)
        
    print(f"\nMCMC Completed in {time.time() - t_start:.2f}s")
    shutil.rmtree(shared_snap_path, ignore_errors=True)
    
    out_file = cfg.get("mcmc_output_file", "mcmc_samples.npz")
    np.savez(
        out_file, 
        chain=sampler.get_chain(), 
        target_QHII=target_QHII, 
        target_P_k=target_P_k,
        qhii_min=qhii_min,
        qhii_max=qhii_max
    )
    print(f"MCMC chain saved to {out_file}")