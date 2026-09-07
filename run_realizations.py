#!/usr/bin/env python3

import os
# Force all underlying C-extensions to use a single thread per worker
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import multiprocessing as mp
import numpy as np
import time
import argparse

import sim_utils

if __name__ == "__main__":
    # --- COMMAND LINE ARGUMENTS ---
    parser = argparse.ArgumentParser(description="Run 21cm Power Spectrum Realizations.")
    parser.add_argument(
        "-c", "--config", 
        type=str, 
        default="config.yaml", 
        help="Path to the input YAML configuration file."
    )
    args = parser.parse_args()

    # Load parameters from the specified YAML file
    print(f"Loading configuration from: {args.config}")
    cfg = sim_utils.load_config(args.config)
    
    # Reproducible set of random seeds
    n_realizations = cfg["n_realizations"]
    rng = np.random.default_rng(42)
    seeds = rng.integers(1, 2**31, size=n_realizations).tolist()
    
    # Draw QHII_mean from the Prior defined in the YAML
    # If qhii_min == qhii_max, this safely returns an array of identical fixed values
    qhii_min = cfg.get("QHII_min", 0.0)
    qhii_max = cfg.get("QHII_max", 1.0)
    qhii_samples = rng.uniform(qhii_min, qhii_max, size=n_realizations).tolist()
    
    kmin = cfg.get("kmin", 0.1)
    kmax = cfg.get("kmax", 1.0)
    nbins = cfg.get("nbins", 1)
    
    # Zip into tuples for the worker pool
    tasks = list(zip(seeds, qhii_samples))
    
    max_workers = cfg["max_workers"]
    print(f"Starting {len(tasks)} runs with {max_workers} workers...")
    if qhii_min == qhii_max:
        print(f"Running in Cosmic Variance Mode (Fixed QHII = {qhii_min})")
    else:
        print(f"Running in Prior Sampling Mode (QHII uniformly sampled between {qhii_min} and {qhii_max})")

    results = []
    t_start_all = time.time()
    
    output_file = cfg["output_file"]
    temp_file = output_file.replace(".npz", "_temp.npz")
    save_interval = cfg["save_interval"]

    # Calculate k_bins for the save array
    k_edges = np.logspace(np.log10(kmin), np.log10(kmax), nbins + 1)
    k_bins = np.sqrt(k_edges[:-1] * k_edges[1:])
    print (f"Using {nbins} logarithmically spaced k-bins from {kmin} to {kmax}")
    print (f"k_bins: {k_bins}")

    # Launch multiprocessing pool
    with mp.Pool(processes=max_workers, initializer=sim_utils.init_worker, initargs=(cfg,), maxtasksperchild=1) as pool:        
        for i, result in enumerate(pool.imap_unordered(sim_utils.run_realization, tasks), 1):
            results.append(result)
            
            # --- INTERMEDIATE ATOMIC SAVE ---
            if i % save_interval == 0 or i == len(tasks):
                current_results = sorted(results, key=lambda r: r["seed"])
                
                np.savez(
                    temp_file,
                    seeds=np.array([r["seed"] for r in current_results]),
                    zeta=np.array([r["zeta"] for r in current_results]),
                    QHII_mean=np.array([r["QHII_mean"] for r in current_results]),
                    P_all=np.array([r["P21"] for r in current_results]),
                    k_bins=k_bins,
                    kount=np.array([r["kount"] for r in current_results]),
                )
                
                os.replace(temp_file, output_file)
                print(f"--> [Progress: {i}/{len(tasks)}] Saved intermediate to {output_file}")

    print(f"\nAll {len(seeds)} realizations completed in {time.time() - t_start_all:.2f}s")