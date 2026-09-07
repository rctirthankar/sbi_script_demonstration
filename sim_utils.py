import os
import tempfile
import shutil
import time
import yaml
import numpy as np
import sys
from contextlib import contextmanager

@contextmanager
def suppress_c_stdout():
    """Redirects OS-level stdout to /dev/null for C/Fortran extensions."""
    sys.stdout.flush()
    original_stdout_fd = os.dup(1)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull_fd, 1)
    try:
        yield
    finally:
        os.dup2(original_stdout_fd, 1)
        os.close(original_stdout_fd)
        os.close(devnull_fd)

def load_config(yaml_path="config.yaml"):
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)

def init_worker(shared_cfg):
    """Initializes globals relying on a single configuration dictionary."""
    global _CFG, _MUSIC_EXEC, _k_edges, _k_bins, _script, _two_lpt
    _CFG = shared_cfg
    _MUSIC_EXEC = os.path.join(_CFG["base_dir"], "music", "MUSIC")
    
    _k_edges = np.logspace(np.log10(_CFG["kmin"]), np.log10(_CFG["kmax"]), _CFG["nbins"] + 1)
    _k_bins = np.sqrt(_k_edges[:-1] * _k_edges[1:])
    
    import script as _script
    from script import two_lpt as _two_lpt

def _scratch_path(seed):
    scratch_base = _CFG.get("scratch_dir", _CFG["base_dir"])
    return os.path.join(scratch_base, f"SBI_seed{seed}_{os.getpid()}")

def _run_music_in_folder(seed):
    # Grab the RAM disk path
    scratch_base = _CFG.get("scratch_dir", _CFG["base_dir"])
    
    music_snaps_path = _scratch_path(seed)
    os.makedirs(music_snaps_path, exist_ok=True)
    
    # FORCE the temporary compilation/run directory into the RAM disk
    tmp_workdir = tempfile.mkdtemp(dir=scratch_base, prefix=f"music_run_seed{seed}_")
    original_dir = os.getcwd()
    omega_l = 1.0 - _CFG["omega_m"]
    
    try:
        os.chdir(tmp_workdir)
        with suppress_c_stdout():
            _two_lpt.run_music(
                _MUSIC_EXEC, _CFG["box"], np.array(_CFG["zlist"]), seed, music_snaps_path,
                _CFG["outroot"], _CFG["dx"], _CFG["omega_m"], omega_l, _CFG["omega_b"], 
                _CFG["h"], _CFG["sigma_8"], _CFG["ns"]
            )
    finally:
        os.chdir(original_dir)
        shutil.rmtree(tmp_workdir, ignore_errors=True)
    return seed, music_snaps_path

def _analyze_realization(seed, qhii_val, music_snaps_path):
    script_files_path = f"{music_snaps_path}/script_files"
    os.makedirs(script_files_path, exist_ok=True)
    music_snap = f"{music_snaps_path}/snap_000"
    
    music_data = _script.default_simulation_data(
        music_snap, script_files_path,
        sigma_8=_CFG["sigma_8"], ns=_CFG["ns"], omega_b=_CFG["omega_b"]
    )
    
    if _CFG["ngrid"] > music_data.default_ngrid:
        raise ValueError(f"Error: ngrid = {_CFG['ngrid']} is larger than snapshot {music_data.default_ngrid}")
        
    matter_fields = _script.matter_fields(music_data, _CFG["ngrid"], script_files_path, overwrite_files=False)
    ionization_map = _script.ionization_map(matter_fields, method='PC')
    fcoll_arr = matter_fields.get_fcoll_for_Mmin(_CFG["log10Mmin"])
    fcoll_mean = np.mean(fcoll_arr * (1 + matter_fields.densitycontr_arr))
    
    zeta = qhii_val / fcoll_mean
    qi_arr = ionization_map.get_qi(zeta * fcoll_arr)
    Delta_HI_arr = (1 - qi_arr) * (1 + matter_fields.densitycontr_arr)
    
    matter_fields.initialize_powspec()
    powspec_21cm_binned, kount = ionization_map.get_binned_powspec(
        Delta_HI_arr, _k_edges, units='mK', convolve=False
    )
    powspec_21cm_binned = _k_bins ** 3 * powspec_21cm_binned / (2 * np.pi ** 2)
    
    return {
        "seed": seed,
        "QHII_mean": qhii_val,
        "zeta": zeta,
        "P21": powspec_21cm_binned,
        "kount": kount,
    }

def run_realization(task_args):
    seed, qhii_val = task_args
    t_start = time.time()
    
    t0 = time.time()
    seed, music_snaps_path = _run_music_in_folder(seed)
    t_music = time.time() - t0
    
    t0 = time.time()
    result = _analyze_realization(seed, qhii_val, music_snaps_path)
    t_analysis = time.time() - t0
    
    shutil.rmtree(music_snaps_path, ignore_errors=True)
    result["time_music"] = t_music
    result["time_analysis"] = t_analysis
    result["time_total"] = time.time() - t_start
    
    print(f"seed {seed}: QHII_mean={result['QHII_mean']:.4f}, zeta={result['zeta']:.2f}, "
          f"P21={result['P21']}, t_total={result['time_total']:.2f}s")
    return result

def generate_shared_music_snapshot(seed, dest_path):
    """Consistently suppresses stdout during shared snapshot generation."""
    os.makedirs(dest_path, exist_ok=True)
    tmp_workdir = tempfile.mkdtemp(prefix=f"music_run_seed{seed}_")
    original_dir = os.getcwd()
    omega_l = 1.0 - _CFG["omega_m"]
    
    try:
        os.chdir(tmp_workdir)
        with suppress_c_stdout():
            _two_lpt.run_music(
                _MUSIC_EXEC, _CFG["box"], np.array(_CFG["zlist"]), seed, dest_path,
                _CFG["outroot"], _CFG["dx"], _CFG["omega_m"], omega_l, _CFG["omega_b"], 
                _CFG["h"], _CFG["sigma_8"], _CFG["ns"]
            )
    finally:
        os.chdir(original_dir)
        shutil.rmtree(tmp_workdir, ignore_errors=True)

def init_mcmc_worker(shared_cfg, shared_snap_path):
    init_worker(shared_cfg)
    global _cached_matter_fields, _cached_ionization_map, _cached_fcoll_arr, _cached_fcoll_mean
    
    script_files_path = f"{shared_snap_path}/script_files"
    os.makedirs(script_files_path, exist_ok=True)
    music_snap = f"{shared_snap_path}/snap_000"
    
    music_data = _script.default_simulation_data(
        music_snap, script_files_path,
        sigma_8=_CFG["sigma_8"], ns=_CFG["ns"], omega_b=_CFG["omega_b"]
    )
    
    _cached_matter_fields = _script.matter_fields(music_data, _CFG["ngrid"], script_files_path, overwrite_files=False)
    _cached_ionization_map = _script.ionization_map(_cached_matter_fields, method='PC')
    _cached_fcoll_arr = _cached_matter_fields.get_fcoll_for_Mmin(_CFG["log10Mmin"])
    _cached_fcoll_mean = np.mean(_cached_fcoll_arr * (1 + _cached_matter_fields.densitycontr_arr))
    _cached_matter_fields.initialize_powspec()

def fast_eval_qhii(qhii_val):
    zeta = qhii_val / _cached_fcoll_mean
    qi_arr = _cached_ionization_map.get_qi(zeta * _cached_fcoll_arr)
    Delta_HI_arr = (1 - qi_arr) * (1 + _cached_matter_fields.densitycontr_arr)
    
    powspec_21cm_binned, kount = _cached_ionization_map.get_binned_powspec(
        Delta_HI_arr, _k_edges, units='mK', convolve=False
    )
    powspec_21cm_binned = _k_bins ** 3 * powspec_21cm_binned / (2 * np.pi ** 2)
    return powspec_21cm_binned, kount