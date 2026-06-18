#  Directory structure

```
bh_entrainment/
├─ microphysics.py  -- set of physical constants and standard atmospheric conversions from Julia Kukulies precip. efficiency project
├─ calc_fmse.py  -- calculate 4D field of frozen moist static energy (FMSE) for the WAM region
├─ calc_mcs_fmse.py -- filter FMSE to WAM MCS tracks, with track_id as dimension
├─ calc_mcs_env_updraft_fmse.py -- compute per-track frozen MSE statistics for region-filtered MCS tracks, or statistics for all (including MCS) updrafts and associated environments 
├─ compute_entrainment_rate.py -- calculates the Becker and Hohenegger (2021) entrainment rate from the initial zarr output by calc_mcs_env_updraft_fmse.py 
├─ submit.py -- smart SLURM submitter for python: can currently handle calc_fmse.py, calc_mcs_fmse.py and calc_mcs_env_updraft_fmse.py by specifying --script script_name (no .py required)
```

## Usage/workflow

### All commands are run from KScale_MCS/scripts/bh_entrainment/. Activate hk26_env first. 

```
conda activate hk26_env
```

### 1. Compute 4-D frozen MSE: 

```
python submit.py --model <model_id> --region <region> --script calc_fmse
```

#### This initializes the zarr store if needed, then submits the chunks which have absent donefiles. 

### 2a. EITHER Compute per-MCS-track updraft and environmental FMSE: NOTE: The default radius = 50km from the updrafts; 10km seemed to be too strict and \n  was returning NaN arrays a lot of the time. 

```
python submit.py --model <model_id> --script calc_mcs_env_updraft_fmse --region <region> --radius <radius(km)> 

```

### Output: ```mcs_env_updraft_fmse_<region>_<radius>.zarr```

### 2b. OR: Computes updraft and environmental FMSE for all updrafts, not just MCS updrafts (as in Becker and Hohenegger, 2021).

```
python submit.py --model <model_id> --script calc_mcs_env_updraft_fmse --region <region> --radius <radius(km)> --no-mcs
```
### Output:  ```env_updraft_fmse_<radius>_<region>_.zarr```



#### Again, initializes a different zarr store and submits the chunks which have absent donefiles. 

### 3a. Compute the entrainment rate using the output zarr from calc_mcs_env_updraft_fmse.py, with an environmental radius matching one of the chosen radii from above, and can filter to a specific surface (land/ocean/all, default=all)

```
python compute_entrainment_rate.py --init --model <model_id> --region <region> (wam default) --radius <radius(km)> --surface <surface>
python compute_entrainment_rate.py --run --model <model_id> --region <region> (wam default) --radius <radius(km)> --surface <surface>

```
#### Output: ```mcs_entr_rate_<region>_<radius>_[<surface>].zarr```, dims: tracks: ; times_3h: ; pressure:

### 3b. There is also an option to compute the entrainment rate for all updrafts and associated environments at different radii, with the above --no-mcs argument. It does not *yet* have functionality to filter by surface (logic is slightly different with the land/sea mask). 

```
python compute_entrainment_rate.py --init --model <model_id> --region <region> (wam default) --radius <radius(km)> --no-mcs
python compute_entrainment_rate.py --run --model <model_id> --region <region> (wam default) --radius <radius(km)> --no-mcs

```
#### Output: ```entr_rate_<region>_<radius>.zarr```, dims: (time: ; pressure: ). 



  

