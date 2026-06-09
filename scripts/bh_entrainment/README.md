#  Directory structure

```
bh_entrainment/
├─ microphysics.py  -- set of physical constants and standard atmospheric conversions from Julia Kukulies precip. efficiency project
├─ calc_fmse.py  -- calculate 4D field of frozen moist static energy (FMSE) for the WAM region
├─ calc_mcs_fmse.py -- filter FMSE to WAM MCS tracks, with track_id as dimension
├─ calc_mcs_env_updraft_fmse.py -- compute per-track frozen MSE statistics for region-filtered MCS tracks. 
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

### 2. Compute per-MCS-track updraft and environmental FMSE: NOTE: The current 'environment' is at radius = 50km from the updrafts (obviously not going to stick with that), but 10km seemed to be too strict and 
### was returning NaN arrays a lot of the time. 

```
python submit.py --model <model_id> --region <region> --script calc_mcs_env_updraft_fmse

```

#### Again, initializes a different zarr store and submits the chunks which have absent donefiles. 

### 3. Compute the entrainment rate using the output zarr from calc_mcs_env_updraft_fmse.py

```
python compute_entrainment_rate.py --run --model <model_id> --region <region> (wam default)

```
#### Output: ```mcs_entr_rate_<region>.zarr```, dims: tracks: ; times_3h: ; pressure: currently no functionality to filter by surface, but should not be difficult to introduce. 
#### I have tested the output from the RAL3 run, it looks reasonable and values are broadly comparable to Becker and Hohenegger (2021), their Fig 6.

