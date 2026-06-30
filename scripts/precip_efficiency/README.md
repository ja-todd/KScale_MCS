#  Directory structure

```
precip_efficiency/
├─ compute_condensation_rates.py  -- calculate 2D field of condensation rates for a selected region (that exists in src.hp_models.REGIONS)
├─ mcs_condensation_rates.py      -- mask the 2D field of condensation rates to the tracked MCS objects 
├─ compute_storm_PE.py            -- compute MCS condensation rates, precip_flux and precip efficiencies from the output of mcs_condensation_rates.py
├─ submit.py                      -- smart slurm submitter for the python scripts in this directory
├─ cond_rates.sh                  -- batch script to run compute_condensation_rates.py for all models in src.hp_models.MODELS 
├─ mcs_PE.sh                      -- batch script to run compute_storm_PE.py for all models in src.hp_models.MODELS
├─ donefiles/                     -- chunk donefiles for the different models 
├─ figs/
├─ slurm/ 
├─── output/                      -- printed outputs from the python scripts run using submit.py   
├─── scripts/                     -- tasks submitted by the SLURM smart submitter 
├─── tasks/                       -- json tasks  
```

# Usage / workflow: 

## Can either run jobs invidually in the terminal using the smart slurm submitter: 

### 1. Activate the conda environment: 

```
conda activate hk26_env
```

### 2a. Compute the 2D condensation rates for a model using the smart SLURM submitter: 

```
python submit.py --model <model> --region <region> --script compute_condensation_rates
```

### 2b. Alternatively, use the simple shell script to submit these at the same time and let slurm handle the rest. 

```
bash cond_rates.sh 
```

### Outputs: ``` condensation_rate_<region>.zarr ``` in the data directory specified in src.hp_models.data_dir(). In ``` donefiles/<model>/```: ```init_<region>_condensation_rate.done ``` and ```chunk_*_condensation_rate.done ```. In slurm/output/: JOBID.err and JOBID.out which show what is printed to the command line in the running scripts. Used mainly to check whether chunks have been picked up correctly or not. 

### 3. Mask the computed condensation rates to the MCS storm objects. Can be run individually as follows: 

```
python compute_storm_PE.py --model <model> --init
```
### THEN: 
``` 
python compute_storm_PE.py --model <model> --run 
```

### Or alternatively, to do this for all models: 

```
bash mcs_PE.sh
```
### OUTPUT: in the data directory specified in src.hp_models.data_dir(): ``` mcs_precip_efficiency_<region>.zarr ```, which contains output PE, condensation rate, precip flux and basetime (3hrly times) as dimensions. 

### 4. To plot some of this output: 

```
plot_PE_output.py
```

### This produces 5 plots in figs/. They are: mcs_counts_durations.pdf; mcs_PE_since_initiation.pdf; PE_normalized_lifecycle.pdf; cr_contribution_lifecycle.pdf and cr_pr_MCS_lifecycle.pdf. 
