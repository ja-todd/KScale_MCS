## (Planned) Directory structure (not yet complete)

```
bh_entrainment/
├─ microphysics.py  -- set of physical constants and standard atmospheric conversions from Julia Kukulies precip. efficiency project
├─ calc_fmse.py  -- calculate 4D field of frozen moist static energy (FMSE) for the WAM region
├─ calc_mcs_fmse.py -- filter FMSE to WAM MCS tracks, with track_id as dimension
├─ process.py -- averaging and weighting MSE fields 
├─ calc_entr_rate.py -- calculates the Becker and Hohenegger (2021) entrainment rate. 
├─ submit.py -- smart SLURM submitter for python: can currently handle calc_fmse.py and calc_mcs_fmse.py by specifying --script script_name (no .py required)
```

