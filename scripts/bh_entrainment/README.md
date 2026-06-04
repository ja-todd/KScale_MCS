## (Planned) Directory structure (not yet complete)

```
bh_entrainment/
├─ microphysics.py  -- set of physical constants and standard atmospheric conversions from Julia Kukulies precip. efficiency project
├─ structure.py  -- developing the code to calculate moist static energy across a region 
├─ get_circular_env.py -- find the xkm radius MSE environment of updrafts 
├─ mcs_tracks_env_mse.py -- filter the MSE fields to the WAM tracks and get environmental MSE
├─ process.py -- averaging and weighting MSE fields 
├─ calc_entr_rate.py -- calculates the Becker and Hohenegger (2021) entrainment rate. 
├─ submit.py -- smart SLURM submitter for python 
```

