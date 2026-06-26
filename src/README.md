#  Directory structure

```
precip_efficiency/
├─ __init__.py              -- ensures that the setuptools.find_packages can find the modules in this directory 
├─ hp_utils.py              -- general helpers for model data which are on the healpix grid 
├─ hp_models.py             -- data directory, path constructors for healpix models in the UM catalog
├─ microphysics.py          -- used predominantly for the scripts/precip_efficiency portion of the project; basic microphysical quantities and conversions. Constructed by Julia Kukulies, see https://github.com/JuliaKukulies/precip_efficiency . 
             
```