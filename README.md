# KScale MCS - MSc Dissertation

MSc dissertation work using global KScale simulations with the Met Office Unified Model, with various science configurations.
Investigating shear-entrainment relationships in mesoscale convective systems over West Africa, and the relationship to precipitation efficiency.

## Structure

- `data/` — raw input data (not used currently)
- `scripts/` — processing scripts - currently entrainment proxies (```proxy_entrainment/```) and entrainment rate calculated using Becker and Hohenegger's (2021) frozen MSE method (```bh_entrainment/```), as well as precipitation efficiency (```precip_efficiency/```) using the state-variable (SV) method from Kukulies et al. (2024). Comparison of general MCS statistics e.g. lifetimes, track densities can be found in ```mcs_stats/```.
- `src/` — files that are imported into many scripts. Contains utils/model configs and helpers for loading/working with healpix data (``hp_utils, hp_models``), and plotting (``plot_utils``). `hp_utils` is largely unaltered from the work done Mark Muetzelfeldt during the 2026 KScale Hackathon, `hp_models` is significantly altered, and `plot_utils` is entirely mine.
- `results/` — output files (currently not used)
- `docs/` — documentation (writing underway, will be filled at project close)
- `tests/` — tests

## Requirements

All of the things in the hk26_env.yaml file.
