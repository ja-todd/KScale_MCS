# KScale MCS - MSc Dissertation

MSc dissertation work using global KScale simulations with the Met Office Unified Model, with various science configurations. 
Investigating shear-entrainment relationships in mesoscale convective systems over West Africa. 

## Structure
- `data/` — raw input data (not used currently)
- `scripts/` — processing scripts - currently entrainment proxies (proxy_entrainment) and entrainment rate calculated using Becker and Hohenegger's (2021) frozen MSE method
- `src/` — source code - files that are imported into many scripts - general utils/model configs and helpers. Much of this was written by Mark Muetzelfeldt during the 2026 KScale Hackathon
- `results/` — output files (currently not used)
- `docs/` — documentation (writing underway)
- `tests/` — tests

## Requirements
All of the things in the hk26_env.yaml file 
