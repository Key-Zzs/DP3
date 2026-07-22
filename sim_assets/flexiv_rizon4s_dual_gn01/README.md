# Dual Flexiv Rizon4s + GN01 simulation assets

The YAML files in this directory are committed source configuration. The URDF
and RMBench runtime bundle are generated and ignored. Rebuild them with:

```bash
conda activate dp3-rmbench
bash scripts/rmbench/flexiv/bootstrap_description.sh --force
```

The base poses and fixed head-camera extrinsic are explicitly marked as
simulation defaults. The home joint vectors come from the local real runtime
configuration, while the official description supplies the Rizon4s/GN01 joint,
limit, mesh, and TCP names.
