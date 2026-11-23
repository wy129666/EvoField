# EvoField: Spatiotemporal Prediction through Coupled Propagation and Correction

EvoField predicts future video frames and spatial fields by coupling two updates in latent space. The Directional Propagation Module aggregates neighborhood information using learnable directional candidates. The Signed Correction Module uses local spatial residuals and temporal differences to refine the prediction.

<p align="center">
  <img src="figures/taxibj_tradeoff.png" alt="Prediction errors and parameter counts on TaxiBJ" width="65%">
</p>

## 1. Environment setup

EvoField uses the OpenSTL training framework. From the repository root, create and activate the Conda environment and install the project:

```bash
conda env create -f environment.yml
conda activate OpenSTL
python -m pip install -e .
```

## 2. Data preparation

Dataset preparation scripts are in [`tools/prepare_data/`](tools/prepare_data/). Place the prepared datasets under `data/` in the repository root and pass that directory as `--data_root`. See [`openstl/datasets/`](openstl/datasets/) for the directory layout and input format expected by each loader.

## 3. Core code

```text
configs/                              Task configurations
openstl/
├── modules/mlo_tau_v3_tsr_modules.py   Directional propagation and signed correction
├── models/simvp_model.py              EvoField model integration
└── methods/tau.py                     Training objective and method wrapper
tools/
├── prepare_data/                      Data preparation scripts
├── train.py                           Training entry point
├── test.py                            Evaluation entry point
└── run_evofield_t20.py                 Navier–Stokes T20 entry point
```

## 4. Acknowledgements

EvoField builds on [OpenSTL](https://github.com/chengtan9907/OpenSTL). We thank its authors and contributors. See [LICENSE](LICENSE) for licensing terms.
