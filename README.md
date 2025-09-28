# Surrogate Models for Crystal Plasticity  

## Introduction  
This repository contains the code accompanying my master's thesis:  
**_Surrogate Models for Crystal Plasticity – Predicting Stress, Strain, and Dislocation Density over Time._**

Crystal plasticity describes how crystalline materials such as metals or ceramics deform under applied loads. A key feature of this behavior is the **stress–strain curve**, which relates applied loading (≈ external force) to actual deformation (≈ change in shape, such as elongation when pulling on an object).  

The **Continuum Dislocation Dynamics (CDD)** framework, developed by Zoller and Schulz [1], provides a physical simulation to predict these deformation characteristics. While accurate, such simulations are **computationally expensive**.  

The goal of this work is to develop **surrogate models** that approximate these predictions at a fraction of the computational cost.  

Specifically, this repository contains experiments with:  
- Classical surrogate models (Response Surface Methodology, Gaussian Processes)  
- Tree-based approaches (LightGBM)  
- Recurrent neural networks (various LSTM architectures)  

## Motivation  
Running CDD simulations requires significant computational resources. Building the dataset alone took **months of simulations on multiple high-performance machines**. With surrogate models, we can:  
- Predict stress–strain time series from static material parameters  
- Reduce simulation time from hours to milliseconds  
- Enable faster exploration of material behavior  

This project therefore contributes towards **efficient material modeling** and has potential to save vast amounts of computational effort in future studies.  

## TL;DR  
- CDD simulations predict deformation behavior of crystalline materials.  
- They are accurate, but very costly to run.  
- This work develops **surrogate models** (LightGBM, LSTMs, etc.) to predict stress–strain curves as time series.  
- The surrogate models provide **fast and approximate predictions** based on material parameters.  

## Repository Structure  
```plaintext
/fccd       # Models and datasets
/notebooks  # Exploration and testing notebooks
            # → See notebooks/demo.ipynb for a usage example
/scripts    # Scripts for generating CDD simulations
```  

## Getting Started  

### Requirements  
- Python ≥ 3.8  
- Recommended: [conda](https://docs.conda.io/en/latest/) or [venv](https://docs.python.org/3/library/venv.html) for environment management  

### Installation  
Clone this repository:  
```bash
git clone https://github.com/your-username/surrogate-crystal-plasticity.git
cd surrogate-crystal-plasticity
```

Create and activate a virtual environment (conda example):  
```bash
conda create -n surrogate-cp python=3.9
conda activate surrogate-cp
```

Install dependencies:  
```bash
pip install -r requirements.txt
```

### Usage  
Run the demo notebook to see surrogate models in action:  
```bash
jupyter notebook notebooks/demo.ipynb
```

Or run training scripts directly (example with LightGBM):  
```bash
python fccd/train_lightgbm.py --config configs/lightgbm.yaml
```

### Datasets  
⚠️ **Note**: The full dataset generated for this thesis is **not publicly released** since it is part of ongoing research. Small sample datasets are included for demonstration.  

## Reference  
[1] Kolja Zoller and Katrin Schulz.  
“Analysis of single crystalline microwires under torsion using a dislocation-based continuum formulation”.  
*Acta Materialia*, 191 (2020), pp. 198–210.  
doi: [10.1016/j.actamat.2020.03.057](https://doi.org/10.1016/j.actamat.2020.03.057)  
