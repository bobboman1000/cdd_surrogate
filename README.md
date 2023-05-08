# Surrogate Models to predict crystal plasticity

## Introduction 

This is the code to my master's thesis 'Surrogate Models for Crystal Plasticity - Predicting Stress, Strain and Dislocation Density over Time'.

Roughly speaking, the continuum dislocation dynamics (CDD) theory describes the deformation behavior of crystalline materials like metal or ceramics under loading. A characteristic property of the deformation behavior is the stress-strain curve: It describes the relation of loading (≈ an applied force) to the actucal deformation (≈ ratio of e.g. elongation when pulling on an object).

Zoller and Schulz [1] provide a simulation that implements the CDD theory. The simulation is able to predict the deformation characteristics such as the stress-strain curve (and more). However, the major downside of such simulations are their high computational costs. Instead, in this work I've built a surrogate model that makes approximate predictions at a much lower computational cost.

The work evaluates different approaches: it compares common surrogate models like RSM and GPs, a tree-based approach using LightGBM and various implementations of Long Short-Term Memory (LSTM) neural networks.

The two major parts of this work were 

(1) Building a dataset. As mentioned, simulations are computationally very costly, hence building a dataset is a huge task on its own. I've structurally generated a dataset using multiple large machines running over months. This alone has the potential to save hours of future work, hence this may become part of a future publication and will not be published at this point.

(2) I've built different time-series models aiming to generate a complete stress-strain time series based on a set of static input parameters. 


## Structure

/fccd contains all models and datasets

/notebooks contains a variety of different notebooks used for testing and exploration. For actual usage documentation see notebooks/demo.ipynb








[1] Kolja Zoller and Katrin Schulz. “Analysis of single crystalline microwires under torsion using a dislocation-based continuum formulation”. In: Acta Materialia 191 (June 2020), pp. 198–210. ISSN: 13596454. doi: 10.1016/j.actamat.2020.03.057. URL: https://linkinghub.elsevier.com/retrieve/pii/S1359645420302548 (visited on 12/10/2022).

