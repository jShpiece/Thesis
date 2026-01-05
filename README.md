ARCH: Adaptive Reconstruction of Cluster Halos
=============================================

This repository contains ARCH, a research-grade gravitational lensing pipeline developed as part of my PhD work in astrophysics. ARCH is designed for joint reconstruction of galaxy cluster mass distributions using multiple gravitational lensing observables, including shear, first flexion, second flexion, and strong-lensing constraints.

This repository does NOT contain the thesis manuscript. It contains the scientific software on which the thesis is based.

---------------------------------------------------------------------

Scientific Scope
----------------

ARCH is intended for methodological development and controlled scientific analysis. The pipeline supports:

- Weak-lensing shear reconstruction
- First- and second-flexion modeling
- Joint inversion of multiple lensing signals
- Cluster-scale mass mapping
- Sensitivity to dark-matter substructure
- Validation against simulated and observational datasets

The emphasis is on flexibility and transparency rather than turnkey automation.

---------------------------------------------------------------------

Repository Structure
--------------------

Top-level layout (representative):

arch/
    Core pipeline modules
    - halo and mass-model objects
    - source catalog handling
    - lensing operators (shear, flexion, reduced fields)
    - reconstruction and optimization routines
    - shared utilities

pipelines/
    End-to-end workflows for specific analyses or datasets

simulations/
    Synthetic data generation and controlled tests

notebooks/
    Exploratory, diagnostic, and validation notebooks

tests/
    Unit tests and regression checks

data/
    Small example or derived datasets only
    (raw survey or simulation data are excluded)

figures/
    Diagnostic and publication-quality plots

requirements.txt
.gitignore

---------------------------------------------------------------------

Installation
------------

ARCH is written in Python.

Requirements:
- Python 3.10 or newer
- NumPy
- SciPy
- Astropy
- Matplotlib
- scikit-learn (used for clustering and validation utilities)

Install dependencies using:

    pip install -r requirements.txt

Development is strongly recommended inside a virtual environment.

---------------------------------------------------------------------

Usage Philosophy
----------------

ARCH is modular by design.

- Individual components are usable independently
- Pipelines are assembled by composing modules rather than relying on a single master script
- Many scripts reflect active research iteration and experimentation
- Notebooks serve as both validation tools and scientific records

Users are expected to understand gravitational lensing theory and reconstruction techniques before applying results.

---------------------------------------------------------------------

Scientific Caveats
------------------

- ARCH explores methodological space; not all configurations are physically equivalent
- Some routines assume weak-lensing limits and may fail in high-kappa regimes
- Treatment of degeneracies (e.g., the mass-sheet degeneracy) depends on boundary conditions and reconstruction context
- Outputs should always be interpreted in light of their assumptions

This code is intended for expert use.

---------------------------------------------------------------------

Relationship to the Thesis
--------------------------

The associated PhD dissertation:

- Motivates the design of ARCH
- Develops the theoretical background
- Evaluates reconstruction performance
- Interprets scientific results produced with this pipeline

This repository represents the implementation layer, not the archival record of the written thesis.

---------------------------------------------------------------------

Project Status
--------------

ARCH is under active development.

- APIs may change as reconstruction strategies evolve
- Stable analysis states corresponding to thesis milestones are tagged
- Some modules may be experimental or provisional

---------------------------------------------------------------------

Attribution
-----------

If you use this code or derivative methods in academic work, please cite the PhD dissertation and any associated publications describing the ARCH pipeline.