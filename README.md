# Estimating affective polarization on a social network
This repository contains information about the data and code used in the paper "Estimating affective polarization on a social network" by Marilena Hohmann and Michele Coscia.

## Code and data availability
The code needed to replicate the synthetic experiments (Figs 2–4, Fig S1) is included here.

The Twitter data set contains personally identifiable data relating to natural persons and the processing took place within the European Union. The Twitter data is, therefore, subject to the General Data Protection Regulation (GDPR). Due these regulations, we cannot make this data set publicly available.

## File overview
The following implementation files need to be imported to run the experiments:

- `affective_polarization.py:` Implements the affective polarization measure proposed in the paper.
- `alternative_measures.py`: Contains code to calculate the alternative measures we compare to.
- `synthetic_experiments.py`: Setup for the synthetic experiments we run. This includes code to generate and modify networks, opinions, disagreement values, and hostility values as described in the paper.

The following files reproduce the synthetic experiments. They compare the different measures' sensitivity to:

- `01_fig_2.py`: the hostility distribution (affective component).
- `02_fig_3.py`: the disagreement distribution (affective component).
- `03_fig_4.py:`: the social distance component.
- `04_fig_s1.py`: the strength of opinions.

## Requirements
To run the code, the following packages have to be installed:

- `networkx==3.4.2`
- `numpy==2.2.4`
- `pandas==2.2.3`
- `scikit-learn==1.6.1`
- `scipy==1.15.2`
- `torch==2.6.0`
- `torch-geometric==2.6.1`