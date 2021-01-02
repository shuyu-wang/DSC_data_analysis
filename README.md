# DSC_data_analysis

This repo contains the code for our paper "a Machine Learning based Method for Differential Scanning Calorimetry Signal Analysis"

Differential scanning calorimetry (DSC) is a powerful method to study temperature induced phase transitions by monitoring the heat capacity changes. Traditional ways of DSC signal analysis require manual baseline subtraction and peak analysis, which inevitably leads to inefficiency and errors caused by human subjective bias. In this study, we propose an automated DSC peak recognition and baseline estimation method based on semi-supervised machine learning. We use the expectation-maximization algorithm to learn the exponential modified Gaussian mixture model and combine the least square method to complete the baseline allocation. We then perform least-square modeling of the net peak signal to interpretate the data. We demonstrate the method’s efficacy using three types of protein data measured by distinctive DSC instruments. It can effectively detect the baseline signals and accurately obtain the thermodynamic parameters from the peak signals for thermal characterization. This signal analysis method improves the speed and accuracy of the DSC signal interpretation.

# Dependency

- numpy
- pandas
- scipy
- matplotlib

# Usage

```
python Test_DSC.py
```

# Acknowledge

We'd like to express our gratitude towards all the colleagues and reviewers for helping us improve the paper. The project is impossible to finish without the open-source implementation.
