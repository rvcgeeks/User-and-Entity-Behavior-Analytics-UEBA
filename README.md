# User-and-Entity-Behavior-Analytics-UEBA
User and Entity Behavior Analytics by deep learning.  
Detecting users anomalous behaviors from users' daily records. 

内部威胁检测
## Details
All data were extracted from **CERT/R4.2** &nbsp;(*ftp://ftp.sei.cmu.edu/pub/cert-data*)

**Data**: data for detection.  

## Changelog since original branch:
- Kept User Data configurable
- Added Unsupervised learning for labelling Data instead of assuming that users start behaving anomalously after certain time
- Added code for sampling CERT Dataset in order to see analytics of other user as well
- Streamlined Step 3 and 4 (Action features and sequence models) for every user configured
- Added scripts for running entire UEBA pipeline.
- Added requirements.txt and made code compatable with latest version of python, tensorflow, pandas, numpy, matplotlib
- Re Implemented Action and Sequence Feature Models using ConvLSTM2D as per paper

## Dependent Libraries
- python greater than 3.10-64-bit
- numpy
- pandas
- scikit-learn
- tensorflow==2.15.0
- matplotlib
- pickle

## Usage
- The repository already contains CERT data extracted for some users, details of which are in Data/config.json, however you can create your own config and execute 0-Sample_Data_from_CERT.py separately
- Run run.bat or run.sh which runs in following steps: 1 -> 2 -> 3 -> 4 -> 5 -> 6 (unsupervised) -> 7 -> 8 (supervised).
- To reproduce old environment as in original branch, open up anaconda prompt and create old python environment "conda create --name myenv python=3.6", activate and execute "pip install -r requirements_old.txt"

*The provided features and deep learning models in this project are very simple samples, and you can add or create your own features and models based on this project.* : )

## Cite this work
This project is a part of our work that has been published in the ACM/IMS Transactions on Data Science. You can cite this work in your researches. 

ACM/IMS Transactions on Data Science, Volume 1, Issue 3 September 2020, Article No.: 16, pp 1–19 https://doi.org/10.1145/3374749

[Paper Link](https://dl.acm.org/doi/10.1145/3374749)
