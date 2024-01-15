#!/bin/bash
set +x


#python3 0-Sample_Data_from_CERT.py

python3 1-FeatureExtraction.py
python3 2-Training_Data_Generating.py

python3 3-Action_Sequence_Training.py
python3 4-Static_Feature_Training.py
python3 5-Role-Feature_and_Mix.py

python3 6-Unsupervised_Isolation_Forest_Training_and_Metrics.py

python3 7-Mix_Users_Data_and_Label.py
python3 8-Supervised_MLP_Training_and_Metrics.py
