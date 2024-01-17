:: @echo off
set WD=%~dp0
cd "%WD%"

::py 0-Sample_Data_from_CERT.py

py 1-FeatureExtraction.py
py 2-Training_Data_Generating.py

py 3-Action_Sequence_Training.py
py 4-Static_Feature_Training.py
py 5-Role-Feature_and_Mix.py

py 6-Unsupervised_Isolation_Forest_Training_and_Metrics.py

py 7-Mix_Users_Data_for_MLP.py
py 8-Supervised_MLP_Training_and_Metrics.py

pause
