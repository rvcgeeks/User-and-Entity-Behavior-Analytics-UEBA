:: @echo off
set WD=%~dp0
cd "%WD%"

py 1-FeatureExtraction.py
py 2-Training_Data_Generating.py

py 3-Action_Sequence_Training.py
py 4-Static_Feature_Training.py
py 5-Role-Feature_and_Mix.py

py 6-Mix_Users_Data_and_Label.py

py 7.1-Unsupervised_Isolation_Forest_Training_and_Metrics.py
py 7.2-Supervised_MLP_Training_and_Metrics.py

pause
