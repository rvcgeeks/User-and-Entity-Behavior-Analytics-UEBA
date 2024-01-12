I choose three users' data for test. And we considered three types of features which includes action features, action sequence and role features.
---------------------------------------
Users： 'EDB0714','TNM0961' and 'HXL0968'
Mix: Files mixed with three types of features for the next process.
role_feature_user_grp.csv: the average of actions features of  all users under the role of "ProductionLineWorker".
dd_weights_user_grp.csv: weights for features for calculating WDD for role and feature models for this particular user group.

---------------------------------------
Every user's files:

            * ./ device.csv,email.csv,file.csv,http.csv,logon.csv,psychometric.csv are all extracted from CERT/R4.2 directly.

            * new: newly arranged data which came from above.

            * feture/sequence:  
                        data_out.csv,data_all.csv, label.csv are total feature data (generated from files in 'new' folder).
                        data_train.csv,test.csv,label_train.csv,label_test.csv are generated from data_all.csv and label_all.csv 
                        for deep model.
                        predict.csv,predict_all.csv are predictions from trained feature model.
                        loss_all.csv,my_loss.csv are two types of loss caculated by trained model.

            * Model:
                        Action: deep model and tensorboard file for action sequence.
                        Feature: deep model and tensorboard file for action features. 
                        Role_loss: loss caculated by role features.

            * Mix:
                        We spliced these three losses together.
