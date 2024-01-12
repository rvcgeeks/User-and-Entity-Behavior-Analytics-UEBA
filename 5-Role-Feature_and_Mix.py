import numpy as np 
import linecache
import matplotlib.pyplot as plt
import os
# from FeatureExtract import path_check


def path_check(path):
    '''
    Check the input path
    '''
    folders=os.path.exists(path)
    if not folders:
        print("Create new folders: ' %s ' successfully!"%(path))
        os.makedirs(path)
    else:
        print("The folder  ' %s ' exits, Please check!"%(path))
        pass
        # os._exit(0)

def role_feature():
    file_name='/feature/data_out.csv'
    user_sets=['ASD0577', 'AAL0706', 'ASM0575', 'GCG0951', 'AAV0450', 'TDG0962', 'CCM0136', 'FRR0832', 'KPP0452', 'ABM0845', 'MJB0588', 'WJD0576', 'ILH0958', 'FEB0306', 'JJB0700', 'AHD0848', 'LDM0587', 'MLG0475', 'NCK0295', 'YJV0699',]
    All_users_features=np.zeros((37))
    # paper 16:6 In this article, we define the average of action features selected from all colleagues under the same role as role features.
    for username in user_sets:
        path=username+file_name
        All_features=np.loadtxt(path,delimiter=',')
        Nums=len(All_features)
        Aver_features=np.zeros((37))
        # print(np.shape(Aver_features))
        for matrix in All_features:
            Aver_features=Aver_features+matrix
        Aver_features=Aver_features/Nums
        All_users_features=All_users_features+Aver_features
            # print (np.shape(matrix))
    role_features=All_users_features/len(user_sets)
    role_features=np.reshape(role_features,(1,37))
    role_feature_file='Data/'+ 'role_features_user_grp.csv'
    np.savetxt(role_feature_file,role_features,delimiter=',',fmt='%f')

def deviations_for_users(username):
    '''
    Calculate the  deviations between user's daily feature and role feature.
    '''

    # username='EDB0714'
    file_name='/feature/label_all.csv'
    role_feature_file='Data/'+ 'role_features_user_grp.csv'
    dd_weights_file='Data/'+'dd_weights_user_grp.csv'
    Role_features=np.loadtxt(role_feature_file,delimiter=',')
    user_features=np.loadtxt('Data/'+ username+file_name,delimiter=',')
    dd_weights=np.loadtxt(dd_weights_file,delimiter=',')
    All_loss=[]
    for matrix in user_features:
        deviations=np.square(np.multiply((Role_features-matrix),dd_weights))
        deviations=deviations.sum()/dd_weights.shape[0]
        All_loss.append(deviations)
    np.savetxt('Data/'+ username+'/role/loss.csv',All_loss,delimiter=',',fmt='%f')
    x_list=range(0,len(user_features))
    print(np.shape(All_loss))
    plt.figure()
    plt.plot(x_list,All_loss)
    plt.xlabel('Days',fontsize=14)
    plt.ylabel('WDD loss',fontsize=14)
    # plt.show()
    plt.savefig('Data/'+ username+'/role/loss.jpg')

def count_line(files_in):
    file_open=open(files_in,'r')
    count=0
    for line in file_open:
        count+=1
    file_open.close()
    return count   




# use the code below step by step.
if __name__ == "__main__":

    # ------ Calculate the role feature ---------
    # role_feature() 
    user_sets=['EDB0714','TNM0961','HXL0968']
    # ----- step 1  每个用户单独计算 role features 
    for user in user_sets:
        feature_role_path='Data/'+ user+'/role'
        path_check(feature_role_path)
        deviations_for_users(user)
    # ------------------------------------
    # ------ step 2 每个用户的各种偏差度进行拼接 (mix data)------
    for username in user_sets:
        user_path='Data/'+ username+'/Mix'
        path_check(user_path)
        file_feature='/feature/myloss_all.csv'
        file_sequence='/sequence/loss_all.csv'
        file_role='/role/loss.csv'
        file_save=user_path+'/Mix_loss.csv'
        # 加载各个特征的loss
        feature_loss=np.loadtxt('Data/'+ username+file_feature,delimiter=',')
        feature_loss=np.reshape(feature_loss,(len(feature_loss),1))
        sequence_loss=np.loadtxt('Data/'+ username+file_sequence,delimiter=',')
        sequence_loss=np.reshape(sequence_loss,(len(sequence_loss),1))
        role_loss=np.loadtxt('Data/'+ username+file_role,delimiter=',')
        role_loss=np.reshape(role_loss,(len(role_loss),1))
        mix_loss=np.concatenate((feature_loss,sequence_loss,role_loss),axis=1)
        np.savetxt(file_save,mix_loss,delimiter=',',fmt='%f')
        print(np.shape(role_loss))
        print(np.shape(mix_loss))
