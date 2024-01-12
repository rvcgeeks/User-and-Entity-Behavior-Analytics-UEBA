import os,linecache,random

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

def count_line(files_in):
    file_open=open(files_in,'r')
    count=0
    for line in file_open:
        count+=1
    file_open.close()
    return count   

def Random_Num(index,number):
    '''
    Generate a list of index from index randomly
     index : the index of files
     number : the length of list generated
     return : a list of index  
    '''
    return random.sample(list(index),number)

def train_test_generate(data_all,label_all,train_data_save,train_label_save,test_data_save,test_label_save,rate=0.8):
    '''
    we extract 'rate' (the defaults is 0.8) percent data for training, and the rest for testing. 
    
    '''
    
    data_in=open(data_all,'r')
    label_in=open(label_all,'r')
    data_train=open(train_data_save,'wt')
    data_test=open(test_data_save,'wt')
    label_train=open(train_label_save,'wt')
    label_test=open(test_label_save,'wt')
    data_num=count_line(data_all)
    train_num=data_num*rate
    index=0
    for line in data_in:
        if index<train_num:
            data_train.writelines(line)
        else:
            data_test.writelines(line)
        index=index+1

    print('%s/%s for training' % (train_num+1, index))
    data_train.close()
    data_test.close()

    index=0
    for label in label_in:
        if index<train_num:
            label_train.writelines(label)
        else:
            label_test.writelines(label)
        index=index+1
    label_train.close()
    label_test.close()
    data_in.close()
    label_in.close()


if __name__ == "__main__":

    # NOTE Labels are required only for Supervised problem
    # ----- step 1 拼接的数据进行标签 (然后根据CERT数据集中的恶意行为进行标签的修正) Generate the label according to the source data
    # -------  the number behind the user is the firt day that the users' anomalous behavior is emerging accoriding to the source data in CERT dataset: ASSUMPTIONS from OBSERVATIONS ------------
    user_sets={'EDB0714':198,'TNM0961':198,'HXL0968':165}
    for username,nums in user_sets.items():
            user_path='Data/'+ username+'/Mix'
            file_save=user_path+'/label_loss.csv'
            file_read=user_path+'/Mix_loss.csv'
            Num_lines=count_line(file_read)
            fileopen=open(file_save,'wt')
            index=0
            while index<Num_lines:
                if index<nums-1:
                    fileopen.writelines('0,1\n')
                else:
                    fileopen.writelines('1,0\n')
                index=index+1
            fileopen.close()
    # -------- step 2 进行各个用户数据的混合：(mix all data) --------
    mix_path='Data/Mix'
    path_check(mix_path)
    file_save=open(mix_path+'/Mix_all_loss.csv','wt')
    label_save=open(mix_path+'/Mix_all_label.csv','wt')
    for username in user_sets:
        user_path='Data/'+ username+'/Mix'
        mix_loss=user_path+'/Mix_loss.csv'
        mix_label=user_path+'/label_loss.csv'
        loss_open=open(mix_loss,'r')
        label_open=open(mix_label,'r')
        for line in loss_open:
            file_save.writelines(line)
        loss_open.close()
        for line in label_open:
            label_save.writelines(line)
        label_open.close()

    user_sets=['Mix','EDB0714','TNM0961','HXL0968']
    for user in user_sets:
        print(user)
        if 'Mix' == user:
            working_folder='Data/'+user+'/'
            file_open=working_folder+'Mix_all_loss.csv'
            label_open=working_folder+'Mix_all_label.csv'
        else:
            working_folder='Data/'+user+'/Mix/'
            file_open=working_folder+'Mix_loss.csv'
            label_open=working_folder+'label_loss.csv'
        # ----- disorder all mixed data ------
        file_save=open(working_folder+'Disorder_loss.csv','wt')
        label_save=open(working_folder+'Disorder_label.csv','wt')
        line_counts=count_line(label_open)
        index=[n for n in range(0,line_counts)]
        Random_index=Random_Num(index,line_counts)
        # print(Random_index)
        for num in Random_index:
            line=linecache.getline(file_open,num+1)
            label=linecache.getline(label_open,num+1)
            file_save.writelines(line)
            label_save.writelines(label)
        file_save.close()
        label_save.close()
        # --------- generate tran and test data for MLP
        data_all=working_folder+'Disorder_loss.csv'
        label_all=working_folder+'Disorder_label.csv'
        data_train=working_folder+'train.csv'
        label_train=working_folder+'label_train.csv'
        data_test=working_folder+'test.csv'
        label_test=working_folder+'label_test.csv'
        train_test_generate(data_all,label_all,data_train,label_train,data_test,label_test,rate=0.7)












