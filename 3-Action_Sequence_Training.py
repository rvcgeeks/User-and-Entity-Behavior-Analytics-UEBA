# -*- coding: UTF-8 -*-
from tensorflow.keras.models import Sequential,load_model,Model
from tensorflow.keras.layers import Dense, Activation,Embedding,Conv2D,MaxPooling2D,Reshape,BatchNormalization,Dropout,Input,concatenate,GlobalAveragePooling2D,Flatten,LSTM,Conv3D
from tensorflow.keras.optimizers import Adam
import numpy as np 
import linecache
import os, json
import tensorflow as tf 
from tensorflow.keras import losses
from tensorflow.keras import losses,metrics
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
import matplotlib.pyplot as plt
import warnings 



# losses.mean_squared_error
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

# -----------------------------------------------

def train(files_train,labels_train,save_path,files_test,predict_save,action_length):
    """ Deep Learning model for users' actions
    # Params:
    ----
        files_train: training data (data for model training)
        labels_train: labels of training data
        save_path: the path of model that is going to be saved 
        files_test: test data 
        predict_save: the predicted labels of test data 
        action_length: the length of input data (include both training data and test data)
    """
    seq_length=4
    input_dims=seq_length*action_length
    # ---------------模型输入 (Model input)---------------
    x_train=np.loadtxt(files_train,delimiter=',')
    y_train=np.loadtxt(labels_train,delimiter=',')

    # ---------------模型结构 (model structure)--------------
    main_input=Input(shape=(input_dims,),dtype='float32',name='MainInput')
    Inputs=Reshape((seq_length,action_length))(main_input)
    print(Inputs)
    layer=LSTM(100,return_sequences=True,activation='tanh')(Inputs)
    layer=LSTM(160,return_sequences=False,activation='tanh')(layer)
    layer=Dense(action_length,activation='relu')(layer)
    print(layer)
    output=layer
    model=Model(inputs=main_input,outputs=output)
    model.compile(optimizer='adam',loss='mse',metrics=['binary_accuracy'])
   
    # ------------------模型保存 (model saving)------------------
    tensorboard_path='Data/'+ USERNAME +'/Model/Action'
    # 训练可视化 (visualization)
    tbCallback=TensorBoard(log_dir=tensorboard_path,histogram_freq=0,write_graph=True,write_grads=True,write_images=True,embeddings_freq=0,embeddings_layer_names=None,embeddings_metadata=None)
    # 保存最好的模型 (save the best model)
    checkpoint=ModelCheckpoint(save_path,monitor='loss',verbose=0,save_best_only='True')
    # -----------------模型训练 (model training)--------------------
    model.fit(x_train,y_train,batch_size=6,epochs=25,shuffle=True,callbacks=[tbCallback,checkpoint])
    # -----------------数据预测 (data prediction)---------------------
    x_test=np.loadtxt(files_test,delimiter=',')
    y_pred=model.predict(x_test)
    np.savetxt(predict_save,y_pred,fmt='%f',delimiter=',')

    
    # # --------------------------- end ------------
    # pauses=input("save this file? Y/N : \n ")
    # if pauses=='Y'or pauses=='y':
    # #     model.save(save_path)
    # #     print("This Model have saved! ")
    #     np.savetxt(predict_save,y_pred,fmt='%f',delimiter=',')
    #     print("Predict files Model have saved! ")
    # else:
    #     print("Stop!")
    #     exit(0)


# ----------------------------- test ----------------
def test(files_test,save_path,predict_save):
    # labels_name='Data/'+username+'/feature/label_test.csv'
    x_test=np.loadtxt(files_test,delimiter=',')
    model=load_model(save_path)
    pred=model.predict(x_test)
    print(np.shape(pred))
    # pred 矩阵中如果大于 0.5 保持不变，否则置 0）
    pred=np.where(pred>0,pred,0)
    np.savetxt(predict_save,pred,fmt='%f',delimiter=',')

def Calculate_deviations(files_test,label_test,save_path,loss_save,figure_save,action_length):
    x_test=np.loadtxt(files_test,delimiter=',')
    y_test=np.loadtxt(label_test,delimiter=',')
    # print (np.shape(x_test))
    # print(np.shape([x_test[0]]))
    # print(len(x_test))
    # exit (0)
    model=load_model(save_path)
    # exit(0)
    All_loss=[]
    x_list=range(0,len(x_test))
    for i in range (len(x_test)):
        x_small_test=np.reshape(x_test[i],(1,4*action_length))
        y_small_test=np.reshape(y_test[i],(1,action_length))
        loss,acc=model.evaluate(x=x_small_test,y=y_small_test,verbose=0)
        All_loss.append(loss)
    np.savetxt(loss_save,All_loss,fmt='%f',delimiter=',')
    # --------- draw pictures 
    plt.figure()
    plt.plot(x_list,All_loss)
    plt.xlabel('Days',fontsize=14)
    plt.ylabel('WDD loss',fontsize=14)
    # plt.show()
    plt.savefig(figure_save)
    # exit(0)

# ---------------------------------- for all  users --------------------------------
if __name__ == "__main__":


    # user_sets={'EDB0714':29,'TNM0961':32,'HXL0968':33}

    # -------- run model for every user separately or it will report errors because of the cache ----------- 
    with open('Data/config.json', 'r') as fh:
        CONFIG = json.load(fh)
    for username,subconfig in CONFIG['monitor'].items():
        USERNAME=username
        action_length=subconfig['max_seq_len']
        print(USERNAME, action_length)

        folder='Data/'+ USERNAME +'/Model/Action/'
        save_path=folder+'model.h5'
        files_train='Data/'+ USERNAME+'/sequence/'+'data_train.csv'
        labels_train='Data/'+USERNAME+'/sequence/'+'label_train.csv'
        files_test='Data/'+USERNAME+'/sequence/'+'data_test.csv'
        label_test='Data/'+USERNAME+'/sequence/'+'label_test.csv'
        predict_save='Data/'+USERNAME+'/sequence/'+'predict.csv'
        loss_save='Data/'+USERNAME+'/sequence/'+'loss.csv'
        figure_save='Data/'+USERNAME+'/sequence/'+'loss.jpg'
        path_check(folder)

        actions_sequence_file='Data/'+USERNAME+'/sequence/'+'actions_sequence.csv'

        # ------------------------- 运行模型 (trining model)-------------------------
        train(files_train,labels_train,save_path,files_test,predict_save,action_length)


        # ---------------- calculate all deviations for all data(train+test)
        predict_save='Data/'+USERNAME+'/sequence/'+'predict_all.csv'
        files_all='Data/'+USERNAME+'/sequence/'+'data_all.csv'
        labels_all='Data/'+USERNAME+'/sequence/'+'label_all.csv'
        loss_save='Data/'+USERNAME+'/sequence/'+'loss_all.csv'
        figure_save='Data/'+USERNAME+'/sequence/'+'loss_all.jpg'

        Calculate_deviations(files_all,labels_all,save_path,loss_save,figure_save,action_length)

    # -----------------------------------------
