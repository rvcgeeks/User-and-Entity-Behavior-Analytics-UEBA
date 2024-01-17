# -*- coding: UTF-8 -*-
import os, json
import linecache
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model,Model
from keras.layers import Input,Reshape,LSTM,ConvLSTM2D,MaxPooling2D,Dropout,Flatten,Dense
from keras.optimizers import Adam
from keras.callbacks import TensorBoard,ModelCheckpoint


def path_check(path):
    folders=os.path.exists(path)
    if not folders:
        print("Create new folders: ' %s ' successfully!"%(path))
        os.makedirs(path)
    else:
        print("The folder  ' %s ' exits, Please check!"%(path))
        os._exit(0)

def count_line(files_in):
    file_open=open(files_in,'r')
    count=0
    for line in file_open:
        count+=1
    return count   

def data_generator(files_name,num_days=5,batch_size=30):
    # 删除最后一行数据，因为可能会有缺漏
    counts=count_line(files_name)-1
    real_counts=counts-num_days+1
    rounds=int(real_counts/batch_size)
    # print (real_counts)
    while (1):   
        for i in range(rounds):
            data_batch=[]
            label_batch=[]
            for j in range (batch_size):
                data_index=i+j+1
                label_index=data_index+4
                data_one=linecache.getline(files_name,data_index).strip().split(',')+linecache.getline(files_name,data_index+1).strip().split(',')+linecache.getline(files_name,data_index+2).strip().split(',')+linecache.getline(files_name,data_index+3).strip().split(',')
                labe_data=linecache.getline(files_name,label_index).strip().split(',')
                data_batch.append(data_one)
                label_batch.append(labe_data)
            # print(np.shape(labe_data))
            # print (data_one)
            # exit (0)
            data_batch=np.array(data_batch)
            label_batch=np.array(label_batch)
            data_out=np.reshape(data_batch,(batch_size,seq_length*5*8*1))
            label_out=np.reshape(label_batch,(batch_size,48))
            i=i+batch_size+1
            yield data_out,label_out



def train(files_train,labels_train,save_path,files_test,predict_save):
    """ Deep Learning model for users' features
    # Params:
    ----
        files_train: training data (data for model training)
        labels_train: labels of training data
        save_path: the path of model that is going to be saved 
        files_test: test data 
        predict_save: the predicted labels of test data 
        
    """
    # ---------------模型输入 (input)---------------
    x_train=np.loadtxt(files_train,delimiter=',')
    y_train=np.loadtxt(labels_train,delimiter=',')

    # ---------------模型结构 (model structure)---------------
    input_dims=seq_length*features_len
    '''
    # LSTM Model
    main_input=Input(shape=(input_dims,),dtype='float32',name='MainInput')
    Inputs=Reshape((4,37))(main_input)
    layer=LSTM(100,return_sequences=True,activation='tanh')(Inputs)
    layer=LSTM(160,return_sequences=False,activation='tanh')(layer)
    layer=Dense(37,activation='relu')(layer)
    '''
    # Convolutional LSTM Model
    input_dims_fixed=seq_length*5*8*1
    x_train=np.pad(x_train, ((0, 0), (0, input_dims_fixed-input_dims)), mode='constant')

    main_input=Input(shape=(input_dims_fixed,),dtype='float32',name='MainInput')
    Inputs=Reshape((seq_length,5,8,1))(main_input)
    layer=ConvLSTM2D(24,kernel_size=(2,3),return_sequences=True,activation='elu')(Inputs)
    layer=ConvLSTM2D(128,kernel_size=(2,3),return_sequences=True,activation='tanh')(layer)
    layer=ConvLSTM2D(48,kernel_size=(2,3),return_sequences=False,activation='tanh')(layer)
    layer=MaxPooling2D(pool_size=(2,2),data_format='channels_last')(layer)
    layer=Dropout(0.5)(layer)
    layer=Flatten()(layer)
    layer=Dense(features_len,activation='relu')(layer)

    output=layer
    model=Model(inputs=main_input,outputs=output)
    model.compile(optimizer='adam',loss='mse',metrics=['binary_accuracy'])
    print(model.summary())

    # ------------------模型保存 (save the model)------------------
    tensorboard_path='Data/'+ USERNAME +'/Model/Feature'
    tbCallback=TensorBoard(log_dir=tensorboard_path,histogram_freq=0,write_graph=True,write_grads=True,write_images=True,embeddings_freq=0,embeddings_layer_names=None,embeddings_metadata=None)
    checkpoint=ModelCheckpoint(save_path,monitor='loss',verbose=0,save_best_only='True')
    model.fit(x_train,y_train,batch_size=6,epochs=30,shuffle=True,callbacks=[tbCallback,checkpoint])
    
    # -----------------数据预测 (data prediction )---------------------
    x_test=np.loadtxt(files_test,delimiter=',')

    # If ConvLSTM2D
    x_test=np.pad(x_test, ((0, 0), (0, input_dims_fixed-input_dims)), mode='constant')

    y_pred=model.predict(x_test)
    np.savetxt(predict_save,y_pred,fmt='%f',delimiter=',')


def retrain(files_train,labels_train,save_path,files_test,predict_save):
    """
    model retraining 
    """
    x_train=np.loadtxt(files_train,delimiter=',')
    y_train=np.loadtxt(labels_train,delimiter=',')
    model=load_model(save_path)

    # ----------------- load the model-----
    tensorboard_path='Data/'+ USERNAME +'/Model/Feature'
    tbCallback=TensorBoard(log_dir=tensorboard_path,histogram_freq=0,write_graph=True,write_grads=True,write_images=True,embeddings_freq=0,embeddings_layer_names=None,embeddings_metadata=None)
    checkpoint=ModelCheckpoint(save_path,monitor='loss',verbose=0,save_best_only='True')

    # If ConvLSTM2D
    input_dims=seq_length*features_len
    input_dims_fixed=seq_length*5*8*1
    x_train=np.pad(x_train, ((0, 0), (0, input_dims_fixed-input_dims)), mode='constant')

    model.fit(x_train,y_train,batch_size=6,epochs=800,shuffle=True,callbacks=[tbCallback,checkpoint])
    
    # ---------------- add test -------
    x_test=np.loadtxt(files_test,delimiter=',')

    # If ConvLSTM2D
    x_test=np.pad(x_test, ((0, 0), (0, input_dims_fixed-input_dims)), mode='constant')

    y_pred=model.predict(x_test)
    np.savetxt(predict_save,y_pred,fmt='%f',delimiter=',')

# ----------------------------- test ----------------
def test(files_test,save_path,predict_save):
    
    # labels_name='Data/'+username+'/feature/label_test.csv'
    x_test=np.loadtxt(files_test,delimiter=',')
    model=load_model(save_path)

    # If ConvLSTM2D
    input_dims=seq_length*features_len
    input_dims_fixed=seq_length*5*8*1
    x_test=np.pad(x_test, ((0, 0), (0, input_dims_fixed-input_dims)), mode='constant')

    pred=model.predict(x_test)
    print(np.shape(pred))
    # pred 矩阵中如果大于 0.5 保持不变，否则置 0）
    pred=np.where(pred>0.47,pred,0)
    np.savetxt(predict_save,pred,fmt='%f',delimiter=',')
    return pred


#  define my loss
def my_loss_forFeatures(label_test,predict_save,myloss_save,figure_save,dd_weights_fn):
    y_true=np.loadtxt(label_test,delimiter=',')
    y_pred=np.loadtxt(predict_save,delimiter=',')
    dd_weights=np.loadtxt(dd_weights_fn,delimiter=',')
    batch_size=np.shape(y_pred)[0]
    All_loss=[]
    for i in range(batch_size):
        # times=np.square(((y_pred[i]+0.6).astype(np.int32)-y_true[i]))
        # ---- Original 
        times=np.square(np.multiply((y_pred[i]-y_true[i]),dd_weights))
        sums=times.sum()/dd_weights.shape[0]
        All_loss.append(sums)
    x_list=range(0,batch_size)    
    np.savetxt(myloss_save,All_loss,fmt='%f',delimiter=',')
    # --------- draw pictures 
    plt.figure()
    plt.plot(x_list,All_loss)
    plt.xlabel('Days',fontsize=14)
    plt.ylabel('WDD loss',fontsize=14)
    # plt.show()
    plt.savefig(figure_save)
    # print(np.shape(loss))

def Calculate_deviations(files_test,label_test,save_path,loss_save,figure_save):
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
    input_dims=seq_length*features_len
    input_dims_fixed=seq_length*5*8*1
    for i in range (len(x_test)):
        x_small_test=np.reshape(x_test[i],(1,input_dims))

        # If ConvLSTM2D
        x_small_test=np.pad(x_small_test, ((0, 0), (0, input_dims_fixed-input_dims)), mode='constant')

        y_small_test=np.reshape(y_test[i],(1,37))
        loss,acc=model.evaluate(x=x_small_test,y=y_small_test,verbose=0)
        All_loss.append(loss)
    np.savetxt(loss_save,All_loss,fmt='%f',delimiter=',')
    # --------- draw pictures 
    plt.figure()
    plt.plot(x_list,All_loss)
    plt.xlabel('Days',fontsize=14)
    plt.ylabel('MSE loss',fontsize=14)
    # plt.show()
    plt.savefig(figure_save)
    # exit(0)





if __name__ == "__main__":
    
    # user_sets={'EDB0714':29,'TNM0961':32,'HXL0968':33}

    # -------- run model for every user separately or it will report errors because of the cache ----------- 
    with open('Data/config.json', 'r') as fh:
        CONFIG = json.load(fh)
    seq_length=4
    features_len=37
    for username,subconfig in CONFIG['monitor'].items():
        USERNAME=username
        print(USERNAME)
        folder='Data/'+ USERNAME +'/Model/Feature/'
        save_path=folder+'model.h5'
        files_train='Data/'+USERNAME+'/feature/'+'data_train.csv'
        labels_train='Data/'+USERNAME+'/feature/'+'label_train.csv'
        files_test='Data/'+USERNAME+'/feature/'+'data_test.csv'
        predict_save='Data/'+USERNAME+'/feature/'+'predict.csv'
        label_test='Data/'+USERNAME+'/feature/'+'label_test.csv'
        loss_save='Data/'+USERNAME+'/feature/'+'loss.csv'
        figure_save='Data/'+USERNAME+'/feature/'+'loss.jpg'
        figure_my_save='Data/'+USERNAME+'/feature/'+'myloss.jpg'
        myloss_save='Data/'+USERNAME+'/feature/'+'myloss.csv'

        train(files_train,labels_train,save_path,files_test,predict_save)
        # retrain(files_train,labels_train,save_path,files_test,predict_save)

        # ------------ calculate all deviations for all data(train+test)
        predict_save='Data/'+USERNAME+'/feature/'+'predict_all.csv'
        files_all='Data/'+USERNAME+'/feature/'+'data_all.csv'
        labels_all='Data/'+USERNAME+'/feature/'+'label_all.csv'
        loss_save='Data/'+USERNAME+'/feature/'+'loss_all.csv'
        figure_save='Data/'+USERNAME+'/feature/'+'loss_all.jpg'
        figure_my_save='Data/'+USERNAME+'/feature/'+'myloss_all.jpg'
        myloss_save='Data/'+USERNAME+'/feature/'+'myloss_all.csv'
        test(files_all,save_path,predict_save)
        Calculate_deviations(files_all,labels_all,save_path,loss_save,figure_save)
        dd_weights_fn='Data/'+'/dd_weights_user_grp.csv'
        my_loss_forFeatures(labels_all,predict_save,myloss_save,figure_my_save,dd_weights_fn)

    # ----------------------------------------------------------
