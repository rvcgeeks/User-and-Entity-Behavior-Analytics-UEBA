
import csv, json
import linecache,pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import roc_curve,auc
from keras.models import load_model,Model
from keras.layers import Input,LSTM,Dropout,Dense,BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import TensorBoard,ModelCheckpoint


def count_line(files_in):
    file_open=open(files_in,'r')
    count=0
    for line in file_open:
        count+=1
    file_open.close()
    return count   

def train(files_train,labels_train,model_path,files_test,label_test,predict_save):
   
    x_train=np.loadtxt(files_train,delimiter=',')
    y_train=np.loadtxt(labels_train,delimiter=',')
    print('num rows x_train ', len(x_train))
    # y_train=np.reshape(y_train,(-1,40,1))

    # --------- model structure -----------
    main_input=Input(shape=(3,),dtype='float32',name='MainInput')
    layer=Dense(10)(main_input)
    # layer=BatchNormalization()(layer)

    # layer=Dropout(0.5)(layer)
    layer=Dense(2,activation='softmax')(layer)
    # layer=LSTM(96,return_sequences=False,activation='tanh')(layer)
    
    output=layer

    # ------------------------
    model=Model(inputs=main_input,outputs=output)
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['binary_accuracy'])
    print(model.summary())

    tensorboard_path='./'
    tbCallback=TensorBoard(log_dir=tensorboard_path,histogram_freq=0,write_graph=True,write_grads=True,write_images=True,embeddings_freq=0,embeddings_layer_names=None,embeddings_metadata=None)
    checkpoint=ModelCheckpoint(model_path,monitor='loss',verbose=0,save_best_only='True')

    model.fit(x_train,y_train,batch_size=6,epochs=80,shuffle=True,callbacks=[tbCallback,checkpoint])
    
    # model.fit(x_train,y_train,batch_size=12,epochs=700,shuffle=True)
    # ---------------- add test -------
    x_test=np.loadtxt(files_test,delimiter=',')
    y_test=np.loadtxt(label_test,delimiter=',')
    print('num rows x_test ', len(x_test))
    y_pred=model.predict(x_test)
    np.savetxt(predict_save,y_pred,fmt='%f',delimiter=',')
    loss,acc=model.evaluate(x=x_test,y=y_test,verbose=0)
    print (acc)
    # -------------------------
    # model.save(save_path)
    # return y_train

def Calculatte(pred_file,label_file):
    # file_label=open(label_file,'r')
    Drn=0
    Dra=0
    with open(pred_file,'r') as csvfile:
        reader=csv.reader(csvfile)
        # i:0-n
        for i,rows in enumerate (reader):
            # print (i) #row is a list
            Pred=int(rows[0])
            line=linecache.getline(label_file,i+1)
            Label=int(line[0])
            if (Label==0 and Pred==0):
                Drn+=1
            if (Label==1 and Pred==1):
                Dra+=1
    print('Drn: ',Drn,' Dra: ',Dra)
    return Dra+Drn


def Count_nor_ano(label):
    Num_nor=0
    Num_ano=0
    with open(label,'r') as csvfile:
        reader=csv.reader(csvfile)
        # i:0-n
        for i,rows in enumerate (reader):
            if int(rows[0])==0:
                Num_nor+=1
            if int(rows[0])==1:
                Num_ano+=1
    print ('Num_normal: ',Num_nor,' Num_anomalous: ',Num_ano)
    return Num_ano+Num_nor

def figure_ponit(user):
    # normal
    x1=[]
    y1=[]
    z1=[]
    # anomalous
    x2=[]
    y2=[]
    z2=[]
    # To see Training Data
    #all_data_in='Data/Mix/'+'Mix_all_loss.csv'
    #all_label_in='Data/Mix/'+'Mix_all_label.csv'
    # To see predicted results
    all_data_in=working_folder+'test.csv'
    all_label_in=working_folder+'predict_label.csv'
    line_counts=count_line(all_label_in)
    for i in range(line_counts):
        line=linecache.getline(all_data_in,i+1)
        label=linecache.getline(all_label_in,i+1)
        line=line.strip()
        line=line.split(',')
        label=label.strip()
        label=label.split(',')
        # print(line)
        # print(label[0])
        # exit(0)
        if label[0]=='0':
            x1.append(float(line[0]))
            y1.append(float(line[1]))
            z1.append(float(line[2]))
            # print (x1,y1,z1)
            # exit(0)
        else:
            x2.append(float(line[0]))
            y2.append(float(line[1]))
            z2.append(float(line[2]))

    
    fig=plt.figure()
    #ax3d=Axes3D(fig)
    ax3d = fig.add_subplot(111,projection='3d')
    ax3d.scatter(x1,y1,z1,c='g',label=user + ' MLP Normal')
    ax3d.scatter(x2,y2,z2,c='r',label=user + ' MLP Anomalous')

    if 'Mix' != user:
        #sequence_dates=working_folder+'../sequence_dates.csv'
        sequence_dates=working_folder+'../sequence/sequence_dates_test.csv'
        #for i, x, y, z in zip(range(line_counts), x1+x2, y1+y2, z1+z2):
        for i, x, y, z in zip(range(line_counts), x2, y2, z2):
            line=linecache.getline(sequence_dates,i)
            line=line.strip()
            ax3d.text(x, y, z, line, color='black', fontsize=6)

    ax3d.set_zlabel('WDD of Role features', fontdict={'size': 13, 'color': 'black'})
    ax3d.set_ylabel('WDD of Action sequence', fontdict={'size': 13, 'color': 'black'})
    ax3d.set_xlabel('WDD of Action Features', fontdict={'size': 13, 'color': 'black'})

    ax3d.legend()
    #plt.show()
    fig.savefig(working_folder+'Point_MLP.jpg')
    pickle.dump(fig, open(working_folder+'Point_MLP_3d.fig.pickle', 'wb'))
    # plt.savefig('Point.jpg')




if __name__ == "__main__":

    with open('Data/config.json', 'r') as fh:
        CONFIG = json.load(fh)
    user_sets=['Mix']+list(CONFIG['monitor'].keys())
    for user in user_sets:
        print('Training for User', user)
        if 'Mix' == user:
            working_folder='Data/'+user+'/'
        else:
            working_folder='Data/'+user+'/Mix/'
        files_train=working_folder+'train.csv'
        labels_train=working_folder+'label_train.csv'
        model_path=working_folder+'MLP.h5'
        files_test=working_folder+'test.csv'
        label_test=working_folder+'label_test.csv'
        predict_save=working_folder+'predict.csv'
        predict_label=working_folder+'predict_label.csv'
        # ------ 训练MLP
        train(files_train,labels_train,model_path,files_test,label_test,predict_save)
        # -------------------------
        # ---------------对预测数据进行标签化 (add label for data)---------------
        predicts=np.loadtxt(predict_save,delimiter=',')
        pred=np.where(predicts>0.5,1,0)
        np.savetxt(predict_label,pred,delimiter=',',fmt='%d')
        # -------------------- end -----------------
        # -------------- 对结果进行统计 ( metrics)：DRA ACC ...---------
        all_label=Count_nor_ano(label_test)
        pred=Calculatte(predict_label,label_test)
        print (pred/all_label)
        # -------------------- end --------------------
        # -------- ROC curve --------------------
        file_open=open(label_test,'r')
        file_open_two=open(predict_save,'r')
        y_true=[]
        y_score=[]
        for line in file_open:
            if line[0]=='0':
                y_true.append(1)
            else:
                y_true.append(0)
        file_open.close()

        for line in file_open_two:
            line=line.strip()
            line=line.split(',')
            y_score.append(float(line[1]))
        file_open_two.close()
        # print (y_score)

        fpr,tpr,threshold=roc_curve(y_true,y_score)
        Auc=auc(fpr,tpr)
        print('FPR: ',fpr)
        print('TPR: ',tpr)
        # --------------base line
        baseline_fpr=[0,0.03703704,0.03803704,0.04703704,0.06407407,0.07407407,0.07407407,0.11111111,0.11311111,0.14814815,0.15814815,0.35564815,0.46444444,0.46444444,1]
        baseline_tpr=[0,0.16184971,0.26184971,0.26184971,0.63815029,0.83815029,0.83815029,0.88265896,0.88265896,0.88265896,0.98643931,0.99321965,0.99321965,1,1]
        Baseline_Auc=auc(baseline_fpr,baseline_tpr)

        # print(Auc)
        plt.figure()
        plt.plot(fpr,tpr,color='red',label='MBS AUC = %0.3f)'% Auc)
        plt.plot(baseline_fpr,baseline_tpr,color='blue',label='Baseline AUC = %0.3f)'% Baseline_Auc)
        plt.plot([0, 1], [0, 1], color='navy',linestyle='--')
        plt.xlim([0.0,1.0])
        plt.ylim([0.0,1.05])
        plt.xlabel('False Positive Rate',fontsize=14)
        plt.ylabel('True Positive Rate',fontsize=14)
        plt.title('Receiver operating characteristic',fontsize=14)
        # 显示图示
        plt.legend(fontsize=12)
        # plt.show()
        plt.savefig(working_folder+'ROC.jpg')
        # -------------------------- end -----------------------------------
        # ------------- 三维散点图 (scatter diagram) ----------------
        figure_ponit(user)
    
