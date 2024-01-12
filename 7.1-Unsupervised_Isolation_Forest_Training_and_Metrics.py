
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
import linecache
import matplotlib.pyplot as plt
import csv
from sklearn.metrics import roc_curve,auc
from mpl_toolkits.mplot3d import Axes3D
import pickle


def count_line(files_in):
    file_open=open(files_in,'r')
    count=0
    for line in file_open:
        count+=1
    file_open.close()
    return count

def train(files_train,predict_label):

    x_train=np.loadtxt(files_train,delimiter=',')
    print('num rows x_train ', len(x_train))
    # y_train=np.reshape(y_train,(-1,40,1))

    clf = IsolationForest(contamination=0.15, random_state=42)
    y_pred = clf.fit_predict(x_train)

    # Calculate silhouette score
    silhouette_avg = silhouette_score(x_train, y_pred)
    print(f"Silhouette Score: {silhouette_avg}")

    # Map labels to 1 and 0 (1 for outliers, 0 for inliers)
    y_pred[y_pred == 1] = 0  # Inliers
    y_pred[y_pred == -1] = 1  # Outliers

    np.savetxt(predict_label,y_pred,fmt='%d',delimiter=',')

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
    #all_data_in=working_folder+'Mix_all_loss.csv'
    #all_label_in=working_folder+'predict_label_IF.csv'
    all_data_in=working_folder+'test.csv'
    all_label_in=working_folder+'predict_label_IF.csv'
    line_counts=count_line(all_label_in)
    #rate=0.7 # TRAIN TEST SPLIT PARTITION
    #start=int(rate*line_counts)
    start = 0
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
    ax3d.scatter(x1,y1,z1,c='g',label=user + ' IF Normal')
    ax3d.scatter(x2,y2,z2,c='r',label=user + ' IF Anomalous')

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
    fig.savefig(working_folder+'Point_IF.jpg')
    pickle.dump(fig, open(working_folder+'Point_IF_3d.fig.pickle', 'wb'))
    # plt.savefig('Point.jpg')




if __name__ == "__main__":

    user_sets=['Mix','EDB0714','TNM0961','HXL0968']
    for user in user_sets:
        print('Training for User', user)
        if 'Mix' == user:
            working_folder='Data/'+user+'/'
        else:
            working_folder='Data/'+user+'/Mix/'
        #files_train=working_folder+'Mix_all_loss.csv'
        #predict_label=working_folder+'predict_label_IF.csv'
        files_train=working_folder+'test.csv'
        predict_label=working_folder+'predict_label_IF.csv'
        # ------ Isolation Forest
        train(files_train,predict_label)
        # -------------------- end -----------------
        # -------------- 对结果进行统计 ( metrics)：Anomaly Counts ...---------
        all_label=Count_nor_ano(predict_label)
        # -------------------------- end -----------------------------------
        # ------------- 三维散点图 (scatter diagram) ----------------
        figure_ponit(user)




