import  numpy as np
from sklearn import  metrics
from matplotlib import pyplot as plt
import os
def eval(file,out_path):
    print(os.path.join(out_path,file+".txt"))
    db=np.genfromtxt(os.path.join(out_path,file+".txt"))
    db=np.nan_to_num(db)
    precision, recall, pr_thresholds = metrics.precision_recall_curve(db[:,1], db[:,0])
    plt.plot(recall, precision, color='darkorange',lw=2, label='P-R curve')
    plt.axis([0,1,0,1])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve Seq')
    plt.savefig(os.path.join(out_path,'pr_curve_'+file+'.png'))
    F1_score = 2 * precision * recall / (precision + recall)
    F1_score = np.nan_to_num(F1_score)
    F1_max_score = np.max(F1_score)
    recall_p100_list = recall[precision==1.0]
    recall_p100 = np.max(recall_p100_list)
    P_R0 = precision[np.argmin(recall)]

    print("P_R0:",P_R0)
    print("recall_p100:",recall_p100)
    print("F1:",F1_max_score)
    print("EP:",(P_R0+recall_p100)/2)
    plt.show()
    plt.close()

if __name__=="__main__":

    sequ=['08']
    for v in sequ:
       eval(v,'../out/kitti/')
