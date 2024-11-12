import  numpy as np
from sklearn import  metrics
from matplotlib import pyplot as plt
import os
def eval(file,out_path):
    print(os.path.join(out_path,file+".txt"))
    db=np.genfromtxt(os.path.join(out_path,file+".txt"))
    db=np.nan_to_num(db)
    print(" db 0:",db[:,0])
    print(" db 1:",db[:,1])
    mask_tp = np.logical_and(db[:,0]>0.5 , db[:,1]==1)
    mask_tn = np.logical_and(db[:,0]<0.5 , db[:,1]==0)
    mask_fp = np.logical_and(db[:,0]>0.5 , db[:,1]==0)
    mask_fn = np.logical_and(db[:,0]<0.5 , db[:,1]==1)
    indexes_tp = np.where(mask_tp)[0]
    indexes_tn = np.where(mask_tn)[0]
    indexes_fp = np.where(mask_fp)[0]
    indexes_fn = np.where(mask_fn)[0]
    db_tp = db[mask_tp]
    db_tn = db[mask_tn]
    db_fp = db[mask_fp]
    db_fn = db[mask_fn]
    print("total:",len(db))
    print("tp:",len(db_tp))
    # print("index fp:",indexes_fp)
    print("tn:",len(db_tn))
    print("fp:",len(db_fp))
    print("index fp:",indexes_fp)
    print("fn:",len(db_fn))
    # print("index fn:",indexes_fn)


if __name__=="__main__":
    #sequ=['0000','0002','0005','0006','0004','0009']
    # sequ=['00','02','05','06','07','08']
    sequ=['0009_4']
    for v in sequ:
       eval(v,'../out/kitti360')
