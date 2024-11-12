import sys
import json
import os.path as osp
import time
import logging
import argparse
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt

from sklearn import metrics



def compute_AP(precision, recall):
    ap = 0.
    for i in range(1, len(precision)):
        ap += (recall[i] - recall[i-1])*precision[i]
    return ap

def compute_F1(precision, recall):
    precision = np.asarray(precision)
    recall = np.asarray(recall)
    F1 = 2*precision*recall/(precision+recall)
    idx = F1.argmax()
    F1 = F1.max()
    
    return F1, idx

def cal_topN(pair_dist, ground_truth_file_name, topn):
    precisions = []
    recalls = []



    des_dists = pair_dist
    try:
        ground_truth = np.load(ground_truth_file_name, allow_pickle='True')['arr_0']
    except:
        ground_truth = np.load(ground_truth_file_name, allow_pickle='True')['data']


    all_have_gt = 0
    tps = 0


    for idx in range(0,len(ground_truth)-1):
        gt_idxes = np.array(ground_truth[int(idx)])

        if not gt_idxes.any():
            continue

        all_have_gt += 1

        matched = des_dists[des_dists[:,0]==int(idx)]
        dis = matched[:,2]
        select_indice = np.argsort(dis)
        matched_sort = matched[select_indice]
        for t in range(topn):
            if matched_sort[t, 1] in gt_idxes:
                tps += 1
                break

    recall_topN = tps/all_have_gt

    return recall_topN



def cal_pr_curve(pair_dist, ground_truth_file_name, thre_range=[0,1], interval=0.01):
    precisions = []
    recalls = []
    f1max=0

    try:
        ground_truth = np.load(ground_truth_file_name, allow_pickle='True')['arr_0']
    except:
        ground_truth = np.load(ground_truth_file_name, allow_pickle='True')['data']


    """Changing the threshold will lead to different test results"""
    for thres in np.arange(thre_range[0], thre_range[1], interval):
        tps = 0
        fps = 0
        tns = 0
        fns = 0
        pre_F1=0
        cout=0
        """Update the start frame index"""
        for idx in range(150,len(ground_truth)-1):
            gt_idxes = np.array(ground_truth[int(idx)])
            reject_flag = False
            
            pair_dist_part = pair_dist[pair_dist[:,0]==int(idx),2]

            matched = pair_dist[pair_dist[:,0]==int(idx)]
            dis = matched[:,2]
            select_indice = np.argsort(dis)
            matched_sort = matched[select_indice]

            if matched_sort[:,2][0]>thres:
                reject_flag = True
            if reject_flag:
                if not gt_idxes.any():
                    tns += 1
                else:
                    fns += 1
            else:
                if matched_sort[:,1][0] in gt_idxes:
                    tps += 1
                else:
                    fps += 1

        if fps == 0:
            precision = 1
        else:
            precision = float(tps) / (float(tps) + float(fps))
        if fns == 0:
            recall = 1
        else:
            recall = float(tps) / (float(tps) + float(fns))

        precisions.append(precision)
        recalls.append(recall)

        F1 = 2*precision*recall/(precision+recall+1e-12)


        message = "thresh: %.3f "%thres + "   precision: %.3f "%precision + "   recall: %.3f "%recall + "   F1: %.3f "%F1
        print(message)
        if recall == 1:
            break


    return precisions, recalls

"""Ploting and saving AUC."""
def plotPRC(precisions, recalls, F1, recall_list, print_2file=False, snapshot=None, dataset='kitti'):
    # initial plot
    plt.clf()

    save_name = '../out/PRC_%s.png'%(dataset)

    recalls, precisions = (list(t) for t in zip(*sorted(zip(recalls, precisions), reverse=True)))
    auc = metrics.auc(recalls, precisions) * 100

    plt.plot(recalls, precisions, linewidth=1.0)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1])
    plt.xlim([0.0, 1])
    plt.title('auc=%.3f '%auc + 'F1=%.3f '%F1 + 'top1=%.3f '%recall_list[0] + 'top%%1=%.3f '%recall_list[1])

    if print_2file:
        plt.savefig(save_name)

    return auc




def eval(cfg):

    pred_file = cfg.pred_file
    gt_file = cfg.gt_file


    if  osp.exists(pred_file):
        pair_dist = np.loadtxt(pred_file)
        pair_dist = np.asarray(pair_dist, dtype='float32')
        pair_dist = pair_dist.reshape((len(pair_dist), 3))
        print('Load descrptor distances predictions with shape of' ,pair_dist.shape)
    thre_range=[0,2.1]
    interval=0.01
    print("thre_range:",thre_range)
    print("interval:",interval)
    precision_ours_fp, recall_ours_fp = cal_pr_curve(pair_dist, gt_file, thre_range, interval)
    ap_ours_fp = compute_AP(precision_ours_fp, recall_ours_fp)
    F1, idx = compute_F1(precision_ours_fp, recall_ours_fp)
    message = "\033[1;31m F1 max\t\t: %.3f \033[0m"%F1

    print("\033[1;36m--------------------Results on "+cfg.dataset+ "-------------------\033[0m")
    print(message)
    # recall@1%
    if cfg.dataset=='kitti':
        x=45   
    elif cfg.dataset=='ford':
        x=38
    elif cfg.dataset=='apollo':
        x=20


    topn = np.array([1,x]) # for KITTI 00 top1%
    recall_list = []
    message = ''
    for i in range(0, topn.shape[0]):
        rec = cal_topN(pair_dist, gt_file, topn[i])
        recall_list.append(rec)
        message = "\033[1;31m top"+str(topn[i])+" recall\t: %.3f \033[0m"%rec
        print(message)
    

    auc = plotPRC(precision_ours_fp, recall_ours_fp, F1, recall_list, True, cfg.dataset)
    message = "\033[1;31m AUC\t\t: %.3f \033[0m"%auc
    print(message)
    message = "\033[1;31m Average Precision: %.3f \033[0m"%ap_ours_fp
    print(message)




def main():


    parser = argparse.ArgumentParser(description='arg parser')

    parser.add_argument('--dataset', type=str, default='kitti',
                        help='kitti or kitti360 ')
    parser.add_argument('--gt_file', type=str, default='../out/kitti/',
                        help='the ground truth .txt file ')
    parser.add_argument('--pred_file', type=str, default='../out/kitti/',
                        help='the prebdicted  .txt file ')

    args = parser.parse_args()

    eval(args)


if __name__ == '__main__':
    main()
