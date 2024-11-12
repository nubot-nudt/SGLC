
import numpy as np
import argparse


def compute_relative_translation_error(gt_translation: np.ndarray, est_translation: np.ndarray):
    r"""Compute the isotropic Relative Translation Error.

    RTE = \lVert t - \bar{t} \rVert_2

    Args:
        gt_translation (array): ground truth translation vector (3,)
        est_translation (array): estimated translation vector (3,)

    Returns:
        rte (float): relative translation error.
    """
    return np.linalg.norm(gt_translation - est_translation)

def compute_relative_rotation_error(gt_rotation: np.ndarray, est_rotation: np.ndarray):
    r"""Compute the isotropic Relative Rotation Error.

    RRE = acos((trace(R^T \cdot \bar{R}) - 1) / 2)

    Args:
        gt_rotation (array): ground truth rotation matrix (3, 3)
        est_rotation (array): estimated rotation matrix (3, 3)

    Returns:
        rre (float): relative rotation error.
    """
    x = 0.5 * (np.trace(np.matmul(est_rotation.T, gt_rotation)) - 1.0)
    
    x = np.clip(x, -1.0, 1.0)
    x = np.arccos(x)
    rre = 180.0 * x / np.pi

    return rre


def main(dataset,gt_pair_poses_file,predict_pair_poses_file):

    gt_pair_poses = np.loadtxt(gt_pair_poses_file)
    predict_pair_poses = np.loadtxt(predict_pair_poses_file)

    success = 0
    success_rte = 0.0
    success_rre = 0.0

    error_rre_list = []
    error_rte_list = []
    for idx in range(0,len(gt_pair_poses)):
        gt_trans = gt_pair_poses[idx,2:].reshape(3,4)
        predict_trans = predict_pair_poses[idx,2:].reshape(3,4)

        if dataset == 'kitti360':
            predict_trans_matrix = np.vstack((predict_trans,[0,0,0,1]))
            predict_trans_matrix = np.linalg.inv(predict_trans_matrix)
            predict_trans = predict_trans_matrix[:3,:]
            
        error_rte = compute_relative_translation_error(gt_trans[:,3],predict_trans[:,3])
        error_rre = compute_relative_rotation_error(gt_trans[:,:3],predict_trans[:,:3])
        
        error_rre_list.append(error_rre)
        error_rte_list.append(error_rte)
        if error_rte<2.0 and error_rre< 5:
            success += 1
            success_rte += error_rte
            success_rre += error_rre

    success_rate = success/len(gt_pair_poses)
    mean_rte = success_rte/success
    mean_rre = success_rre/success

    print("\033[1;36m-----------results-----------\033[0m")
    print(f"\033[1;31m RR:\t{success_rate*100:.3f}\033[0m")
    print(f"\033[1;31m RTE:\t{mean_rte:.3f}\033[0m")
    print(f"\033[1;31m RRE:\t{mean_rre:.3f}\033[0m")


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--dataset', type=str, default='kitti',
                        help='kitti or kitti360 ')
    parser.add_argument('--gt_poses', type=str, default='../out/kitti/',
                        help='the ground truth poses .txt file ')
    parser.add_argument('--pred_file', type=str, default='../out/kitti/',
                        help='the estimated poses .txt file ')

    args = parser.parse_args()
    main(args.dataset,args.gt_poses,args.pred_file)