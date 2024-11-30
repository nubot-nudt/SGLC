
import numpy as np
import open3d as o3d


def main():

    cloud_file_1 = "../demo_data/positive/000052.bin" 
    cloud_file_2 = "../demo_data/positive/004501.bin"  
    trans_file = "../out/demo_pair_pose.txt"


    cloud1 = np.fromfile(cloud_file_1,dtype=np.float32).reshape((-1,4))[:,:3]
    cloud2 = np.fromfile(cloud_file_2,dtype=np.float32).reshape((-1,4))[:,:3]
    print("cloud1 shape:",cloud1.shape)
    print("cloud2 shape:",cloud2.shape)

    color1 = np.zeros((cloud1.shape[0],3))
    color1[:,0] = 1
    color2 = np.zeros((cloud2.shape[0],3))
    color2[:,1] = 1

    raw_cloud = np.concatenate([cloud1,cloud2])

    transform = np.loadtxt(trans_file).reshape(3,4)
    transform = np.vstack([transform,[0,0,0,1]])
    transform=  np.linalg.inv(transform)

    
    NP = cloud2.shape[0]
    xyz1 = np.hstack([cloud2, np.ones((NP, 1))]).T
    cloud2 = (transform @ xyz1).T[:, :3]

    cloud_reg = np.concatenate([cloud1,cloud2])
    color_total = np.concatenate([color1,color2])

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Registration")

    vis_raw = o3d.visualization.Visualizer()
    vis_raw.create_window(window_name="Raw")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud_reg[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(color_total)
    vis.add_geometry(pcd)

    pcd_raw = o3d.geometry.PointCloud()
    pcd_raw.points = o3d.utility.Vector3dVector(raw_cloud[:, :3])
    pcd_raw.colors = o3d.utility.Vector3dVector(color_total)
    vis_raw.add_geometry(pcd_raw)


    while True:
        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()

        vis_raw.update_geometry(pcd_raw)
        vis_raw.poll_events()
        vis_raw.update_renderer()

if __name__ == "__main__":
    main()