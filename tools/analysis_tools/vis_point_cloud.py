import open3d as o3d
import pickle 
import torch
from scipy.spatial.transform import Rotation as R
import numpy as np
from mmdet3d.ops.detr3d_utils.pc_util import scale_points, shift_scale_points
from vod.frame.transformations import transform_pcl
from vod.frame import FrameDataLoader, FrameLabels, FrameTransformMatrix
from vod.configuration import KittiLocations
from glob import glob
from pathlib import Path as P 
def main():

    

    ret_dict = pickle.load(open('fixed_ret_dict2.pkl','rb'))
    points = pickle.load(open('fixed_point_cloud2.pkl','rb'))
    og_bbx = pickle.load(open('original_gt2.pkl','rb'))
    print(ret_dict.keys())

    

    first_points = points[:,:3]
    dim_min = first_points.min(axis=0)[0]
    dim_max = first_points.max(axis=0)[0]
    mult_factor = dim_max - dim_min
    mult_factor = 1/mult_factor

    shifted = shift_scale_points(
        first_points[None,::],
        torch.stack(
            [
                dim_min[None,...],
                dim_max[None,...]
            ]
        )

    )








    lidar_pcd = o3d.geometry.PointCloud()
    # lidar_pcd.points = o3d.utility.Vector3dVector(first_points.cpu().numpy())
    lidar_pcd.points = o3d.utility.Vector3dVector(shifted.squeeze(0).cpu().numpy())
    geometries = [lidar_pcd]

    ### getting bbx:

    is_gt = int(np.sum(ret_dict['gt_box_present']))
    box_centers = ret_dict['gt_box_centers_normalized'][:is_gt]
    box_size = ret_dict['gt_box_sizes_normalized'][:is_gt]
    angles = ret_dict['gt_box_angles'][:is_gt]

    # gt_bbx = torch.hstack((box_centers,box_size,angles.unsqueeze(1)))

    for i in range(len(box_centers)):
        # offset = -(box['h']/2) 
        ext = box_size[i]
        xyz = box_centers[i]
        offset = ext[2]/2
        # xyz[2] += offset
        rot = (np.pi/2) -angles[i] 
        rot = angles[i]


        # rot = -(box['rotation']+ np.pi / 2) 
        angle = np.array([0, 0, rot]) 
        rot_matrix = R.from_euler('XYZ', angle).as_matrix()
        obbx = o3d.geometry.OrientedBoundingBox(xyz, rot_matrix, ext.T)
        geometries += [obbx]    
    
    


    vis = o3d.visualization.Visualizer()
    vis.create_window() 

    for g in geometries:
        vis.add_geometry(g)
    vis.run()  # user changes the view and press "q" to terminate

    return




def get_kitti_locations(vod_data_path):
    kitti_locations = KittiLocations(root_dir=vod_data_path,
                            output_dir="output/",
                            frame_set_path="",
                            pred_dir="",
                            )
    return kitti_locations

def vod_to_o3d(vod_bbx,vod_calib):
    # modality = 'radar' if is_radar else 'lidar'
    # split = 'testing' if is_test else 'training'    
    
    COLOR_PALETTE = {
            'Cyclist': (1, 0.0, 0.0),
            'Pedestrian': (0.0, 1, 0.0),
            'Car': (0.0, 0.3, 1.0),
            'Others': (0.75, 0.75, 0.75)
        }
    box_list = []
    for box in vod_bbx:
        if box['label_class'] in ['Cyclist','Pedestrian','Car']:
            # Conver to lidar_frame 
            # NOTE: O3d is stupid and plots the center of the box differently,
            offset = -(box['h']/2) 
            old_xyz = np.array([[box['x'],box['y']+offset,box['z']]])
            xyz = transform_pcl(old_xyz,vod_calib.t_lidar_camera)[0,:3] #convert frame
            extent = np.array([[box['l'],box['w'],box['h']]])
            
            # ROTATION MATRIX
            rot = -(box['rotation']+ np.pi / 2) 
            angle = np.array([0, 0, rot])
            rot_matrix = R.from_euler('XYZ', angle).as_matrix()
            
            # CREATE O3D OBJECT
            obbx = o3d.geometry.OrientedBoundingBox(xyz, rot_matrix, extent.T)
            obbx.color = COLOR_PALETTE.get(box['label_class'],COLOR_PALETTE['Others']) # COLOR
            
            box_list += [obbx]
    return box_list





def vis_val():

    kitti_locations = get_kitti_locations('/home/gabriel/mmdetection3d/data/vod')
    # val_frm = '/home/gabriel/mmdetection3d/data/vod/lidar/ImageSets/val.txt'
    # frame_ids  = [P(f).stem for f in glob(str(dt_path)+"/*")]
    frame_data = FrameDataLoader(kitti_locations,
                                "000002","")
    vod_calib = FrameTransformMatrix(frame_data)

    vod_labels = FrameLabels(frame_data.get_labels()).labels_dict
    bbxes = vod_to_o3d(vod_labels,vod_calib)

    points = []

    # ret0 = pickle.load(open('/home/gabriel/mmdetection3d/pre_enc_output.pkl','rb'))
    ret1 = pickle.load(open('/home/gabriel/mmdetection3d/layer0_output.pkl','rb'))
    ret2 = pickle.load(open('/home/gabriel/mmdetection3d/layer1_output.pkl','rb'))
    ret3 = pickle.load(open('/home/gabriel/mmdetection3d/layer2_output.pkl','rb'))
    ret4 = pickle.load(open('/home/gabriel/mmdetection3d/vote_agg_ret.pkl','rb'))

    stuff = [ret1,ret2,ret3,ret4]

    geometries = []
    for ret in stuff:
        points = ret[0].squeeze(0).cpu().numpy()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        colors = np.zeros_like(points)
        colors[:,0] = 1
        geometries += [pcd]



    # enc_xyz = pickle.load(open('00000_enc_xyz.pkl','rb'))
    # query_xyz = pickle.load(open('00000_query_xyz.pkl','rb'))
    # point_cloud = pickle.load(open('00000_point_clouds.pkl','rb'))


    # enc_pcd = o3d.geometry.PointCloud()
    # query_pcd = o3d.geometry.PointCloud()
    # point_pcd = o3d.geometry.PointCloud()
    
    # enc_colors = np.zeros_like(enc_xyz.squeeze(0).cpu().numpy())
    # query_colors = np.zeros_like(query_xyz.squeeze(0).cpu().numpy())
    # point_colors = np.zeros_like(point_cloud.squeeze(0)[:,:3].cpu().numpy())
    
    # enc_colors[:,0] = 1
    # query_colors[:,1] = 1
    # point_colors[:,2] = 1

    
    # enc_pcd.points = o3d.utility.Vector3dVector(enc_xyz.squeeze(0).cpu().numpy())
    # query_pcd.points = o3d.utility.Vector3dVector(query_xyz.squeeze(0).cpu().numpy())
    # point_pcd.points = o3d.utility.Vector3dVector(point_cloud.squeeze(0)[:,:3].cpu().numpy())

    # enc_pcd.colors = o3d.utility.Vector3dVector(enc_colors)
    # query_pcd.colors = o3d.utility.Vector3dVector(query_colors)
    # point_pcd.colors = o3d.utility.Vector3dVector(point_colors)

    # vis = o3d.visualization.Visualizer()
    # vis.create_window() 

    # geometries = [enc_pcd,query_pcd,point_pcd]

    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = 'defaultUnlit'
    mat.point_size = 6.0
    mat2 = o3d.visualization.rendering.MaterialRecord()
    mat2.shader = 'defaultUnlit'
    mat2.point_size = 4.0
    mat3 = o3d.visualization.rendering.MaterialRecord()
    mat3.shader = 'defaultUnlit'
    mat3.point_size = 2.0







    # o3d.visualization.draw()

    o3d.visualization.draw([{'name': 'enc_pcd', 'geometry': enc_pcd, 'material': mat},
    {'name': 'query_pcd', 'geometry': query_pcd, 'material': mat2},
    {'name': 'point_pcd', 'geometry': point_pcd, 'material': mat3}]+bbxes, show_skybox=False)
    # o3d.visualization.draw([{'name': 'query_pcd', 'geometry': query_pcd, 'material': mat2}], show_skybox=False)
    # o3d.visualization.draw([{'name': 'point_pcd', 'geometry': point_pcd, 'material': mat3}], show_skybox=False)
    # for g in geometries:
    #     vis.add_geometry(g)
    # vis.run() 

if __name__ == "__main__":
    vis_val()
    # main()