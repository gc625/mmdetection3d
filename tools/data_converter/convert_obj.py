import trimesh

def to_ply(input_path, output_path, original_type):
    mesh = trimesh.load(input_path, file_type=original_type)  # read file
    mesh.export(output_path, file_type='ply')  # convert to ply'



to_ply('/home/gabriel/mmdetection3d/demo/kitti_000008/kitti_000008_pred.obj','/home/gabriel/mmdetection3d/demo/kitti_000008/kitti_000008_pred.ply','obj')