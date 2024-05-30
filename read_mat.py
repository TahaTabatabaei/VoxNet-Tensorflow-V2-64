import numpy as np
import scipy.io as sio
import open3d as o3d

def load_mat(datapath):
    # Load the .mat file
    return sio.loadmat(datapath)

def creat_pointcloud(mat_dictionary, grid_size):
    # Assuming the point cloud data is stored in a variable named 'pointCloud'
    points = mat_dictionary['xyz']
    color = mat_dictionary['color']

    # Create an Open3D point cloud object
    point_cloud = o3d.geometry.PointCloud()

    # Assign colors to the point cloud object
    # Open3D expects color values to be in the range [0, 1]
    color = color / 255.0
    point_cloud.colors = o3d.utility.Vector3dVector(color)

    # Assign points to the point cloud object
    point_cloud.points = o3d.utility.Vector3dVector(points)

    # fit to unit cube. 
    point_cloud.scale( (grid_size[0] - 1) / (np.max(point_cloud.get_max_bound() - point_cloud.get_min_bound())),
            center=point_cloud.get_center())
    
    return point_cloud

def voxelize(point_cloud):
    # Create a voxel grid from the point cloud with a voxel_size of 0.01
    voxel_grid=o3d.geometry.VoxelGrid.create_from_point_cloud(point_cloud,voxel_size=0.01)

    return voxel_grid

def visualize(voxel_grid):
    # Initialize a visualizer object
    vis = o3d.visualization.Visualizer()
    # Create a window, name it and scale it
    vis.create_window(window_name='dummy Visualize', width=1400, height=900)

    # Add the voxel grid to the visualizer
    vis.add_geometry(voxel_grid)

    # We run the visualizater
    vis.run()
    # Once the visualizer is closed destroy the window and clean up
    vis.destroy_window()

    return

def creat_occupancy_grid(voxel_grid, grid_size):
    # creat an occupany grid with grid size dims
    occupancy_grid = np.zeros(grid_size, dtype=bool)

    # if any voxel lies in an occupancy voxel, set it True
    for voxel in voxel_grid.get_voxels():
        # print(voxel.grid_index)
        index = voxel.grid_index.astype(int)
        if np.all(index >= 0) and np.all(index < grid_size):
            occupancy_grid[tuple(index)] = True
            # print(index)

    return occupancy_grid


if __name__ == "__main__":

    # grid_size = (64, 64, 64)
    grid_size = (32, 32, 32)

    print("loading data...")
    mat = load_mat("xyz5.mat")

    print("creating point cloud...")
    pointcloud = creat_pointcloud(mat, grid_size)

    print("generating voxel grid...")
    voxel_grid = voxelize(pointcloud)

    print("fitting occupancy grid...")
    ocp_grid = creat_occupancy_grid(voxel_grid, grid_size)

    savepath = f"grids/occupancy_grid_{grid_size[0]}.npy"
    np.save(file=savepath, arr=ocp_grid)
    print(f"saving occupancy grid in: {savepath}")

