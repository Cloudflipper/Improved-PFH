import query_improved as query_pfh
import utility as utils
import numpy as np
from timer import Timer
import open3d as o3d
import filter as flt
import random


def rotation_matrix_3d(theta_x, theta_y, theta_z):
    """Generate a 3D rotation matrix around the x, y, and z axes."""
    # Rotation around the x-axis
    R_x = np.array(
        [
            [1, 0, 0],
            [0, np.cos(theta_x), -np.sin(theta_x)],
            [0, np.sin(theta_x), np.cos(theta_x)],
        ]
    )

    # Rotation around the y-axis
    R_y = np.array(
        [
            [np.cos(theta_y), 0, np.sin(theta_y)],
            [0, 1, 0],
            [-np.sin(theta_y), 0, np.cos(theta_y)],
        ]
    )

    # Rotation around the z-axis
    R_z = np.array(
        [
            [np.cos(theta_z), -np.sin(theta_z), 0],
            [np.sin(theta_z), np.cos(theta_z), 0],
            [0, 0, 1],
        ]
    )

    # Combined rotation matrix
    R = np.dot(R_z, np.dot(R_y, R_x))

    # Convert to 4x4 homogeneous transformation matrix
    R_homogeneous = np.vstack(
        [
            np.hstack([R, np.zeros((3, 1))]),  # Add a column of zeros for translation
            np.array([0, 0, 0, 1]),  # Add a row for homogeneous coordinates
        ]
    )

    return R_homogeneous


def translation_matrix(a, b):
    """Generate a 2D translation matrix"""
    return np.array([[1, 0, 0, a], [0, 1, 0, b], [0, 0, 1, 0], [0, 0, 0, 1]])


def combined_transformation_matrix(R, a, b):
    """Generate a combined transformation matrix for rotation and translation"""
    T = translation_matrix(a, b)
    return np.dot(T, R)


def apply_transformation_to_point_cloud(point_cloud, transformation_matrix):
    """Apply a homogeneous transformation matrix to a Nx3 point cloud matrix"""
    # Ensure the transformation matrix is 4x4
    if transformation_matrix.shape == (3, 3):
        # Create a 4x4 transformation matrix by adding a translation component
        transformation_matrix = np.vstack(
            [
                np.hstack([transformation_matrix, np.zeros((3, 1))]),
                np.array([0, 0, 0, 1]),
            ]
        )

    # Convert the Nx3 point cloud to a Nx4 homogeneous point cloud
    homogeneous_point_cloud = np.hstack(
        (point_cloud, np.ones((point_cloud.shape[0], 1)))
    )

    # Apply the transformation matrix
    transformed_point_cloud_homogeneous = np.dot(
        homogeneous_point_cloud, transformation_matrix.T
    )

    # Convert back to a Nx3 point cloud
    transformed_point_cloud = transformed_point_cloud_homogeneous[:, :3]

    return transformed_point_cloud


def normalize_point_cloud_uniformly(point_cloud):
    """Normalize a Nx3 point cloud matrix so that all values are between 0 and 1, scaled uniformly across all dimensions"""
    # Find the overall minimum and maximum values in the point cloud
    min_value = np.min(point_cloud)
    max_value = np.max(point_cloud)

    # Calculate the range
    range_value = max_value - min_value

    # Avoid division by zero by adding a small epsilon to the range
    epsilon = 1e-8
    range_value = np.where(range_value == 0, epsilon, range_value)

    # Normalize the point cloud to be between 0 and 1
    normalized_point_cloud = (point_cloud - min_value) / range_value

    return normalized_point_cloud


def read_pcd_to_array(pcd_file_path):
    # read pcd file
    pcd = o3d.io.read_point_cloud(pcd_file_path)

    # convert to numpy array
    points = np.asarray(pcd.points)

    return points


def random_sample_half_rows(array):
    """
    Randomly samples half the rows from an N x 3 array.
    :param array: Input array (N x 3)
    :return: Sampled array with half the rows
    """
    # Get the total number of rows
    total_rows = array.shape[0]

    # Determine the sample size (half of the rows)
    sample_size = total_rows // 4

    # Randomly select row indices without replacement
    selected_indices = np.random.choice(total_rows, sample_size, replace=False)

    # Extract the selected rows
    sampled_array = array[selected_indices, :]
    return sampled_array


def main():
    # load pc data
    # pcd_path = "table_scene_mug_stereo_textured.pcd"  # source
    random.seed(1145)
    print("Estimated runtime of PFH-ICP for room_scan1.pcd: <1140s (19 minutes)\n")
    print(
        "When the program is done, try rotating the point cloud to see the alignment as shown in the report."
    )
    pcd_path = "room_scan1.pcd"
    # pcd_path = "ism_test_cat.pcd"
    # source = utils.read_pcd_to_numpy(pcd_path)  # convert to numpy array (N,3)
    source = read_pcd_to_array(pcd_path)
    source = random_sample_half_rows(source)
    source = normalize_point_cloud_uniformly(source)
    source += np.random.normal(loc=0, scale=0.001, size=source.shape)

    # flt.expolre_histogram(source, 10, threshold=1.0)
    theta_x = np.pi / 3
    theta_y = 0
    theta_z = np.pi / 4

    R = rotation_matrix_3d(theta_x, theta_y, theta_z)
    a = 0.2  # Translate right
    b = 0.3  # Translate up
    M = combined_transformation_matrix(R, a, b)
    # Apply transformation to source to get target
    target = apply_transformation_to_point_cloud(source, M)
    target += np.random.normal(loc=0, scale=0.001, size=target.shape)
    # target = normalize_point_cloud(target)

    work_timer = Timer()
    work_timer.start()
    aligned_cloud = query_pfh.icp_with_fpfh_within_query(
        source,
        target,
        radius=0.02,
        num_bins=10,
        max_iters=1000,
        patience=5,
        timer=work_timer,
        filter_function="arctan",
        retainment_ratio="sparse",
        clip_mode="light",
    )
    # work_timer.clip()
    # icp.ICP(aligned_cloud, target, 0.042)
    work_timer.stop()
    print(f"Elapsed time: {work_timer.get_elapsed_time():.4f} seconds")

    utils.show_pic(source, target, aligned_cloud)
    # query_pfh.evaluate_threshold(source, 10, threshold=1.0)


if __name__ == "__main__":
    main()
