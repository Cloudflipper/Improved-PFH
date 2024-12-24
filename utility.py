import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from numpy.linalg import svd


def read_pcd_to_numpy(file_path):
    """
    Reads a PCD file and converts the point cloud data into a N*3 NumPy array.

    Parameters:
        file_path (str): Path to the PCD file.

    Returns:
        np.ndarray: N*3 point cloud data.
    """
    with open(file_path, "rb") as f:
        # Parse the PCD file header
        header = []
        while True:
            line = f.readline().decode("utf-8").strip()
            header.append(line)
            if line.startswith("DATA"):
                data_format = line.split()[1]
                break

        # Extract field information
        fields = []
        sizes = []
        counts = []
        points = 0
        for line in header:
            if line.startswith("FIELDS"):
                fields = line.split()[1:]
            elif line.startswith("SIZE"):
                sizes = list(map(int, line.split()[1:]))
            elif line.startswith("COUNT"):
                counts = list(map(int, line.split()[1:]))
            elif line.startswith("POINTS"):
                points = int(line.split()[1])

        # Ensure the file contains at least x, y, z
        if len(fields) < 3 or fields[:3] != ["x", "y", "z"]:
            raise ValueError("PCD file must contain at least the x, y, z fields.")

        # Read the data section
        if data_format == "ascii":
            data = np.loadtxt(f, dtype=np.float32)
        elif data_format == "binary":
            # Calculate the byte size of each point
            point_size = sum(sizes[i] * counts[i] for i in range(len(fields)))
            data = np.frombuffer(f.read(points * point_size), dtype=np.float32)
            data = data.reshape((-1, len(fields)))
        else:
            raise ValueError(f"Unsupported PCD data format: {data_format}")

    # Extract the x, y, z columns
    return data[:, :3]


def compute_normals(points, radius):
    """
    Compute the normal vectors of a point cloud.
    :param points: Point cloud coordinates, shape as (N, 3)
    :param radius: Search neighborhood radius
    :return: Normal vector array, shape as (N, 3)
    This part is of complexity O(M logM)
    """
    normals = []
    kdtree = KDTree(points)

    for p in points:
        # Find the indices of points in the neighborhood
        indices = kdtree.query_ball_point(p, radius)
        if len(indices) < 3:
            normals.append(np.array([0.0, 0.0, 0.0]))
            continue
        # Calculate the covariance matrix of the neighborhood points
        neighbors = points[indices]
        centroid = np.mean(neighbors, axis=0)
        cov_matrix = np.cov((neighbors - centroid).T)
        eigvals, eigvecs = np.linalg.eigh(cov_matrix)
        normal = eigvecs[:, np.argmin(eigvals)]
        normals.append(normal)

    return np.array(normals)


def compute_point_pair_features(p_s, n_s, p_t, n_t):
    """
    Calculate the PFH features between two points.
    :param p_s: Source point coordinate
    :param n_s: Source point normal vector
    :param p_t: Target point coordinate
    :param n_t: Target point normal vector
    :return: PFH feature vector (alpha, phi, theta, d)
    Complexity O(1)
    """
    dp = p_t - p_s
    d = np.linalg.norm(dp)
    dp_normalized = dp / (d + 1e-10)  # Avoid division by zero

    alpha = np.arccos(np.clip(np.dot(n_s, dp_normalized), -1.0, 1.0))
    phi = np.arccos(np.clip(np.dot(n_t, dp_normalized), -1.0, 1.0))
    theta = np.arccos(np.clip(np.dot(n_s, n_t), -1.0, 1.0))

    return alpha, phi, theta, d


def get_tranform(p, q):
    p_mean = np.mean(p, axis=0)
    q_mean = np.mean(q, axis=0)
    p_centered = p - p_mean
    q_centered = q - q_mean
    S = np.matmul(p_centered.reshape(-1, 3).T, q_centered.reshape(-1, 3))
    u, sigma, vt = svd(S)
    third = np.linalg.det(np.matmul(u, vt))
    R = np.matmul(vt.T, np.array([[1, 0, 0], [0, 1, 0], [0, 0, third]]))
    R = np.matmul(R, u.T)
    return R, q_mean - np.matmul(R, p_mean)


def cal_distance(pc1, pc2):
    pc1_expanded = pc1.reshape(-1, 1, 3)  # (N, 1, 3)
    pc2_expanded = pc2.reshape(1, -1, 3)  # (1, M, 3)

    distances = np.linalg.norm(pc1_expanded - pc2_expanded, axis=2)  # (N, M)

    # Find the minimum distance from each point to the nearest point
    min_distances = np.min(distances, axis=1)  # (N,)

    # Calculate the average distance
    return np.mean(min_distances)


def view_pc(pcs, fig=None, color="b", marker="o"):
    """Visualize a pc.

    inputs:
        pc - a list of numpy 3 x 1 matrices that represent the points.
        color - specifies the color of each point cloud.
            if a single value all point clouds will have that color.
            if an array of the same length as pcs each pc will be the color corresponding to the
            element of color.
        marker - specifies the marker of each point cloud.
            if a single value all point clouds will have that marker.
            if an array of the same length as pcs each pc will be the marker corresponding to the
            element of marker.
    outputs:
        fig - the pyplot figure that the point clouds are plotted on

    """
    markersize = 0.2
    alpha = 0.1
    # Construct the color and marker arrays
    if hasattr(color, "__iter__"):
        if len(color) != len(pcs):
            raise Exception("color is not the same length as pcs")
    else:
        color = [color] * len(pcs)

    if hasattr(marker, "__iter__"):
        if len(marker) != len(pcs):
            raise Exception("marker is not the same length as pcs")
    else:
        marker = [marker] * len(pcs)

    if hasattr(markersize, "__iter__"):
        if len(markersize) != len(pcs):
            raise Exception("markersize is not the same length as pcs")
    else:
        markersize = [markersize] * len(pcs)

    if hasattr(alpha, "__iter__"):
        if len(alpha) != len(pcs):
            raise Exception("alpha is not the same length as pcs")
    else:
        alpha = [alpha] * len(pcs)

    # Start plt in interactive mode
    ax = []
    if fig == None:
        plt.ion()
        # Make a 3D figure
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

    else:
        ax = fig.gca()

    # Draw each point cloud
    for pc, c, m in zip(pcs, color, marker):
        x = []
        y = []
        z = []
        for pt in pc:
            x.append(pt[0])
            y.append(pt[1])
            z.append(pt[2])

        ax.scatter3D(x, y, z, color=c, marker=m, s=markersize[0], alpha=0.1)

    # Set the labels
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.set_xlim([0.3, 0.9])
    ax.set_ylim([0.2, 0.8])
    ax.set_zlim([0.4, 0.8])

    # Update the figure
    plt.draw()
    plt.pause(0.05)
    plt.ioff()  # turn off interactive plotting

    # Return a handle to the figure so the user can make adjustments
    return fig


def show_pic(source, target, aligned, normal=False):
    fig = view_pc([source])
    view_pc([target], fig, "r")
    view_pc([aligned], fig, "g")
    plt.show()
    if normal:
        fig.set_xlim([0, 1])
        fig.set_ylim([0, 1])
        fig.set_zlim([0, 1])
    pass
