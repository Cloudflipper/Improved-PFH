import numpy as np
from scipy.spatial import KDTree
import utility as utils
from timer import Timer
from joblib import Parallel, delayed
from filter import expolre_histogram, retainment_analysis


def cnt_point(sz):
    return min(sz, 8 * np.log2(sz))


def compute_normals_within_query(points, radius, query_size=-1):
    normals = []
    kdtree = KDTree(points)
    if query_size == -1:
        query_size = cnt_point(len(points))
    for p in points:
        indices = kdtree.query(p, k=query_size)
        indices = indices[1]
        if len(indices) < 3:
            normals.append(np.array([0.0, 0.0, 0.0]))
            continue
        neighbors = points[indices]
        centroid = np.mean(neighbors, axis=0)
        cov_matrix = np.cov((neighbors - centroid).T)
        eigvals, eigvecs = np.linalg.eigh(cov_matrix)
        normal = eigvecs[:, np.argmin(eigvals)]
        normals.append(normal)

    return np.array(normals)


def compute_fpfh_histogram_1D(
    points, normals, num_bins=5, radius=0.1, query_way=cnt_point
):
    kdtree = KDTree(points)
    pfh_descriptors = []
    indice_set = []

    for i, (p, n) in enumerate(zip(points, normals)):
        histogram = np.zeros(num_bins * 3)
        indices = kdtree.query(p, cnt_point(len(points)))
        dist = indices[0]
        indices = indices[1]
        indice_set.append(indices)
        for j in indices:
            if i == j:
                continue
            features = utils.compute_point_pair_features(p, n, points[j], normals[j])
            alpha, phi, theta, d = features
            alpha_bin = int(alpha / np.pi * num_bins)
            phi_bin = int(phi / np.pi * num_bins)
            theta_bin = int(theta / np.pi * num_bins)

            alpha_bin = min(alpha_bin, num_bins - 1)
            phi_bin = min(phi_bin, num_bins - 1)
            theta_bin = min(theta_bin, num_bins - 1)

            weight = 1 - ((d - dist[-1] / 2) / dist[-1]) ** 2 * 4
            histogram[alpha_bin] += 1
            histogram[phi_bin + num_bins] += 1
            histogram[theta_bin + 2 * num_bins] += 1

        pfh_descriptors.append(histogram.flatten())

    fpfh_descriptors = []
    for i, (p, spfh, indices) in enumerate(zip(points, pfh_descriptors, indice_set)):
        fpfh = spfh
        for j in indices:
            if i == j:
                continue
            distance = np.linalg.norm(points[i] - points[j])
            weight = radius * 10 / (distance + 1e-6)
            fpfh += weight * pfh_descriptors[j]

        fpfh /= len(indices)
        fpfh_descriptors.append(fpfh)

    return np.array(fpfh_descriptors)


# def icp_with_fpfh_within_query(
#     source,
#     target,
#     max_iters=50,
#     tolerance=1e-6,
#     num_bins=20,
#     radius=0.03,
#     patience=10,
#     timer=None,
# ):

#     source_normals = compute_normals_within_query(source, radius=0.1)
#     target_normals = compute_normals_within_query(target, radius=0.1)

#     if timer is not None:
#         print(f"Normal calculation time is: {timer.clip():.4f}")

#     results = Parallel(n_jobs=2, backend="loky")(
#         delayed(compute_fpfh_histogram_1D)(
#             data,
#             normals,
#             num_bins,
#             radius=radius,  # timer=timer
#         )
#         for data, normals in [(source, source_normals), (target, target_normals)]
#     )

#     source_pfh, target_pfh = results

#     if timer is not None:
#         print(f"PFH calculation time is: {timer.clip():.4f}")

#     curr_patience = last_error = 0
#     for iteration in range(max_iters):
#         correspondences = []
#         for i, s_desc in enumerate(source_pfh):
#             distances = np.linalg.norm(target_pfh - s_desc, axis=1)
#             closest_idx = np.argmin(distances)
#             correspondences.append((i, closest_idx))

#         src_points = np.array([source[i] for i, _ in correspondences])
#         tgt_points = np.array([target[j] for _, j in correspondences])

#         R, t = utils.get_tranform(src_points, tgt_points)

#         source = np.dot(source, R.T) + t

#         mean_error = np.mean(
#             np.linalg.norm(np.dot(src_points, R.T) + t - tgt_points, axis=1)
#         )
#         if mean_error < tolerance:
#             print("Final Error:", f"{mean_error:.4g}")
#             break
#         if mean_error < last_error * 0.999:
#             curr_patience = 0
#         else:
#             curr_patience += 1
#         if curr_patience > patience:
#             print("Final Error:", f"{mean_error:.4g}")
#             break
#         last_error = mean_error

#     if timer is not None:
#         print(f"ICP calculation time is: {timer.clip():.4f}")

#     return source


def icp_with_fpfh_within_query(
    source,
    target,
    max_iters=50,
    tolerance=1e-6,
    num_bins=20,
    radius=0.03,
    patience=10,
    timer=None,
    filter_function="arctan",
    retainment_ratio="dense",
    clip_mode="normal",
):

    retainment_nums = []

    source_normals = compute_normals_within_query(source, radius=0.1)
    target_normals = compute_normals_within_query(target, radius=0.1)

    if timer is not None:
        print(f"Normal calculation time is: {timer.clip():.4f}")

    results = Parallel(n_jobs=2, backend="loky")(
        delayed(compute_fpfh_histogram_1D)(data, normals, num_bins, radius=radius)
        for data, normals in [(source, source_normals), (target, target_normals)]
    )

    source_pfh, target_pfh = results

    if timer is not None:
        print(f"PFH calculation time is: {timer.clip():.4f}")

    work_source_index = expolre_histogram(
        source_pfh,
        num_bins,
        filter_function,
        retainment_ratio,
        clip_mode,
        retainment_nums,
    )
    work_target_index = expolre_histogram(
        target_pfh,
        num_bins,
        filter_function,
        retainment_ratio,
        clip_mode,
        retainment_nums,
    )

    work_source = source[work_source_index]
    target_source = target[work_target_index]
    # utils.show_pic(source, work_source, target_source)
    if timer is not None:
        print(f"PFH filter time is: {timer.clip():.4f}")

    curr_patience = last_error = 0
    min_error = np.inf
    for iteration in range(max_iters):
        correspondences = []
        for i, s_desc in enumerate(source_pfh[work_source_index]):
            distances = np.linalg.norm(target_pfh[work_target_index] - s_desc, axis=1)
            closest_idx = np.argmin(distances)
            correspondences.append((i, closest_idx))

        src_points = np.array([work_source[i] for i, _ in correspondences])
        tgt_points = np.array([target_source[j] for _, j in correspondences])

        R, t = utils.get_tranform(src_points, tgt_points)

        source = np.dot(source, R.T) + t

        mean_error = np.mean(
            np.linalg.norm(np.dot(src_points, R.T) + t - tgt_points, axis=1)
        )

        print("Iteration: ", iteration, "    Error: ", mean_error)
        if mean_error < tolerance:
            print("Final Error:", f"{mean_error:.4g}")
            break
        if mean_error < min_error * 0.9:
            curr_patience = 0
            min_error = mean_error
        else:
            curr_patience += 1
            min_error = min(mean_error, min_error)
        if curr_patience > patience:
            print("Final Error:", f"{mean_error:.4g}")
            break

        source_normals = compute_normals_within_query(source, radius=0.1)
        source_pfh = compute_fpfh_histogram_1D(
            source, source_normals, num_bins, radius=radius
        )
        work_source_index = expolre_histogram(
            source_pfh,
            num_bins,
            filter_function,
            retainment_ratio,
            clip_mode,
            retainment_nums,
        )
        work_source = source[work_source_index]

    if timer is not None:
        print(f"ICP calculation time is: {timer.clip():.4f}")

    retainment_analysis(retainment_nums)
    return source


def main():
    source = np.loadtxt("source_cloud.csv", delimiter=",")
    target = np.loadtxt("cloud_icp_target3.csv", delimiter=",")
    work_timer = Timer()
    work_timer.start()
    aligned_cloud = icp_with_fpfh_within_query(
        source, target, radius=0.02, num_bins=8, max_iters=1000, timer=work_timer
    )
    work_timer.stop()
    print(f"Elapsed time: {work_timer.get_elapsed_time():.6f} seconds")
    utils.show_pic(source, target, aligned_cloud)
    pass
    # for radius in np.linspace(0.01,0.1,10):
    #     aligned_cloud = icp_with_pfh(source, target,radius=radius)
    # # fig=utils.view_pc([source])
    # # utils.view_pc([target],fig,'r')
    # # np.savetxt('aligned_cloud.csv', aligned_cloud, delimiter=',')
    # # utils.view_pc([aligned_cloud],fig,'g')
    # # plt.show()
    #     print(radius,f"{cal_distance(aligned_cloud,target):.6f}")
    # print("SAVED AS 'aligned_cloud.csv'")


if __name__ == "__main__":
    main()
