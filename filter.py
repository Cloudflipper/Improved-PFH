# import utility as utils
import numpy as np
import scipy.stats as stats


def arct_func(evaluator, retainment_ratio):
    if retainment_ratio == "dense":
        return (1 / np.pi) * np.arctan(-20 * evaluator + 20) + 0.65
    if retainment_ratio == "medium":
        return (1 / np.pi) * np.arctan(-20 * evaluator + 16) + 0.65
    if retainment_ratio == "sparse":
        return (1 / np.pi) * np.arctan(-20 * evaluator + 12) + 0.65
    print("Please set retainment ratio to dense, medium, or sparse. Filter failed")
    return np.ones_like(evaluator)


def clip_func(evaluator, retainment_ratio):
    if retainment_ratio == "dense":
        return (evaluator < 1).astype(int)
    if retainment_ratio == "medium":
        return (evaluator < 0.9).astype(int)
    if retainment_ratio == "sparse":
        return (evaluator < 0.8).astype(int)
    print("Please set retainment ratio to dense, medium, or sparse. Filter failed")
    return np.ones_like(evaluator)


def smooth_func(evaluator, retainment_ratio):
    if retainment_ratio == "dense":
        return (1 / np.pi) * np.arctan(-10 * evaluator + 10) + 0.65
    if retainment_ratio == "medium":
        return (1 / np.pi) * np.arctan(-10 * evaluator + 8) + 0.65
    if retainment_ratio == "sparse":
        return (1 / np.pi) * np.arctan(-10 * evaluator + 6) + 0.65
    print("Please set retainment ratio to dense, medium, or sparse. Filter failed")
    return np.ones_like(evaluator)


def call_filter_function(
    evaluator, function_type="arctan", retainment_ratio="high", clip_mode="normal"
):
    if function_type == "arctan":
        retention_probs = arct_func(evaluator, retainment_ratio)
    elif function_type == "clip":
        retention_probs = clip_func(evaluator, retainment_ratio)
    elif function_type == "smooth":
        retention_probs = smooth_func(evaluator, retainment_ratio)
    else:
        print("Please set function_type to smooth, arctan or clip. Filter failed")
        return np.ones_like(evaluator)
    if clip_mode == "normal":
        return np.clip(retention_probs, 0.15, 1)
    elif clip_mode == "protective":
        return np.clip(retention_probs, 0.25, 1)
    elif clip_mode == "light":
        return np.clip(retention_probs, 0.15, 0.75)
    elif clip_mode == "protective-light":
        return np.clip(retention_probs, 0.25, 0.75)
    else:
        print(
            "Please set clip mode to normal, protective, light or protective-light. Clip failed. Use normal clip."
        )
        return np.clip(retention_probs, 0.15, 1)


def expolre_histogram(
    fpfh,
    num_bins=20,
    filter_function="arctan",
    retainment_ratio="dense",
    clip_mode="normal",
    retainment_nums=None,
):

    evaluator = np.array(
        [
            (
                np.max(fpfh[i, 0:num_bins]) / np.sum(fpfh[i, 0:num_bins])
                + np.max(fpfh[i, num_bins : 2 * num_bins])
                / np.sum(fpfh[i, num_bins : 2 * num_bins])
                + np.max(fpfh[i, 2 * num_bins : 3 * num_bins])
                / np.sum(fpfh[i, 2 * num_bins : 3 * num_bins])
            )
            for i in range(fpfh.shape[0])
        ]
    )
    retention_probs = call_filter_function(
        evaluator, filter_function, retainment_ratio, clip_mode="normal"
    )

    sampled_mask = np.random.rand(fpfh.shape[0]) < retention_probs
    sampled_indices = np.where(sampled_mask)[0]
    retainment_num = sampled_indices.size / fpfh.shape[0]
    print("Retainment Ratio", retainment_num)
    if retainment_nums is not None:
        retainment_nums.append(retainment_num)
    return sampled_indices


def retainment_analysis(retainment_nums):
    if len(retainment_nums) == 0:
        print("Vacant list")
    data = np.array(retainment_nums)

    max_value = np.max(data)
    min_value = np.min(data)
    mean_value = np.mean(data)

    confidence = 0.90
    n = len(data)
    stderr = stats.sem(data)
    margin_of_error = stderr * stats.t.ppf((1 + confidence) / 2, n - 1)
    lower_bound = mean_value - margin_of_error
    upper_bound = mean_value + margin_of_error

    print(f"MAX: {max_value}")
    print(f"MIN: {min_value}")
    print(f"AVG: {mean_value}")
    print(f"90% confidence interval: ({lower_bound}, {upper_bound})")
