from contextlib import contextmanager
from pathlib import Path
from typing import Tuple, List, Dict, Union, Callable

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from tqdm import tqdm, trange

import torch
from torch import Tensor
from pandas import DataFrame
import numpy as np
import cv2 as cv

from gixd_detectron import *
from gixd_detectron.exp_utils import *
from gixd_detectron.plot_utils import *
from gixd_detectron.plot_utils.plot_res import plot_test_results

from gixd_detectron.ml.postprocessing import (
    Postprocessing,
    PostprocessingPipeline,
    StandardPostprocessing,
    MergeBoxesPostprocessing,
    SmallQFilter,
    TargetBasedSmallQFilter,
)

from gixd_detectron.metrics.match_criteria import (
    Matcher,
    IoUMatcher,
    QMatcher,
)

from gixd_detectron.metrics.average_precision import recall_precision_curve_with_intensities

from utensil import (group_detections, calculate_weighted_average, calculate_weighted_average_and_score,
                     filter_boxes_by_score, calculate_x_center_precision, calculate_y_center_precision,
                     calculate_center_distance,
                     draw_boxes_basic, flip_image_variations, flip_boxes)

import pickle

def postprocessing_images(predictions, scores):
    min_score = 0.01

    # Initialize postprocessing steps with desired parameters
    standard_postprocessing = StandardPostprocessing(nms_level=0.01, score_level=min_score)
    small_q_filter = SmallQFilter(min_q_pix=50.0)
    merge_boxes_postprocessing = MergeBoxesPostprocessing(
        min_score=min_score)  # , min_iou=0.5, max_q=5.0, mode='mean-quantile',
    # quantile=0.8)

    # Combine postprocessing steps into a pipeline
    pipeline = PostprocessingPipeline(
        standard_postprocessing,
        small_q_filter,
        merge_boxes_postprocessing
    )

    refined_predictions, refined_scores = pipeline(predictions, scores)

    return refined_predictions, refined_scores

# load in peak detection model
model_path = 'model_gixd_1-120.pt'
model = torch.load(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)
# load in images as a np array
x = np.load("images_s_for_detectron.npy")
# ground truths as np array
labels = np.load("boxes_for_detectron.npy", allow_pickle=True)
# confidences as np array
confidences = np.load("scores_for_detectron.npy", allow_pickle=True)

model.to(device)

fp_wa = 0
tp_wa = 0
tn_wa = 0
x_diff_wa = []
y_diff_wa = []

fp_standard = 0
tp_standard = 0
tn_standard = 0
x_diff_standard = []
y_diff_standard = []

# Initialize cumulative counts for TP and FN for each score level
cumulative_tps = {1: 0, 0.5: 0, 0.1: 0}
cumulative_fns = {1: 0, 0.5: 0, 0.1: 0}

# Initialize cumulative counts for TP and FN for each score level
cumulative_tps_standard = {1: 0, 0.5: 0, 0.1: 0}
cumulative_fns_standard = {1: 0, 0.5: 0, 0.1: 0}

model.eval()

predictions = []
for i, image in enumerate(x):
    print(i)
    images_flipped = flip_image_variations(image)
    for im in range(len(images_flipped)):
        t = torch.tensor(images_flipped[im].copy(), dtype=torch.float, device=device).unsqueeze(0).unsqueeze(0)
        pred, score = model(t)
        pred_np = pred[0].cpu().detach()
        score_np = score[0].cpu().detach()
        pred, score = postprocessing_images(pred_np, score_np)
        plot = False
        if im == 0:
            pred0 = pred.numpy().copy()
            score0 = score.numpy().copy()
            predictions.append(pred0)
            if plot:
                plt.figure(figsize=(15, 7.5))
                plt.imshow(images_flipped[0])
                plt.axis('off')
                draw_boxes_basic(labels[i], color="b")
                draw_boxes_basic(pred)
                plt.savefig("images/" + str(i) + "_o" + str(im), bbox_inches='tight', pad_inches=0)
                plt.close()

                plt.figure(figsize=(15, 7.5))
                plt.imshow(images_flipped[0])
                plt.axis('off')
                plt.savefig("images/" + str(i), bbox_inches='tight', pad_inches=0)
                plt.close()

        elif im == 1:
            pred = flip_boxes(pred.numpy(), 'vertical', (512, 1024))
            pred1 = pred.copy()
            score1 = score.numpy().copy()
            if plot:
                plt.figure(figsize=(15, 7.5))
                plt.imshow(images_flipped[0])
                plt.axis('off')
                draw_boxes_basic(labels[i], color="b")
                draw_boxes_basic(pred)
                plt.savefig("images/" + str(i) + "_o" + str(im), bbox_inches='tight', pad_inches=0)
                plt.close()
        elif im == 2:
            pred = flip_boxes(pred.numpy(), 'horizontal', (512, 1024))
            pred2 = pred.copy()
            score2 = score.numpy().copy()
            if plot:
                plt.figure(figsize=(15, 7.5))
                plt.imshow(images_flipped[0])
                plt.axis('off')
                draw_boxes_basic(labels[i], color="b")
                draw_boxes_basic(pred)
                plt.savefig("images/" + str(i) + "_o" + str(im), bbox_inches='tight', pad_inches=0)
                plt.close()
        elif im == 3:
            pred = flip_boxes(pred.numpy(), 'horizontal', (512, 1024))
            pred = flip_boxes(pred, 'vertical', (512, 1024))
            pred3 = pred.copy()
            score3 = score.numpy().copy()
            if plot:
                plt.figure(figsize=(15, 7.5))
                plt.imshow(images_flipped[0])
                plt.axis('off')
                draw_boxes_basic(labels[i], color="b")
                draw_boxes_basic(pred)
                plt.savefig("images/" + str(i) + "_o" + str(im), bbox_inches='tight', pad_inches=0)
                plt.close()

    pred_col = np.concatenate((pred0, pred1, pred2, pred3), axis=0)
    score_col = np.concatenate((score0, score1, score2, score3), axis=None)

    # groups = group_detections(pred_col, score_col, iou_threshold=0.1)
    # pred_ave, score_ave = calculate_weighted_average_and_score(groups, pred_col, score_col, min_presence=4)

    pred_ave, score_ave = postprocessing_images(torch.from_numpy(pred_col), torch.from_numpy(score_col))
    # pred_ave, score_ave = postprocessing_images(torch.from_numpy(pred_ave), torch.from_numpy(score_ave))
    pred_ave = pred_ave.numpy().copy()
    score_ave = score_ave.numpy().copy()

    # pred_ave, score_ave = filter_boxes_by_score(pred_ave, score_ave, 0.6)

    # plt.figure(figsize=(15, 7.5))
    # plt.imshow(images_flipped[0])
    # draw_boxes_basic(labels[i], color="b")
    # draw_boxes_basic(pred_ave)
    # plt.axis('off')
    # plt.savefig("images/" + str(i) + "_min2", bbox_inches='tight', pad_inches=0)
    # plt.close()

    # plt.figure(figsize=(15, 7.5))
    # plt.imshow(images_flipped[0])
    # draw_boxes_basic(pred_ave)
    # draw_boxes_basic(labels[i], color="b")
    # plt.savefig("images/" + str(i) + "")
    # plt.close()

    predicted = pred_ave.copy()
    target = labels[i].copy()
    target_standard = labels[i].copy()

    conf = np.array(confidences[i])

    """FOR WA"""
    matcher = QMatcher(10., rel=False, min_iou=0.01)

    match_matrix, row_indices, col_indices = matcher(torch.from_numpy(target),
                                                     torch.from_numpy(predicted))

    missed_indices = np.array(list(set(np.arange(target.shape[0])).difference(row_indices)))
    fp_indices = np.array(list(set(np.arange(predicted.shape[0])).difference(col_indices)))

    tn_wa += missed_indices.size
    fp_wa += fp_indices.size

    tp_wa += np.sum(match_matrix[row_indices, col_indices] < 1)

    avg_x_diff_wa = calculate_x_center_precision(match_matrix, row_indices, col_indices, target, predicted)
    print(avg_x_diff_wa)
    x_diff_wa.append(avg_x_diff_wa)

    avg_y_diff_wa = calculate_y_center_precision(match_matrix, row_indices, col_indices, target, predicted)
    y_diff_wa.append(avg_y_diff_wa)

    # Calculate the indices of ground truth boxes that were missed (FN)
    all_indices = set(range(len(conf)))
    matched_indices = set(row_indices)
    missed_indices = np.array(list(all_indices - matched_indices), dtype=int)

    # Initialize dictionaries to hold TP and FN counts for each score level
    true_positives_by_score = {1: 0, 0.5: 0, 0.1: 0}
    false_negatives_by_score = {1: 0, 0.5: 0, 0.1: 0}

    # Calculate TP for each score level using matched ground truth scores
    matched_scores = conf[row_indices]
    for score_level in [1, 0.5, 0.1]:
        true_positives_by_score[score_level] = np.sum(matched_scores == score_level)

    # Calculate FN for each score level using missed ground truth scores
    missed_scores = conf[missed_indices]
    for score_level in [1, 0.5, 0.1]:
        false_negatives_by_score[score_level] = np.sum(missed_scores == score_level)

    # Update cumulative counts for TP and FN for each score level
    for score_level in [1, 0.5, 0.1]:
        cumulative_tps[score_level] += true_positives_by_score[score_level]
        cumulative_fns[score_level] += false_negatives_by_score[score_level]

    # Printing the counts for TP and FN for each score level
    for score_level in [1, 0.5, 0.1]:
        print(
            f"Score Level {score_level}: True Positives = {true_positives_by_score[score_level]}, False Negatives = {false_negatives_by_score[score_level]}")

    """FOR STANDARD"""
    matcher = QMatcher(10., rel=False, min_iou=0.01)

    match_matrix, row_indices, col_indices = matcher(torch.from_numpy(target_standard),
                                                     torch.from_numpy(pred0))

    missed_indices = np.array(list(set(np.arange(target_standard.shape[0])).difference(row_indices)))
    fp_indices = np.array(list(set(np.arange(pred0.shape[0])).difference(col_indices)))

    tn_standard += missed_indices.size
    fp_standard += fp_indices.size

    tp_standard += np.sum(match_matrix[row_indices, col_indices] < 1)

    avg_x_diff_standard = calculate_x_center_precision(match_matrix, row_indices, col_indices, target_standard, pred0)
    print(avg_x_diff_standard)
    x_diff_standard.append(avg_x_diff_standard)

    avg_y_diff_standard = calculate_y_center_precision(match_matrix, row_indices, col_indices, target_standard, pred0)
    y_diff_standard.append(avg_y_diff_standard)

    # Calculate the indices of ground truth boxes that were missed (FN)
    all_indices = set(range(len(conf)))
    matched_indices = set(row_indices)
    missed_indices = np.array(list(all_indices - matched_indices), dtype=int)

    # Initialize dictionaries to hold TP and FN counts for each score level
    true_positives_by_score = {1: 0, 0.5: 0, 0.1: 0}
    false_negatives_by_score = {1: 0, 0.5: 0, 0.1: 0}

    # Calculate TP for each score level using matched ground truth scores
    matched_scores = conf[row_indices]
    for score_level in [1, 0.5, 0.1]:
        true_positives_by_score[score_level] = np.sum(matched_scores == score_level)

    # Calculate FN for each score level using missed ground truth scores
    missed_scores = conf[missed_indices]
    for score_level in [1, 0.5, 0.1]:
        false_negatives_by_score[score_level] = np.sum(missed_scores == score_level)

    # Update cumulative counts for TP and FN for each score level
    for score_level in [1, 0.5, 0.1]:
        cumulative_tps_standard[score_level] += true_positives_by_score[score_level]
        cumulative_fns_standard[score_level] += false_negatives_by_score[score_level]

    # Printing the counts for TP and FN for each score level
    for score_level in [1, 0.5, 0.1]:
        print(
            f"Score Level {score_level}: True Positives = {true_positives_by_score[score_level]}, False Negatives = {false_negatives_by_score[score_level]}")

print("WA")
print("TP: " + str(tp_wa))
print("FN: " + str(tn_wa))
for score_level in [1, 0.5, 0.1]:
    print(f"Cumulative True Positives for score {score_level}: {cumulative_tps[score_level]}")
    print(f"Cumulative False Negatives for score {score_level}: {cumulative_fns[score_level]}")
    print(f"Recall for score {score_level}: {round(cumulative_tps[score_level] / (cumulative_tps[score_level] + cumulative_fns[score_level]), 4)}")
print("FP: " + str(fp_wa))
print("Average x distance difference: " + str(np.mean(np.array(x_diff_wa))/2))
print("Average y distance difference: " + str(np.mean(np.array(y_diff_wa))/2))

print("\nStandard")
print("TP: " + str(tp_standard))
print("FN: " + str(tn_standard))
for score_level in [1, 0.5, 0.1]:
    print(f"Cumulative True Positives for score {score_level}: {cumulative_tps_standard[score_level]}")
    print(f"Cumulative False Negatives for score {score_level}: {cumulative_fns_standard[score_level]}")
    print(
        f"Recall for score {score_level}: {round(cumulative_tps_standard[score_level] / (cumulative_tps_standard[score_level] + cumulative_fns_standard[score_level]), 4)}")
print("FP: " + str(fp_standard))
print("Average x distance difference: " + str(np.mean(np.array(x_diff_standard))/2))
print("Average y distance difference: " + str(np.mean(np.array(y_diff_standard))/2))

predictions = np.array(predictions, dtype=object)
with open('object_arrays.pkl', 'wb') as file:
    pickle.dump(predictions, file)
# np.save('own_predictions.npy', predictions)
