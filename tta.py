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


def tta(image, min_presence=2):

    images_flipped = flip_image_variations(image)
    for im in range(len(images_flipped)):
        t = torch.tensor(images_flipped[im].copy(), dtype=torch.float, device=device).unsqueeze(0).unsqueeze(0)
        pred, score = model(t)
        pred_np = pred[0].cpu().detach()
        score_np = score[0].cpu().detach()
        pred, score = postprocessing_images(pred_np, score_np)

        if im == 0:
            pred0 = pred.numpy().copy()
            score0 = score.numpy().copy()

        elif im == 1:
            pred = flip_boxes(pred.numpy(), 'vertical', (512, 1024))
            pred1 = pred.copy()
            score1 = score.numpy().copy()

        elif im == 2:
            pred = flip_boxes(pred.numpy(), 'horizontal', (512, 1024))
            pred2 = pred.copy()
            score2 = score.numpy().copy()

        elif im == 3:
            pred = flip_boxes(pred.numpy(), 'horizontal', (512, 1024))
            pred = flip_boxes(pred, 'vertical', (512, 1024))
            pred3 = pred.copy()
            score3 = score.numpy().copy()

    pred_col = np.concatenate((pred0, pred1, pred2, pred3), axis=0)
    score_col = np.concatenate((score0, score1, score2, score3), axis=None)

    groups = group_detections(pred_col, score_col, iou_threshold=0.1)
    pred_ave, score_ave = calculate_weighted_average_and_score(groups, pred_col, score_col, min_presence=min_presence)

    pred_ave, score_ave = postprocessing_images(torch.from_numpy(pred_ave), torch.from_numpy(score_ave))
    pred_ave = pred_ave.numpy().copy()
    score_ave = score_ave.numpy().copy()

    return pred_ave, score_ave


if __name__ == '__main__':
    # load in model for peak detection
    model_path = 'model_gixd_1-120.pt'
    model = torch.load(model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(device)
    # load in images as a np array
    test_images = np.load("images_s_for_detectron.npy")
    model.to(device)
    model.eval()

    for i, image in enumerate(test_images):
        predictions, score = tta(image, min_presence=2)
        print(predictions)
