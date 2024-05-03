import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def draw_boxes_basic(boxes, color='r'):
    """
    draws boxes
    """
    for box in boxes:
        x = box[0]
        y = box[3]
        x_width = box[2] - box[0]
        y_width = box[3] - box[1]
        rect = patches.Rectangle((x, y), x_width, -y_width, linewidth=2, edgecolor=color, facecolor='none')
        plt.gca().add_patch(rect)


def flip_image_variations(image):
    # Original Image
    original_image = image

    # Flip vertically (along the horizontal axis)
    flipped_vertically = np.flipud(image)

    # Flip horizontally (along the vertical axis)
    flipped_horizontally = np.fliplr(image)

    # Flip both vertically and horizontally
    flipped_both = np.flipud(np.fliplr(image))

    return [original_image, flipped_vertically, flipped_horizontally, flipped_both]


def flip_boxes(boxes, flip_type, image_size):
    """
    Adjust bounding box coordinates for flipped images.

    Parameters:
    - boxes (np.array): Array of bounding boxes, shape (N, 4) with structure [x_min, y_min, x_max, y_max].
    - flip_type (str): Type of flip, 'horizontal' or 'vertical'.
    - image_size (tuple): Size of the image as (height, width).

    Returns:
    - np.array: Adjusted bounding boxes.
    """
    flipped_boxes = boxes.copy()
    H, W = image_size

    if flip_type == 'horizontal':
        flipped_boxes[:, [0, 2]] = W - boxes[:, [2, 0]]
    elif flip_type == 'vertical':
        flipped_boxes[:, [1, 3]] = H - boxes[:, [3, 1]]
    else:
        raise ValueError("flip_type must be 'horizontal' or 'vertical'")

    return flipped_boxes




def iou_np(boxes1, boxes2):
    """
    Compute IoU between two sets of boxes using NumPy for efficiency.

    :param boxes1: NumPy array of bounding boxes, shape (N, 4).
    :param boxes2: NumPy array of bounding boxes, shape (M, 4).
    :return: IoU matrix of shape (N, M) where each element [i, j] is the IoU of boxes1[i] and boxes2[j].
    """
    # Split the coordinates for easier computation
    x11, y11, x12, y12 = np.split(boxes1, 4, axis=1)
    x21, y21, x22, y22 = np.split(boxes2, 4, axis=1)

    # Compute intersections
    xI1 = np.maximum(x11, np.transpose(x21))
    yI1 = np.maximum(y11, np.transpose(y21))
    xI2 = np.minimum(x12, np.transpose(x22))
    yI2 = np.minimum(y12, np.transpose(y22))
    inter_area = np.maximum((xI2 - xI1), 0) * np.maximum((yI2 - yI1), 0)

    # Compute unions
    box1_area = (x12 - x11) * (y12 - y11)
    box2_area = (x22 - x21) * (y22 - y21)
    union_area = box1_area + np.transpose(box2_area) - inter_area

    # Compute IoU
    iou = inter_area / union_area
    return iou


def group_detections(predictions, scores, iou_threshold=0.5):
    """
    Group detections based on IoU threshold, optimized with NumPy.

    :param predictions: NumPy array of bounding box coordinates, shape (N, 4).
    :param scores: NumPy array of confidence scores, shape (N,).
    :param iou_threshold: IoU threshold for grouping.
    :return: A list of tuples, each containing (group_indices, group_scores).
    """
    n = len(predictions)
    indices = np.arange(n)
    grouped_indices = []
    grouped_scores = []

    while indices.size > 0:
        current = indices[0]
        current_box = predictions[current:current + 1]  # Make it (1, 4) for broadcasting
        ious = iou_np(current_box, predictions[indices])[0]  # Compute IoUs with the rest

        # Find detections with IoU above the threshold
        in_group = indices[ious >= iou_threshold]

        # Group indices and scores
        grouped_indices.append(in_group)
        grouped_scores.append(scores[in_group])

        # Remove grouped detections from further consideration
        indices = np.setdiff1d(indices, in_group)

    return [(group, score) for group, score in zip(grouped_indices, grouped_scores)]


def calculate_weighted_average(groups, predictions, scores):
    """
    Calculate the weighted average of bounding box coordinates and weighted scores for each group based on confidence scores.

    :param groups: A list of tuples, each containing (group_indices, group_scores) for each group.
    :param predictions: NumPy array of bounding box coordinates, shape (N, 4).
    :param scores: NumPy array of confidence scores, shape (N,).
    :return: A tuple containing two lists:
             - A list of weighted averaged bounding boxes for each group.
             - A list of weighted scores for each group.
    """
    weighted_averages = []
    weighted_scores = []

    for group_indices, group_scores in groups:
        # Extract the bounding boxes and scores for the current group
        group_boxes = predictions[group_indices]
        # Normalize scores to sum to 1 for weighted averaging
        weights = group_scores / group_scores.sum()

        # Calculate the weighted average for each coordinate (x_min, y_min, x_max, y_max)
        weighted_avg_box = np.dot(weights, group_boxes)
        weighted_averages.append(weighted_avg_box)

        # Calculate the weighted score for the group
        weighted_score = np.dot(weights, group_scores)
        weighted_scores.append(weighted_score)

    return np.array(weighted_averages), np.array(weighted_scores)


def calculate_weighted_average_and_score(groups, predictions, scores, min_presence=2):
    """
    Calculate the weighted average of bounding box coordinates and weighted scores for each group
    based on confidence scores. Assumes detections are consistent across a minimum number of augmentations.

    :param groups: A list of tuples, each containing (group_indices, group_scores).
    :param predictions: NumPy array of bounding box coordinates, shape (N, 4).
    :param scores: NumPy array of confidence scores, shape (N,).
    :param min_presence: Minimum number of augmentations a detection must appear in to be aggregated.
    :return: A tuple containing two lists:
             - A list of weighted averaged bounding boxes for groups meeting the min_presence criteria.
             - A list of weighted scores for these groups.
    """
    weighted_averages = []
    weighted_scores = []

    for group_indices, group_scores in groups:
        if len(group_indices) < min_presence:
            continue  # Skip groups not present in enough augmentations

        # Normalize scores to sum to 1 for weighted averaging
        weights = group_scores / group_scores.sum()

        # Calculate the weighted average for each coordinate (x_min, y_min, x_max, y_max)
        weighted_avg_box = np.dot(weights, predictions[group_indices])
        weighted_averages.append(weighted_avg_box)

        # Calculate the weighted score for the group
        weighted_score = np.dot(weights, group_scores)
        weighted_scores.append(weighted_score)

    return np.array(weighted_averages), np.array(weighted_scores)


def filter_boxes_by_score(boxes, scores, score_threshold):
    """
    Filter out bounding boxes based on a minimum score threshold.

    :param boxes: NumPy array of weighted averaged bounding box coordinates, shape (N, 4).
    :param scores: NumPy array of weighted scores for each bounding box, shape (N,).
    :param score_threshold: Minimum score threshold for a box to be retained.
    :return: A tuple containing two NumPy arrays:
             - The filtered list of bounding boxes.
             - The filtered list of scores corresponding to the retained boxes.
    """
    # Identify boxes with scores above the threshold
    high_score_indices = scores >= score_threshold

    # Filter out boxes and scores based on the threshold
    filtered_boxes = boxes[high_score_indices]
    filtered_scores = scores[high_score_indices]

    return filtered_boxes, filtered_scores


def calculate_x_center_precision(match_matrix, row_indices, col_indices, predicted_boxes, labeled_boxes):
    """
    Calculate the average distance in x direction between the centers of labeled and predicted boxes for matched pairs.

    :param match_matrix: A matrix indicating matches between predicted and labeled boxes.
                         Values < 1 indicate a match, values > 1 indicate no match.
    :param row_indices: Indices of rows in match_matrix that are part of a match.
    :param col_indices: Indices of columns in match_matrix that are part of a match.
    :param predicted_boxes: Numpy array of predicted bounding boxes, shape (N, 4).
    :param labeled_boxes: Numpy array of labeled (ground truth) bounding boxes, shape (M, 4).
    :return: The average distance in the x direction between the centers of matched labeled and predicted boxes.
    """
    # Filter matches with match_matrix value lower than 1 (indicating a match)
    matched_pairs = [(r, c) for r, c, v in zip(row_indices, col_indices, match_matrix[row_indices, col_indices]) if v < 1]

    if not matched_pairs:
        return None  # No matches found, return None to indicate no average distance can be calculated

    # Calculate the center x position for matched pairs and the difference
    x_center_diffs = []
    for r, c in matched_pairs:
        predicted_box = predicted_boxes[r]
        labeled_box = labeled_boxes[c]

        # Calculate the center x positions for both predicted and labeled boxes
        predicted_center_x = (predicted_box[0] + predicted_box[2]) / 2.0 + predicted_box[0]
        labeled_center_x = (labeled_box[0] + labeled_box[2]) / 2.0 + labeled_box[0]

        # Calculate the absolute difference in center x positions
        x_center_diff = abs(predicted_center_x - labeled_center_x)
        x_center_diffs.append(x_center_diff)

    # Calculate the average of center x differences
    avg_x_center_diff = np.mean(x_center_diffs)

    return avg_x_center_diff


def calculate_y_center_precision(match_matrix, row_indices, col_indices, predicted_boxes, labeled_boxes):
    """
    Calculate the average distance in y direction between the centers of labeled and predicted boxes for matched pairs.

    :param match_matrix: A matrix indicating matches between predicted and labeled boxes.
                         Values < 1 indicate a match, values > 1 indicate no match.
    :param row_indices: Indices of rows in match_matrix that are part of a match.
    :param col_indices: Indices of columns in match_matrix that are part of a match.
    :param predicted_boxes: Numpy array of predicted bounding boxes, shape (N, 4).
    :param labeled_boxes: Numpy array of labeled (ground truth) bounding boxes, shape (M, 4).
    :return: The average distance in the y direction between the centers of matched labeled and predicted boxes.
    """
    # Filter matches with match_matrix value lower than 1 (indicating a match)
    matched_pairs = [(r, c) for r, c, v in zip(row_indices, col_indices, match_matrix[row_indices, col_indices]) if v < 1]

    if not matched_pairs:
        return None  # No matches found, return None to indicate no average distance can be calculated

    # Calculate the center y position for matched pairs and the difference
    y_center_diffs = []
    for r, c in matched_pairs:
        predicted_box = predicted_boxes[r]
        labeled_box = labeled_boxes[c]

        # Calculate the center y positions for both predicted and labeled boxes
        predicted_center_y = (predicted_box[1] + predicted_box[3]) / 2.0 + predicted_box[1]
        labeled_center_y = (labeled_box[1] + labeled_box[3]) / 2.0 + labeled_box[1]

        # Calculate the absolute difference in center y positions
        y_center_diff = abs(predicted_center_y - labeled_center_y)
        y_center_diffs.append(y_center_diff)

    # Calculate the average of center y differences
    avg_y_center_diff = np.mean(y_center_diffs)

    return avg_y_center_diff


def calculate_center_distance(match_matrix, row_indices, col_indices, predicted_boxes, labeled_boxes):
    """
    Calculate the Euclidean distance between the centers of labeled and predicted boxes for matched pairs.

    :param match_matrix: A matrix indicating matches between predicted and labeled boxes.
                         Values < 1 indicate a match, values > 1 indicate no match.
    :param row_indices: Indices of rows in match_matrix that are part of a match.
    :param col_indices: Indices of columns in match_matrix that are part of a match.
    :param predicted_boxes: Numpy array of predicted bounding boxes, shape (N, 4).
    :param labeled_boxes: Numpy array of labeled (ground truth) bounding boxes, shape (M, 4).
    :return: The average Euclidean distance between the centers of matched labeled and predicted boxes.
    """
    # Filter matches with match_matrix value lower than 1 (indicating a match)
    matched_pairs = [(r, c) for r, c, v in zip(row_indices, col_indices, match_matrix[row_indices, col_indices]) if v < 1]

    if not matched_pairs:
        return None  # No matches found, return None to indicate no average distance can be calculated

    # Calculate the Euclidean distance between the centers of the boxes for matched pairs
    distances = []
    for r, c in matched_pairs:
        predicted_box = predicted_boxes[r]
        labeled_box = labeled_boxes[c]

        # Calculate the center positions for both predicted and labeled boxes
        predicted_center_x = (predicted_box[0] + predicted_box[2]) / 2.0
        predicted_center_y = (predicted_box[1] + predicted_box[3]) / 2.0
        labeled_center_x = (labeled_box[0] + labeled_box[2]) / 2.0
        labeled_center_y = (labeled_box[1] + labeled_box[3]) / 2.0

        # Calculate Euclidean distance between centers
        distance = np.sqrt((predicted_center_x - labeled_center_x)**2 + (predicted_center_y - labeled_center_y)**2)
        distances.append(distance)

    # Calculate the average of the Euclidean distances
    avg_distance = np.mean(distances)

    return avg_distance

