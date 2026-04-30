# THIS FILE SHOULD NOT BE CHANGED

# Description:
# This file should be used for computing scores for each of the project topics
# Usage: python evaluation.py <project_code> <path_2_output_predictions>

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from skimage import io
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


def list_all_files(folder_path):
    return sorted(
        [f for f in Path(folder_path).rglob("*") if f.is_file()],
        key=lambda x: x.name.lower(),
    )


############## DET ##########################################################################
def iou_bboxs(bbox1, bbox2):
    # input bboxes
    b1_x_min, b1_y_min, b1_x_max, b1_y_max = bbox1
    b2_x_min, b2_y_min, b2_x_max, b2_y_max = bbox2
    # intersection area
    intersection_width = min(b1_x_max, b2_x_max) - max(b1_x_min, b2_x_min)
    intersection_height = min(b1_y_max, b2_y_max) - max(b1_y_min, b2_y_min)
    # if no intersection
    if intersection_width <= 0 or intersection_height <= 0:
        return 0

    intersection_area = intersection_width * intersection_height

    # union area
    box1_area = (b1_x_max - b1_x_min) * (b1_y_max - b1_y_min)
    box2_area = (b2_x_max - b2_x_min) * (b2_y_max - b2_y_min)

    union_area = box1_area + box2_area - intersection_area

    # calculate IoU
    iou = intersection_area / union_area
    return iou


def map_ref_pred(
    ref_boxes, pred_boxes, mapping, reverse_mapping, all_confs, epsilon=0.5
):
    """
    Match the reference boxes to the predictions and vice versa.
    Boxes are only matched if they have IoU over epsilon.
    """
    for ref_idx, ref_box in ref_boxes.iterrows():
        ref_name = f"{ref_box['filename']}_{ref_box['object_id']}"
        ref_bbox = [ref_box["xmin"], ref_box["ymin"], ref_box["xmax"], ref_box["ymax"]]
        mapping[ref_name] = []
        for pred_idx, pred_box in pred_boxes.iterrows():
            pred_name = f"{pred_box['filename']}_{pred_idx}"
            if pred_name not in reverse_mapping.keys():
                reverse_mapping[pred_name] = []
            pred_bbox = [
                pred_box["xmin"],
                pred_box["ymin"],
                pred_box["xmax"],
                pred_box["ymax"],
            ]
            conf = pred_box["confidence"]
            if pred_name not in all_confs.keys():
                all_confs[pred_name] = conf
            iou_score = iou_bboxs(ref_bbox, pred_bbox)
            if iou_score >= epsilon:
                mapping[ref_name].append((pred_name, iou_score, conf, iou_score))
                reverse_mapping[pred_name].append(ref_name)
    # Don't return anything, we update both mappings in place


def process_mappings(mapping, reverse_mapping, all_confs):
    """
    Calcualte thresholds for FPs and FNs for the prec-recall curve.
    A reference detection is a FN at a threshold higher than the confidence of its highest confidence match.
    A detection is a FP at a threshold where it is no longer the highest-IoU detection for at least one reference box.
    """
    FN_thresholds = []
    for k in mapping.keys():
        matches = mapping[k]
        if len(matches) == 0:
            FN_thresh = -1.0  # If there is no match, it's always a FN
        else:
            FN_thresh = np.max(
                [match[2] for match in matches]
            )  # As soon as the first match appears, it is no longer a FN
        FN_thresholds.append(FN_thresh)
    FN_thresholds.append(-1.0)  # Add an extra threshold that will never be reached

    FN_thresholds = np.array(FN_thresholds)
    FN_thresholds = FN_thresholds[np.argsort(-FN_thresholds)]

    FP_thresholds = []
    for k in reverse_mapping.keys():
        FP_thresh = all_confs[k]  # If there's no match, it's always a FP
        matches = reverse_mapping[k]
        if len(matches) != 0:
            for match in matches:
                ref_matches = mapping[match]
                ordering = list(
                    np.argsort([-rm[1] for rm in ref_matches])
                )  # Sort the matches by descending IoU
                curr_elem = [rm[0] for rm in ref_matches].index(k)
                rank = ordering.index(
                    curr_elem
                )  # Where is the current element in the ordering?
                if rank == 0:
                    FP_thresh = (
                        -1.0
                    )  # If this is the best match for any ref box, then it's never a FP
                    break
                else:
                    all_match_confs = [
                        ref_matches[ordering[ridx]][2]
                        for ridx in np.arange(rank - 1, -1, -1)
                    ]  # Find the confidences of all the matches with higher IoU
                    min_match_conf = np.min(all_match_confs)
                    if (
                        min_match_conf > all_confs[k]
                    ):  # There is a better match that appears before this one. For this ref box, it's a FP.
                        pass
                    else:
                        FP_thresh = min(
                            FP_thresh, min_match_conf
                        )  # This match is the first until another one appears later. What is the lowest confidence for which this is still not a FP?
        FP_thresholds.append(FP_thresh)
    FP_thresholds.append(-1.0)  # Add an extra threshold that will never be reached
    FP_thresholds = np.array(FP_thresholds)
    FP_thresholds = FP_thresholds[np.argsort(-FP_thresholds)]
    return FN_thresholds, FP_thresholds


def compute_avg_precision(project_code, folder_ref, folder_pred):
    ref_files = list_all_files(folder_ref)
    pred_files = list_all_files(folder_pred)

    mapping, reverse_mapping = {}, {}
    all_confs = {}
    confidences = []
    for ref_file, pred_file in zip(ref_files, pred_files):
        pred_boxes = pd.read_csv(pred_file)
        ref_boxes = pd.read_csv(ref_file)
        for pred_idx, pred_box in pred_boxes.iterrows():
            confidences.append(pred_box["confidence"])
        map_ref_pred(ref_boxes, pred_boxes, mapping, reverse_mapping, all_confs, 0.5)
    FN_thresholds, FP_thresholds = process_mappings(mapping, reverse_mapping, all_confs)
    ref_num = len(mapping.keys())
    confidences.append(-1.0)  # Add an extra threshold that will never be reached
    confidences = np.array(confidences)
    confidences = confidences[
        np.argsort(-confidences)
    ]  # Sort the confidences in descending order
    pred_thresholds = np.flip(np.unique(confidences[:-1]))
    FP_num, TP_num, pred_num = 0, 0, 0
    precisions, recalls = [1], [0]
    for thresh in pred_thresholds:
        while thresh <= confidences[pred_num]:
            pred_num += 1
        while thresh <= FP_thresholds[FP_num]:
            FP_num += 1
        while thresh <= FN_thresholds[TP_num]:
            TP_num += 1
        prec = (pred_num - FP_num) / pred_num
        rec = TP_num / ref_num
        precisions.append(prec)
        recalls.append(rec)
    precisions.extend([0, 0])
    precisions = np.array(precisions)
    recalls.extend([recalls[-1], 1.0])
    recalls = np.array(recalls)
    # AUC = np.sum((recalls[1:]-recalls[:-1])*precisions[1:])

    interp_prec = np.flip(np.maximum.accumulate(np.flip(precisions)))
    interp_rec = recalls
    AUC2 = np.sum((interp_rec[1:] - interp_rec[:-1]) * interp_prec[1:])

    return AUC2


###########  SEG ###################################################################
label_dict = {
    "void": (0, 0, 0),
    "flat": (128, 64, 128),
    "construction": (70, 70, 70),
    "object": (153, 153, 153),
    "nature": (107, 142, 35),
    "sky": (70, 130, 180),
    "human": (220, 20, 60),
    "vehicle": (0, 0, 142),
}


def labmask_to_onehot_pixel_counts(mask, label_dict):
    # prepare empty list
    onehot_labels = []
    # class_pixel_counts = {label: 0 for label in label_dict}
    # SOLUTION
    # iterate over all classes defined in the label_dict
    for key in label_dict:
        rgb = label_dict[key]
        # prepare empty binary mask
        class_mask = np.zeros((mask.shape[:2]), dtype=bool)

        # make binary mask of positions given by all rgb values
        # corresponding to a specific class
        label_pos = np.equal(mask, rgb)
        # add label positions to binary mask
        class_mask = np.logical_or(class_mask, np.all(label_pos, axis=-1))

        # add binary mask of the respective class to the list
        onehot_labels.append(class_mask)
        # class_pixel_counts[key] = np.sum(class_mask)
    # END OF SOLUTION
    # return class_pixel_counts
    return onehot_labels


def labmask_to_onehot_pixel_counts_alternative(mask, label_dict):
    # prepare empty list
    onehot_labels = []
    # class_pixel_counts = {label: 0 for label in label_dict}
    # SOLUTION
    # iterate over all classes defined in the label_dict
    for key in label_dict:
        rgb = label_dict[key]
        # prepare empty binary mask

        # make binary mask of positions given by all rgb values
        # corresponding to a specific class
        label_pos = np.equal(mask, rgb)
        # add label positions to binary mask

        # add binary mask of the respective class to the list
        onehot_labels.append(np.all(label_pos, axis=-1))
        # class_pixel_counts[key] = np.sum(class_mask)
    # END OF SOLUTION
    # return class_pixel_counts
    return onehot_labels


def multiply_by_weights(class_wise_scores):
    # weights for the whole dataset (including the secret)
    weights = [0.105, 0.387, 0.217, 0.018, 0.152, 0.035, 0.012, 0.073]  # 8 classes

    # multiply
    result = [a * b for a, b in zip(class_wise_scores, weights)]

    # calculate the mean
    mean = sum(result)
    return mean


def classwise_iou_mask_4_img_pair(img_ref, img_pre):
    onehot_ref = labmask_to_onehot_pixel_counts(img_ref, label_dict)
    onehot_pre = labmask_to_onehot_pixel_counts(img_pre, label_dict)

    classwise_scores = []
    for idx, class_key in enumerate(label_dict):
        mask_ref = onehot_ref[idx]
        mask_pre = onehot_pre[idx]

        intersection = np.logical_and(mask_ref, mask_pre)
        union = np.logical_or(mask_ref, mask_pre)
        if np.sum(union) != 0:
            iou = np.sum(intersection) / np.sum(union)
        else:
            iou = None
        classwise_scores.append(iou)
    return classwise_scores


def compute_avg_iou_mask(project_code, folder_ref, folder_pre):
    # reading the images
    ref_files = list_all_files(folder_ref)
    pre_files = list_all_files(folder_pre)

    score_global = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    n_class_img = [0, 0, 0, 0, 0, 0, 0, 0]
    for ref_file, pre_file in zip(ref_files, pre_files):
        ref_img = io.imread(ref_file)
        pre_img = io.imread(pre_file)

        score_class_wise_pair = classwise_iou_mask_4_img_pair(ref_img, pre_img)
        score_global = [
            a + b if a is not None else b
            for a, b in zip(score_class_wise_pair, score_global)
        ]
        n_class_img = [
            b + 1 if a is not None else b
            for a, b in zip(score_class_wise_pair, n_class_img)
        ]

    score_global = [a / n_cl for a, n_cl in zip(score_global, n_class_img)]
    print(f"class-wise scores = {score_global}")
    final_score = multiply_by_weights(score_global)
    return final_score


###########  COL, SUP  #####################################################################
def dssim(img_ref, img_pred):
    ssim_none = ssim(img_ref, img_pred, channel_axis=2, data_range=255)
    dssim_score = 1 - ssim_none
    return dssim_score


def compute_simple_score_images(project_code, folder_ref, folder_pre):
    # reading the images
    ref_files = list_all_files(folder_ref)
    pre_files = list_all_files(folder_pre)

    scores = []
    for ref_file, pre_file in zip(ref_files, pre_files):
        # Open images
        ref_img = io.imread(ref_file)
        pre_img = io.imread(pre_file)

        if project_code == "COL":
            score_pair = dssim(ref_img, pre_img)
        elif project_code == "SUP":
            score_pair = psnr(ref_img, pre_img)
        # print(f'Img {ref_file}: score pair = {score_pair}')
        scores.append(score_pair)
    return sum(scores) / len(scores)


# #### code below should not be changed ############################################################################
def get_arguments():
    if len(sys.argv) != 4:
        print(
            "Usage: python evaluation.py <project_code> <path_2_reference> <path_2_output_predictions>"
        )
        sys.exit(1)

    try:
        project_code = sys.argv[1]
        path_2_reference = sys.argv[2]
        path_2_output_predictions = sys.argv[3]
    except Exception as e:
        print(e)
        sys.exit(1)
    return project_code, path_2_reference, path_2_output_predictions


if __name__ == "__main__":
    project_type, path_2_ground_truth, path_2_predictions = get_arguments()
    if project_type == "DET":
        score = compute_avg_precision(
            project_type, path_2_ground_truth, path_2_predictions
        )
    elif project_type == "SEG":
        score = compute_avg_iou_mask(
            project_type, path_2_ground_truth, path_2_predictions
        )
    elif project_type == "COL" or project_type == "SUP":
        score = compute_simple_score_images(
            project_type, path_2_ground_truth, path_2_predictions
        )

    print(f"Final score for {project_type} project score= {score}")
