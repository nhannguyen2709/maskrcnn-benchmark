# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import argparse
import cv2

from maskrcnn_benchmark.config import cfg
from predictor import DeepDriveDemo

import numpy as np
import os
import pickle
import time
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Webcam Demo")
    parser.add_argument(
        "--config-file",
        default="../configs/caffe2/e2e_mask_rcnn_R_50_FPN_1x_caffe2.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--img-file",
        metavar="FILE",
        help="path to image file",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for the prediction to be shown",
    )
    parser.add_argument(
        "--min-image-size",
        type=int,
        default=224,
        help="Smallest size of the image to feed to the model. "
            "Model was trained with 800, which gives best results",
    )
    parser.add_argument(
        "--show-mask-heatmaps",
        dest="show_mask_heatmaps",
        help="Show a heatmap probability for the top masks-per-dim masks",
        action="store_true",
    )
    parser.add_argument(
        "--masks-per-dim",
        type=int,
        default=2,
        help="Number of heatmaps per dimension to show",
    )
    parser.add_argument(
        "opts",
        help="Modify model config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    # load config from file and command-line arguments
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    # prepare object that handles inference plus adds predictions on top of image
#    deepdrive_demo = DeepDriveDemo(
#        cfg,
#        confidence_threshold=args.confidence_threshold,
#        show_mask_heatmaps=args.show_mask_heatmaps,
#        masks_per_dim=args.masks_per_dim,
#        min_image_size=args.min_image_size,
#    )

#    start_time = time.time()
#    img = cv2.imread(args.img_file)
#    composite = deepdrive_demo.run_on_opencv_image(img)
#    print("Time: {:.2f} s / img".format(time.time() - start_time))
#    cv2.imwrite('demo_' + args.img_file, composite)

    deepdrive_demo = DeepDriveDemo(
        cfg,
        confidence_threshold=args.confidence_threshold,
        show_mask_heatmaps=args.show_mask_heatmaps,
        masks_per_dim=args.masks_per_dim,
        min_image_size=640,
    )
    # data_root = 'datasets/bdd100k/images/100k/train/'
    # save_dir = data_root.replace("train", "train_teacher_pkl")
    data_root = 'datasets/bdd100k/images/100k/val/'
    save_dir = data_root.replace("val", "val_teacher_pkl")

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    for img_file in tqdm(os.listdir(data_root)):
        img_path = os.path.join(data_root, img_file)
        img = cv2.imread(img_path)
        box_cls, box_regression, boxes = deepdrive_demo._dump_box_cls_box_reg(img)
        boxes = [o.to(deepdrive_demo.cpu_device) for o in boxes]
        boxes = boxes[0]
        # height, width = img.shape[:-1]
        # boxes = boxes.resize((width, height))
        boxes = deepdrive_demo.select_top_predictions(boxes)
        # overlay_img = deepdrive_demo.overlay_boxes(img, boxes)
        # cv2.imwrite("demo" + img_file, overlay_img)
        bboxes = boxes.bbox.numpy()
        labels = boxes.get_field("labels").numpy()
        box_cls = [b.cpu().numpy() for b in box_cls]
        # box_regression = [b.cpu().numpy() for b in box_regression]
        save_dict = {}
        # save_dict["box_cls"] = box_cls
        # save_dict["box_reg"] = box_regression
        save_dict["box_cls"] = [b.astype(dtype=np.float16) for b in box_cls[1:]]
        save_dict["bboxes"] = bboxes.astype(dtype=np.float16)
        save_dict["labels"] = labels.astype(dtype=np.float16)

        save_path = os.path.join(save_dir, img_file.replace(".jpg", ".pkl"))
        with open(save_path, "wb") as f:
            pickle.dump(save_dict, f)

if __name__ == "__main__":
    main()
