import os

import torch
import torch.utils.data
from PIL import Image
import json

from maskrcnn_benchmark.structures.bounding_box import BoxList


class DeepDriveDataset(torch.utils.data.Dataset):

    CLASSES = (
        "__background",
        "bike",
        "bus",
        "car",
        "motor",
        "person",
        "rider",
        "traffic light",
        "traffic sign",
        "train",
        "truck"
    )

    def __init__(self, data_dir, labels_dir, split, labels_dir_tss2, use_occluded=False, transforms=None):
        self.root = data_dir # 'bdd100k/'
        self.image_set = split
        self.keep_occluded = use_occluded
        self.transforms = transforms

        self._annopath = os.path.join(self.root, labels_dir)
        with open(self._annopath) as f:
            self.annotations = json.load(f)
        self._imgpath = os.path.join(self.root, "images/100k", self.image_set, "%s")

        if 'train' in labels_dir_tss2:
            _annopath_tss2_train = os.path.join(self.root, labels_dir_tss2)
            with open(_annopath_tss2_train) as f:
                    annotations_tss2_train = json.load(f)
            self.annotations += annotations_tss2_train
            self._imgpath_tss2 = os.path.join(self.root, "images/tss2", "70kImages", "%s")
        
        self.ids = [dic['name'] for dic in self.annotations]
        self.im_infos = [dic['im_info'] for dic in self.annotations]
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}

        cls = DeepDriveDataset.CLASSES
        self.class_to_ind = dict(zip(cls, range(len(cls))))

    def __getitem__(self, index):
        img_id = self.ids[index]
        if '.jpg' in img_id:
            img = Image.open(self._imgpath % img_id).convert("RGB")
        elif '.bmp' in img_id:
            img = Image.open(self._imgpath_tss2 % img_id).convert("RGB")

        target = self.get_groundtruth(index)
        target = target.clip_to_image(remove_empty=True)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target, index

    def __len__(self):
        return len(self.ids)

    def get_groundtruth(self, index):
        img_id = self.ids[index]
        anno = list(filter(
            lambda annotations: annotations['name'] == img_id, self.annotations))[0]
        anno = self._preprocess_annotation(anno)

        height, width = anno["im_info"]
        target = BoxList(anno["boxes"], (width, height), mode="xyxy")
        target.add_field("labels", anno["labels"])
        target.add_field("occluded", anno["occluded"])
        return target

    def _preprocess_annotation(self, anno):
        boxes = []
        gt_classes = []
        occluded_boxes = []
        TO_REMOVE = 1
        # Filter out segmentation annotations
        target = anno['labels']
        target = [label for label in target if "box2d" in label.keys()]
        for obj in target:
            occluded = obj['attributes']['occluded'] == 1
            # Use to filter occluded boxes
            # if not self.keep_occluded and occluded:
            #     continue
            name = obj['category']
            bb = obj['box2d']
            # Make pixel indexes 0-based
            # Refer to "https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/datasets/pascal_voc.py#L208-L211"
            box = [
                bb['x1'],
                bb['y1'],
                bb['x2'],
                bb['y2']
            ]
            bndbox = tuple(
                map(lambda x: x - TO_REMOVE, list(map(int, box)))
            )

            boxes.append(bndbox)
            gt_classes.append(self.class_to_ind[name])
            occluded_boxes.append(occluded)
        
        im_info = (anno['im_info']['height'], anno['im_info']['width'])

        res = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(gt_classes),
            "occluded": torch.tensor(occluded_boxes),
            "im_info": im_info
        }
        return res

    def get_img_info(self, index):
        img_id = self.ids[index]
        im_info = self.im_infos[index]
        return im_info

    def map_class_id_to_class_name(self, class_id):
        return DeepDriveDataset.CLASSES[class_id]
