import os
import torch
import argparse
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.utils.c2_model_loading import load_c2_format


parser = argparse.ArgumentParser(description="Trim Detectron weights and save in PyTorch format.")
parser.add_argument(
    "--pretrained_path",
    default="",
    help="path to detectron pretrained weight(.pkl)",
    type=str)
parser.add_argument(
    "--save_path",
    default="",
    help="path to save the converted model",
    type=str)
parser.add_argument(
    "--cfg",
    default="",
    help="path to config file",
    type=str)


def removekey(d, listofkeys):
    r = dict(d)
    for key in listofkeys:
        print('key: {} is removed'.format(key))
        r.pop(key)
    return r


args = parser.parse_args()
DETECTRON_PATH = os.path.expanduser(args.pretrained_path)
print('Detectron path: {}'.format(DETECTRON_PATH))

# cfg.merge_from_file(args.cfg)
# _d = load_c2_format(cfg, DETECTRON_PATH)
_d = torch.load(DETECTRON_PATH)
newdict = _d

keys_to_be_removed = [
    'rpn.head.cls_logits.bias', 
    'rpn.head.cls_logits.weight', 
    'rpn.head.bbox_pred.bias', 
    'rpn.head.bbox_pred.weight',] # retinanet_X-101-32x8d-FPN_1x_model
newdict['model'] = removekey(_d['model'],
                             keys_to_be_removed)
torch.save(newdict, args.save_path)
print('Saved to {}.'.format(args.save_path))
