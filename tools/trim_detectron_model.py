import os
import torch
import argparse
from maskrcnn_benchmark.utils.model_serialization import strip_prefix_if_present


parser = argparse.ArgumentParser(description="Trim RetinaNet weights and save in .pth format.")
parser.add_argument(
    "--pretrained_path",
    default="",
    help="path to RetinaNet pretrained weight(.pth)",
    type=str)
parser.add_argument(
    "--save_path",
    default="",
    help="path to save the converted model",
    type=str)


def removekey(d, listofkeys):
    for key in listofkeys:
        print('key: {} is removed'.format(key))
        d.pop(key)
    return d


args = parser.parse_args()
retinanet_pretrained_path = os.path.expanduser(args.pretrained_path)
print('RetinaNet path: {}'.format(retinanet_pretrained_path))

_d = torch.load(retinanet_pretrained_path, map_location=torch.device("cpu"))
_d['model'] = strip_prefix_if_present(_d['model'], prefix='module.')
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
