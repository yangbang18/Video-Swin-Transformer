import argparse
import torch
parser = argparse.ArgumentParser()
parser.add_argument('checkpoint', type=str)
parser.add_argument('save_path', type=str)
args = parser.parse_args()

checkpoint = torch.load(args.checkpoint)

out = {}
out['state_dict'] = {}

for k, v in checkpoint.items():
    if 'swin.backbone' in k:
        out['state_dict'][k.replace('swin.', '')] = v

torch.save(out, args.save_path)
