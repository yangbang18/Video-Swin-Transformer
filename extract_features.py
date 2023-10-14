import argparse
import os
import os.path as osp
import warnings

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.fileio.io import file_handlers
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from mmcv.runner.fp16_utils import wrap_fp16_model

from mmaction.datasets import build_dataloader, build_dataset
from mmaction.models import build_model
from mmaction.utils import register_module_hooks

from torch.utils.data import Dataset, DataLoader
from mmaction.datasets.pipelines import Compose, Resize, CenterCrop, Normalize, FormatShape, Collect, ToTensor
from PIL import Image
from tqdm import tqdm
import numpy as np
import h5py

# TODO import test functions from mmcv and delete them from mmaction2
try:
    from mmcv.engine import multi_gpu_test, single_gpu_test
except (ImportError, ModuleNotFoundError):
    warnings.warn(
        'DeprecationWarning: single_gpu_test, multi_gpu_test, '
        'collect_results_cpu, collect_results_gpu from mmaction2 will be '
        'deprecated. Please install mmcv through master branch.')
    from mmaction.apis import multi_gpu_test, single_gpu_test


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMAction2 test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--out',
        default=None,
        help='output result file in pkl/yaml/json format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g.,'
        ' "top_k_accuracy", "mean_class_accuracy" for video dataset')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        default={},
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        default={},
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        default={},
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. For example, '
        "'--cfg-options model.backbone.depth=18 model.backbone.with_cp=True'")
    parser.add_argument(
        '--average-clips',
        choices=['score', 'prob', None],
        default=None,
        help='average type when averaging test clips')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--onnx',
        action='store_true',
        help='Whether to test with onnx model or not')
    parser.add_argument(
        '--tensorrt',
        action='store_true',
        help='Whether to test with TensorRT engine or not')
    
    parser.add_argument('--video_root', type=str, default='/mnt/new_VC_data/MSRVTT/all_frames')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--img_fn_format', type=str, default='%05d.jpg')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--save_path', type=str, default='./msrvtt_VideoSiwn.hdf5')
    parser.add_argument('--dense', action='store_true')

    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both '
            'specified, --options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args


def turn_off_pretrained(cfg):
    # recursively find all pretrained in the model config,
    # and set them None to avoid redundant pretrain steps for testing
    if 'pretrained' in cfg:
        cfg.pretrained = None

    # recursively turn off pretrained value
    for sub_cfg in cfg.values():
        if isinstance(sub_cfg, dict):
            turn_off_pretrained(sub_cfg)


@torch.no_grad()
def inference_pytorch(args, cfg, distributed):
    """Get predictions by pytorch models."""
    if args.average_clips is not None:
        # You can set average_clips during testing, it will override the
        # original setting
        if cfg.model.get('test_cfg') is None and cfg.get('test_cfg') is None:
            cfg.model.setdefault('test_cfg',
                                 dict(average_clips=args.average_clips))
        else:
            if cfg.model.get('test_cfg') is not None:
                cfg.model.test_cfg.average_clips = args.average_clips
            else:
                cfg.test_cfg.average_clips = args.average_clips

    # remove redundant pretrain steps for testing
    turn_off_pretrained(cfg.model)

    # build the model and load checkpoint
    model = build_model(
        cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))

    if len(cfg.module_hooks) > 0:
        register_module_hooks(model, cfg.module_hooks)

    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')

    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)

    dataset = VideoDataset(args.video_root, limit=args.limit, img_fn_format=args.img_fn_format)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, drop_last=False)

    model = model.to('cuda')
    model.eval()
    
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    db = h5py.File(args.save_path, 'a')
    for imgs, vids in tqdm(loader):
        if all(vid in db for vid in vids):
            continue
        feats = model.extract_feat(imgs.to('cuda')) # (batch_size, dim, T, H, W)
        B, C = feats.shape[:2]
        if args.dense:
            feats = feats.view(B, C, -1) # (batch_size, dim, T * H * W)
        else:
            feats = feats.mean((-1, -2)) # spatio avearge pooling, 
        feats = feats.cpu().numpy()
        feats = np.transpose(feats, (0, 2, 1)) # (batch_size, T, dim)

        for vid, feat in zip(vids, feats):
            if vid not in db:
                db[vid] = feat
    db.close()


def get_uniform_ids_from_k_snippets(length, k, offset=0):
    uniform_ids = []
    bound = [int(i) for i in np.linspace(0, length, k+1)]
    for i in range(k):
        idx = (bound[i] + bound[i+1]) // 2
        uniform_ids.append(idx + offset)
    return uniform_ids


class VideoDataset(Dataset):
    def __init__(self, video_root, clip_len=64, limit=None, img_fn_format='%05d.jpg') -> None:
        super().__init__()
        self.video_root = video_root
        self.clip_len = clip_len
        
        if limit is None:
            limit = int(1e9)

        self.vids = [vid for vid in os.listdir(video_root) if ('video' in vid and int(vid[5:]) < limit)]
        self.vids = sorted(self.vids, key=lambda x: int(x[5:]))

        self.img_fn_format = img_fn_format
        
        self.preprocess = Compose([
            Resize(scale=(-1, 224)),
            CenterCrop(crop_size=(224, 224)),
            Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False),
            FormatShape(input_format='NCTHW'),
            Collect(keys=['imgs'], meta_keys=[]),
            ToTensor(keys=['imgs']),
        ])

    def __len__(self):
        return len(self.vids)
    
    def __getitem__(self, index):
        path = os.path.join(self.video_root, self.vids[index])
        n_total_frames = len(os.listdir(path))
        frames_ids = get_uniform_ids_from_k_snippets(n_total_frames, self.clip_len, offset=1)
        
        results = {}
        results['imgs'] = imgs = [
            np.array(Image.open(os.path.join(path, self.img_fn_format % fid))) for fid in frames_ids
        ]
        results['modality'] = 'RGB'
        results['num_clips'] = 1
        results['clip_len'] = self.clip_len
        results = self.preprocess(results)

        # CTHW
        return results['imgs'].squeeze(0), self.vids[index]


def main():
    args = parse_args()

    if args.tensorrt and args.onnx:
        raise ValueError(
            'Cannot set onnx mode and tensorrt mode at the same time.')

    cfg = Config.fromfile(args.config)

    cfg.merge_from_dict(args.cfg_options)

    # Load output_config from cfg
    output_config = cfg.get('output_config', {})
    if args.out:
        # Overwrite output_config from args.out
        output_config = Config._merge_a_into_b(
            dict(out=args.out), output_config)

    # Load eval_config from cfg
    eval_config = cfg.get('eval_config', {})
    if args.eval:
        # Overwrite eval_config from args.eval
        eval_config = Config._merge_a_into_b(
            dict(metrics=args.eval), eval_config)
    if args.eval_options:
        # Add options from args.eval_options
        eval_config = Config._merge_a_into_b(args.eval_options, eval_config)

    dataset_type = cfg.data.test.type

    # set cudnn benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.data.test.test_mode = True

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # The flag is used to register module's hooks
    cfg.setdefault('module_hooks', [])

    outputs = inference_pytorch(args, cfg, distributed)


if __name__ == '__main__':
    main()
