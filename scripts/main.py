import sys
import time
import os.path as osp
import os
import csv
import statistics
import random
import numpy as np
import argparse
import torch
import torch.nn as nn

import torchreid
from torchreid.utils import (
    Logger, check_isfile, set_random_seed, collect_env_info,
    resume_from_checkpoint, load_pretrained_weights, compute_model_complexity
)

from default_config import (
    imagedata_kwargs, optimizer_kwargs, videodata_kwargs, engine_run_kwargs,
    get_default_config, lr_scheduler_kwargs
)
from plot_results import generate_plots


def build_datamanager(cfg):
    if cfg.data.type == 'image':
        return torchreid.data.ImageDataManager(**imagedata_kwargs(cfg))
    else:
        return torchreid.data.VideoDataManager(**videodata_kwargs(cfg))


def build_engine(cfg, datamanager, model, optimizer, scheduler):
    if cfg.data.type == 'image':
        if cfg.loss.name == 'softmax':
            engine = torchreid.engine.ImageSoftmaxEngine(
                datamanager,
                model,
                optimizer=optimizer,
                scheduler=scheduler,
                use_gpu=cfg.use_gpu,
                label_smooth=cfg.loss.softmax.label_smooth
            )

        else:
            engine = torchreid.engine.ImageTripletEngine(
                datamanager,
                model,
                optimizer=optimizer,
                margin=cfg.loss.triplet.margin,
                weight_t=cfg.loss.triplet.weight_t,
                weight_x=cfg.loss.triplet.weight_x,
                scheduler=scheduler,
                use_gpu=cfg.use_gpu,
                label_smooth=cfg.loss.softmax.label_smooth
            )

    else:
        if cfg.loss.name == 'softmax':
            engine = torchreid.engine.VideoSoftmaxEngine(
                datamanager,
                model,
                optimizer=optimizer,
                scheduler=scheduler,
                use_gpu=cfg.use_gpu,
                label_smooth=cfg.loss.softmax.label_smooth,
                pooling_method=cfg.video.pooling_method
            )

        else:
            engine = torchreid.engine.VideoTripletEngine(
                datamanager,
                model,
                optimizer=optimizer,
                margin=cfg.loss.triplet.margin,
                weight_t=cfg.loss.triplet.weight_t,
                weight_x=cfg.loss.triplet.weight_x,
                scheduler=scheduler,
                use_gpu=cfg.use_gpu,
                label_smooth=cfg.loss.softmax.label_smooth
            )

    return engine


def reset_config(cfg, args):
    if args.root:
        cfg.data.root = args.root
    if args.sources:
        cfg.data.sources = args.sources
    if args.targets:
        cfg.data.targets = args.targets
    if args.transforms:
        cfg.data.transforms = args.transforms


def check_cfg(cfg):
    if cfg.loss.name == 'triplet' and cfg.loss.triplet.weight_x == 0:
        assert cfg.train.fixbase_epoch == 0, \
            'The output of classifier is not included in the computational graph'



def benchmark_model(model, cfg, inference_params, macs):
    """Measure batch-1 inference latency on the active device."""
    model.eval()
    device = next(model.parameters()).device
    sample = torch.randn(
        1, 3, cfg.data.height, cfg.data.width, device=device
    )
    with torch.no_grad():
        for _ in range(20):
            model(sample)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        timings = []
        for _ in range(100):
            start = time.perf_counter()
            model(sample)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            timings.append((time.perf_counter() - start) * 1000.0)

    mean_ms = statistics.mean(timings)
    median_ms = statistics.median(timings)
    total_params = sum(parameter.numel() for parameter in model.parameters())
    row = {
        'architecture': cfg.model.name,
        'input_shape': '1x3x{}x{}'.format(cfg.data.height, cfg.data.width),
        'inference_parameters': int(inference_params),
        'all_parameters_including_classifier': int(total_params),
        'fp32_weight_size_mb_all_parameters': round(total_params * 4 / (1024 ** 2), 3),
        'MACs_estimated': int(macs),
        'FLOPs_estimated_2_per_MAC': int(macs * 2),
        'flops_scope': 'Conv2d and Linear layers in evaluation feature-extraction path',
        'device': str(device),
        'batch1_latency_mean_ms': round(mean_ms, 4),
        'batch1_latency_median_ms': round(median_ms, 4),
        'batch1_throughput_images_per_second': round(1000.0 / mean_ms, 3)
    }
    output_dir = cfg.data.save_dir
    os.makedirs(output_dir, exist_ok=True)
    csv_path = osp.join(output_dir, 'benchmarks', 'model_complexity.csv')
    os.makedirs(osp.dirname(csv_path), exist_ok=True)
    with open(csv_path, 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    print('Benchmark CSV saved to "{}"'.format(csv_path))
    print(
        'Batch-1 inference latency: {:.3f} ms (mean), {:.3f} images/s'.format(
            mean_ms, 1000.0 / mean_ms
        )
    )



def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--config-file', type=str, default='', help='path to config file'
    )
    parser.add_argument(
        '-s',
        '--sources',
        type=str,
        nargs='+',
        help='source datasets (delimited by space)'
    )
    parser.add_argument(
        '-t',
        '--targets',
        type=str,
        nargs='+',
        help='target datasets (delimited by space)'
    )
    parser.add_argument(
        '--transforms', type=str, nargs='+', help='data augmentation'
    )
    parser.add_argument(
        '--root', type=str, default='', help='path to data root'
    )
    parser.add_argument(
        'opts',
        default=None,
        nargs=argparse.REMAINDER,
        help='Modify config options using the command-line'
    )
    args = parser.parse_args()

    cfg = get_default_config()
    cfg.use_gpu = torch.cuda.is_available()
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    reset_config(cfg, args)
    cfg.merge_from_list(args.opts)
    set_random_seed(cfg.train.seed)
    check_cfg(cfg)

    os.makedirs(cfg.data.save_dir, exist_ok=True)
    log_name = 'test.log' if cfg.test.evaluate else 'train.log'
    log_name += time.strftime('-%Y-%m-%d-%H-%M-%S')
    sys.stdout = Logger(osp.join(cfg.data.save_dir, log_name))

    print('Show configuration\n{}\n'.format(cfg))
    print('Collecting env info ...')
    print('** System info **\n{}\n'.format(collect_env_info()))

    if cfg.use_gpu:
        torch.backends.cudnn.benchmark = True

    datamanager = build_datamanager(cfg)

    print('Building model: {}'.format(cfg.model.name))
    model = torchreid.models.build_model(
        name=cfg.model.name,
        num_classes=datamanager.num_train_pids,
        loss=cfg.loss.name,
        pretrained=cfg.model.pretrained,
        use_gpu=cfg.use_gpu
    )
    num_params, flops = compute_model_complexity(
        model, (1, 3, cfg.data.height, cfg.data.width)
    )
    print(
        'Model complexity (inference path): params={:,}, MACs~{:,}, FLOPs~{:,}'.format(
            num_params, flops, flops * 2
        )
    )

    if cfg.model.load_weights and check_isfile(cfg.model.load_weights):
        load_pretrained_weights(model, cfg.model.load_weights)

    if cfg.use_gpu:
        model = nn.DataParallel(model).cuda()

    benchmark_model(model, cfg, num_params, flops)

    optimizer = torchreid.optim.build_optimizer(model, **optimizer_kwargs(cfg))
    scheduler = torchreid.optim.build_lr_scheduler(
        optimizer, **lr_scheduler_kwargs(cfg)
    )

    if cfg.model.resume and check_isfile(cfg.model.resume):
        cfg.train.start_epoch = resume_from_checkpoint(
            cfg.model.resume, model, optimizer=optimizer, scheduler=scheduler
        )

    print(
        'Building {}-engine for {}-reid'.format(cfg.loss.name, cfg.data.type)
    )
    engine = build_engine(cfg, datamanager, model, optimizer, scheduler)
    if cfg.model.resume and check_isfile(cfg.model.resume):
        from torchreid.utils import load_checkpoint
        resume_state = load_checkpoint(cfg.model.resume)
        engine.best_mAP = resume_state.get('best_mAP', resume_state.get('mAP', 0.0)) or 0.0
        rng_state = resume_state.get('rng_state')
        if rng_state:
            random.setstate(rng_state['python'])
            np.random.set_state(rng_state['numpy'])
            torch.set_rng_state(rng_state['torch'])
            if torch.cuda.is_available() and rng_state.get('cuda') is not None:
                torch.cuda.set_rng_state_all(rng_state['cuda'])
    engine.run(**engine_run_kwargs(cfg))
    generate_plots(cfg.data.save_dir)


if __name__ == '__main__':
    main()
