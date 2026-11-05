# OSNet-x1.0 Market-1501 experiment

This project retains the Torchreid OSNet-x1.0 architecture and the ICCV 2019
from-scratch training settings in
`configs/im_osnet_x1_0_market1501_scratch.yaml`.

## Paper-aligned training settings

- OSNet-x1.0, initialized from scratch (`pretrained: False`)
- Market-1501 training identities used as classification classes
- 256 x 128 input resolution
- batch size 64
- SGD, initial learning rate 0.065, momentum 0.9
- weight decay 0.0005
- 350 epochs
- learning-rate milestones 150, 225, 300 with gamma 0.1
- cross-entropy with label smoothing
- random horizontal flip and random erasing

The paper does not specify every data-loader implementation detail. The
Torchreid `RandomSampler` default is made explicit in this configuration; it
should not be described as a paper-mandated sampler.

## Run training

Run from the repository root. The `--root` argument must point to the parent
directory that contains the `market1501` dataset directory expected by
Torchreid.

```bash
python scripts/main.py \
  --config-file configs/im_osnet_x1_0_market1501_scratch.yaml \
  --root /path/to/dataset-root
```

For Colab, install the repository after selecting a compatible PyTorch runtime:

```bash
pip install -e . --no-build-isolation
```

Then mount Drive and pass the appropriate Drive dataset root using `--root`.
Do not upload the dataset, checkpoints, or generated results to the Git repo.

## Generated outputs

All experiment outputs are written to `results/OSNet_Market1501/`:

- `logs/training_metrics.csv`: epoch loss, classification accuracy, learning
  rate used during that epoch, epoch duration, average batch/data time.
- `logs/evaluation_metrics.csv`: Rank-1/5/10/20 and mAP at each evaluation.
- `logs/experiment_summary.csv`: total wall-clock run time and final/best metrics.
- `logs/tensorboard/`: TensorBoard event files.
- `checkpoints/latest.pth`: latest periodic checkpoint.
- `checkpoints/best_mAP.pth`: checkpoint with best evaluated mAP.
- `checkpoints/epoch_XXX.pth`: periodic checkpoint copies (every evaluation).
- `benchmarks/model_complexity.csv`: inference parameters, estimated MACs,
  estimated FLOPs, theoretical FP32 weight size, and batch-1 latency/throughput.
- `plots/`: loss, classification-accuracy and retrieval-metric plots.

The complexity counter counts Conv2d and Linear operations on the evaluation
feature-extraction path. It treats one multiply-accumulate as one MAC; the
reported conventional FLOPs estimate is 2 x MACs. Pooling, normalization and
activation operations are not included in that estimate. Latency is measured
on the device used for the run after warm-up, with batch size one.

## Resume training

Set `model.resume` to a checkpoint path through the config override:

```bash
python scripts/main.py \
  --config-file configs/im_osnet_x1_0_market1501_scratch.yaml \
  --root /path/to/dataset-root \
  model.resume results/OSNet_Market1501/checkpoints/latest.pth
```

The checkpoint contains model, optimizer, scheduler, epoch, best mAP and random
number-generator states. Resume from `latest.pth`, not `best_mAP.pth`, when the
goal is to continue the most recent training trajectory.

## Feature and ChannelGate visualizations

After training, choose one pedestrian image (for example, a Market-1501 query
image) and run:

```bash
python scripts/visualize_osnet.py \
  --image /path/to/pedestrian.jpg \
  --checkpoint results/OSNet_Market1501/checkpoints/best_mAP.pth
```

This writes stage feature activation maps, a final activation overlay, and
per-channel ChannelGate sigmoid weights. The overlay is a mean-channel feature
activation visualization, not Grad-CAM and not a class-specific explanation.
ChannelGate values are channel-wise gates, not spatial attention maps.

## Important validation before the full run

Before committing to all 350 epochs, perform a short Colab smoke test by
temporarily overriding the epoch count and evaluation frequency on the command
line, then verify that CSV files, plots and checkpoints are created. Restore
the full 350-epoch configuration for the real experiment. Confirm batch size
64 fits the actual T4 session with the real data loader before proceeding.
