import argparse
import csv
import os

import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

import torchreid


def load_checkpoint_weights(model, checkpoint_path):
    checkpoint = torch.load(
        checkpoint_path, map_location='cpu', weights_only=False
    )
    state_dict = checkpoint.get('state_dict', checkpoint)
    cleaned = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            key = key[7:]
        cleaned[key] = value
    model.load_state_dict(cleaned, strict=True)
    return checkpoint


def preprocess_image(image_path, height, width, device):
    image = Image.open(image_path).convert('RGB')
    resized = image.resize((width, height), Image.Resampling.BILINEAR)
    array = np.asarray(resized).astype('float32') / 255.0
    mean = np.asarray([0.485, 0.456, 0.406], dtype='float32')
    std = np.asarray([0.229, 0.224, 0.225], dtype='float32')
    array = (array - mean) / std
    tensor = torch.from_numpy(array.transpose(2, 0, 1)).unsqueeze(0)
    return image, tensor.to(device)


def main():
    parser = argparse.ArgumentParser(
        description='Visualize OSNet feature activations and ChannelGate weights.'
    )
    parser.add_argument('--image', required=True, help='Path to one pedestrian image')
    parser.add_argument('--checkpoint', required=True, help='Path to best_mAP.pth')
    parser.add_argument('--output', default='results/OSNet_Market1501/visualizations')
    parser.add_argument('--height', type=int, default=256)
    parser.add_argument('--width', type=int, default=128)
    parser.add_argument('--num-classes', type=int, default=751)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torchreid.models.build_model(
        name='osnet_x1_0',
        num_classes=args.num_classes,
        loss='softmax',
        pretrained=False
    ).to(device)
    checkpoint = load_checkpoint_weights(model, args.checkpoint)
    model.eval()

    original_image, input_tensor = preprocess_image(
        args.image, args.height, args.width, device
    )

    feature_maps = {}
    gate_records = []
    handles = []

    def feature_hook(stage_name):
        def hook(module, inputs, output):
            feature_maps[stage_name] = output.detach().cpu()
        return hook

    def gate_hook(block_name, branch_index):
        def hook(module, inputs, output):
            gate_records.append({
                'block': block_name,
                'branch': branch_index,
                'weights': output.detach().cpu().flatten().numpy()
            })
        return hook

    for stage_name in ['conv2', 'conv3', 'conv4', 'conv5']:
        stage = getattr(model, stage_name)
        handles.append(stage.register_forward_hook(feature_hook(stage_name)))

    for module_name, module in model.named_modules():
        if module.__class__.__name__ == 'OSBlock':
            gate = module.gate
            activation = gate.gate_activation
            if activation is not None:
                branch_counter = {'value': 0}

                def make_pre_hook(name, counter):
                    def pre_hook(module, inputs):
                        counter['value'] += 1
                    return pre_hook

                def make_gate_hook(name, counter):
                    def hook(module, inputs, output):
                        gate_records.append({
                            'block': name,
                            'branch': counter['value'],
                            'weights': output.detach().cpu().flatten().numpy()
                        })
                    return hook

                counter = {'value': 0}

                def gate_pre(module, inputs, counter=counter):
                    counter['value'] += 1

                handles.append(activation.register_forward_pre_hook(gate_pre))
                handles.append(activation.register_forward_hook(
                    make_gate_hook(module_name, counter)
                ))

    with torch.no_grad():
        model(input_tensor)

    for handle in handles:
        handle.remove()

    os.makedirs(os.path.join(args.output, 'feature_maps'), exist_ok=True)
    os.makedirs(os.path.join(args.output, 'channel_attention'), exist_ok=True)
    os.makedirs(os.path.join(args.output, 'activation_overlays'), exist_ok=True)

    activation_map = None
    for stage_name in ['conv2', 'conv3', 'conv4', 'conv5']:
        feature_tensor = feature_maps[stage_name][0]
        stage_map = feature_tensor.mean(dim=0).numpy()
        stage_map = np.maximum(stage_map, 0)
        stage_map = stage_map - stage_map.min()
        if stage_map.max() > 0:
            stage_map = stage_map / stage_map.max()

        plt.figure(figsize=(5, 7))
        plt.imshow(stage_map, cmap='jet')
        plt.axis('off')
        plt.title('{} mean-channel activation'.format(stage_name))
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                args.output, 'feature_maps',
                '{}_feature_activation.png'.format(stage_name)
            ),
            dpi=200, bbox_inches='tight'
        )
        plt.close()
        if stage_name == 'conv5':
            activation_map = stage_map

    resized_activation = Image.fromarray(
        np.uint8(activation_map * 255)
    ).resize(original_image.size, Image.Resampling.BILINEAR)
    plt.figure(figsize=(6, 8))
    plt.imshow(original_image)
    plt.imshow(np.asarray(resized_activation), cmap='jet', alpha=0.45)
    plt.axis('off')
    plt.title('Feature activation overlay (not a class-specific Grad-CAM)')
    plt.tight_layout()
    plt.savefig(
        os.path.join(args.output, 'activation_overlays', 'activation_overlay.png'),
        dpi=200, bbox_inches='tight'
    )
    plt.close()

    csv_path = os.path.join(
        args.output, 'channel_attention', 'channel_gate_weights.csv'
    )
    with open(csv_path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['OSBlock', 'branch', 'channel', 'gate_weight'])
        for record in gate_records:
            for channel_index, weight in enumerate(record['weights']):
                writer.writerow([
                    record['block'], record['branch'], channel_index,
                    float(weight)
                ])

    # Plot mean gate weight per branch for every OSBlock.
    grouped = {}
    for record in gate_records:
        key = (record['block'], record['branch'])
        grouped[key] = float(np.mean(record['weights']))

    plot_path = os.path.join(
        args.output, 'channel_attention', 'channel_gate_branch_means.png'
    )
    labels = [
        '{} / call {}'.format(key[0], key[1])
        for key in grouped.keys()
    ]
    values = list(grouped.values())
    plt.figure(figsize=(12, 5))
    plt.bar(range(len(values)), values)
    plt.xticks(range(len(labels)), labels, rotation=75, ha='right', fontsize=7)
    plt.ylabel('Mean sigmoid gate weight')
    plt.title('OSNet ChannelGate weights by OSBlock branch call')
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()

    print('Loaded checkpoint epoch: {}'.format(checkpoint.get('epoch', 'unknown')))
    print('Saved feature map, activation overlay and ChannelGate weights to: {}'.format(
        os.path.abspath(args.output)
    ))


if __name__ == '__main__':
    main()
