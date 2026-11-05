import csv
import os
import argparse
import matplotlib.pyplot as plt


def read_rows(path):
    if not os.path.isfile(path):
        return []
    with open(path, 'r', newline='') as csv_file:
        return list(csv.DictReader(csv_file))


def generate_plots(results_dir):
    logs_dir = os.path.join(results_dir, 'logs')
    plots_dir = os.path.join(results_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    train_rows = read_rows(os.path.join(logs_dir, 'training_metrics.csv'))
    eval_rows = read_rows(os.path.join(logs_dir, 'evaluation_metrics.csv'))

    if train_rows:
        epochs = [int(row['epoch']) for row in train_rows]
        losses = [float(row['loss']) for row in train_rows]
        accuracies = [
            float(row['classification_accuracy_percent'])
            for row in train_rows
        ]

        plt.figure(figsize=(8, 5))
        plt.plot(epochs, losses)
        plt.xlabel('Epoch')
        plt.ylabel('Cross-entropy loss')
        plt.title('OSNet training loss')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'loss_curve.png'), dpi=200)
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.plot(epochs, accuracies)
        plt.xlabel('Epoch')
        plt.ylabel('Classification accuracy (%)')
        plt.title('OSNet training classification accuracy')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'accuracy_curve.png'), dpi=200)
        plt.close()

    if eval_rows:
        selected = [
            row for row in eval_rows
            if row.get('dataset', '').lower() == 'market1501'
        ]
        if not selected:
            selected = eval_rows
        eval_epochs = [int(row['epoch']) for row in selected]

        plt.figure(figsize=(9, 5))
        for metric in ['Rank-1', 'Rank-5', 'Rank-10', 'Rank-20', 'mAP']:
            values = [
                float(row[metric]) * 100.0
                for row in selected if row.get(metric, '') != ''
            ]
            if len(values) == len(eval_epochs):
                plt.plot(eval_epochs, values, marker='o', label=metric)
        plt.xlabel('Epoch')
        plt.ylabel('Retrieval performance (%)')
        plt.title('OSNet Market-1501 retrieval metrics')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            os.path.join(plots_dir, 'retrieval_metrics.png'), dpi=200
        )
        plt.close()

    print('Plots saved in: {}'.format(os.path.abspath(plots_dir)))


def main():
    parser = argparse.ArgumentParser(
        description='Create plots from OSNet experiment CSV logs.'
    )
    parser.add_argument(
        '--results', default='results/OSNet_Market1501',
        help='Experiment output directory'
    )
    args = parser.parse_args()
    generate_plots(args.results)


if __name__ == '__main__':
    main()
