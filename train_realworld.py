#!/usr/bin/env python3
"""
Real-world heterophilic benchmark experiments (Table 3 in paper).

Supported datasets (from Platonov et al., 2023 + Lim et al., 2021):
  roman_empire, amazon_ratings, minesweeper, tolokers, questions, penn94

Usage:
    python train_realworld.py --config config_realworld.yaml
    python train_realworld.py --config config_realworld.yaml --dataset roman_empire

Results are saved to results_realworld.csv (or --output).
"""

import argparse
import csv
import math
import os
import time
from collections import defaultdict

import torch
import yaml

from models import create_model, train_model


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_dataset(name: str, data_dir: str = './data'):
    """Load a real-world dataset via torch_geometric."""
    from torch_geometric.data import Data

    os.makedirs(data_dir, exist_ok=True)

    if name in ('roman_empire', 'amazon_ratings', 'minesweeper', 'tolokers', 'questions'):
        from torch_geometric.datasets import HeterophilousGraphDataset
        ds = HeterophilousGraphDataset(root=os.path.join(data_dir, name), name=name)
        data = ds[0]
        # Use first fixed split (index 0)
        data.train_mask = data.train_mask[:, 0]
        data.val_mask   = data.val_mask[:, 0]
        data.test_mask  = data.test_mask[:, 0]
        return data

    elif name == 'penn94':
        from torch_geometric.datasets import LINKXDataset
        ds = LINKXDataset(root=os.path.join(data_dir, 'penn94'), name='Penn94')
        data = ds[0]
        # Generate a stratified 60/20/20 split if not present
        if not hasattr(data, 'train_mask') or data.train_mask is None:
            data = _make_split(data, train_ratio=0.6, val_ratio=0.2, seed=42)
        elif data.train_mask.dim() > 1:
            data.train_mask = data.train_mask[:, 0]
            data.val_mask   = data.val_mask[:, 0]
            data.test_mask  = data.test_mask[:, 0]
        return data

    elif name in ('chameleon', 'squirrel'):
        from torch_geometric.datasets import WikipediaNetwork
        ds = WikipediaNetwork(root=os.path.join(data_dir, name), name=name, geom_gcn_preprocess=True)
        data = ds[0]
        if data.train_mask.dim() > 1:
            data.train_mask = data.train_mask[:, 0]
            data.val_mask   = data.val_mask[:, 0]
            data.test_mask  = data.test_mask[:, 0]
        return data

    elif name == 'actor':
        from torch_geometric.datasets import Actor
        ds = Actor(root=os.path.join(data_dir, 'actor'))
        data = ds[0]
        if data.train_mask.dim() > 1:
            data.train_mask = data.train_mask[:, 0]
            data.val_mask   = data.val_mask[:, 0]
            data.test_mask  = data.test_mask[:, 0]
        return data

    else:
        raise ValueError(
            f"Unknown dataset '{name}'. Supported: roman_empire, amazon_ratings, "
            "minesweeper, tolokers, questions, penn94, chameleon, squirrel, actor"
        )


def _make_split(data, train_ratio=0.6, val_ratio=0.2, seed=42):
    """Generate stratified train/val/test split."""
    import numpy as np
    from torch_geometric.data import Data

    torch.manual_seed(seed)
    np.random.seed(seed)

    n = data.num_nodes
    y = data.y.numpy()
    classes = np.unique(y)

    train_idx, val_idx, test_idx = [], [], []
    for c in classes:
        idx = np.where(y == c)[0]
        np.random.shuffle(idx)
        n_train = max(1, int(len(idx) * train_ratio))
        n_val = max(1, int(len(idx) * val_ratio))
        train_idx.extend(idx[:n_train].tolist())
        val_idx.extend(idx[n_train:n_train + n_val].tolist())
        test_idx.extend(idx[n_train + n_val:].tolist())

    data.train_mask = torch.zeros(n, dtype=torch.bool)
    data.val_mask   = torch.zeros(n, dtype=torch.bool)
    data.test_mask  = torch.zeros(n, dtype=torch.bool)
    data.train_mask[train_idx] = True
    data.val_mask[val_idx]     = True
    data.test_mask[test_idx]   = True
    return data


def run_experiments(cfg: dict, output_path: str, dataset_filter=None):
    datasets = cfg['datasets']
    if dataset_filter:
        datasets = [d for d in datasets if d == dataset_filter]

    models_cfg = cfg['models']
    num_layers_list = cfg['num_layers'] if isinstance(cfg['num_layers'], list) else [cfg['num_layers']]
    hidden_dim = cfg['hidden_dim']
    num_stalks = cfg['num_stalks']
    model_seeds = cfg['model_seeds']
    epochs = cfg.get('epochs', 500)
    lr = cfg.get('lr', 0.01)
    data_dir = cfg.get('data_dir', './data')

    fieldnames = ['dataset', 'model', 'flags', 'num_layers', 'model_seed', 'test_acc', 'val_acc', 'epochs_run']
    write_header = not os.path.exists(output_path)

    print(f"\nRunning real-world experiments → {output_path}")

    total = len(datasets) * len(models_cfg) * len(num_layers_list) * len(model_seeds)
    done = 0
    t0 = time.time()

    with open(output_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        for ds_name in datasets:
            print(f"\nLoading dataset: {ds_name}")
            try:
                data = load_dataset(ds_name, data_dir)
            except Exception as e:
                print(f"  ERROR loading {ds_name}: {e}")
                done += len(models_cfg) * len(num_layers_list) * len(model_seeds)
                continue

            input_dim = data.x.size(1)
            output_dim = int(data.y.max().item()) + 1
            print(f"  {data.num_nodes} nodes, {data.edge_index.size(1)} edges, "
                  f"{input_dim} features, {output_dim} classes")

            for model_cfg in models_cfg:
                model_name = model_cfg['name']
                flags = {k: v for k, v in model_cfg.items() if k != 'name'}
                flags_str = ','.join(f"{k}={v}" for k, v in flags.items()) if flags else 'default'

                for num_layers in num_layers_list:
                    for mseed in model_seeds:
                        try:
                            model = create_model(
                                model_name,
                                input_dim=input_dim,
                                hidden_dim=hidden_dim,
                                output_dim=output_dim,
                                num_stalks=num_stalks,
                                num_layers=num_layers,
                                **flags,
                            )
                            _, _, history = train_model(
                                model, data,
                                lr=lr, epochs=epochs, seed=mseed,
                            )

                            best_val = max(history.get('val_acc', [0.0]))
                            best_idx = history['val_acc'].index(best_val) if 'val_acc' in history else -1
                            best_test = history['test_acc'][best_idx]

                        except RuntimeError as e:
                            if 'out of memory' in str(e).lower():
                                print(f"  OOM: {ds_name} {model_name} layers={num_layers}")
                                best_val, best_test, best_idx = float('nan'), float('nan'), 0
                            else:
                                raise

                        writer.writerow({
                            'dataset': ds_name,
                            'model': model_name,
                            'flags': flags_str,
                            'num_layers': num_layers,
                            'model_seed': mseed,
                            'test_acc': f"{best_test:.4f}",
                            'val_acc': f"{best_val:.4f}",
                            'epochs_run': history.get('stopped_epoch') or len(history['test_acc']),
                        })
                        f.flush()

                        done += 1
                        elapsed = time.time() - t0
                        eta = (elapsed / done) * (total - done) if done > 0 else 0
                        print(f"  [{done}/{total}] {ds_name} {model_name}({flags_str}) "
                              f"L={num_layers} seed={mseed} "
                              f"test={best_test:.3f}  ETA {eta/60:.1f}min")

    print(f"\nDone. Results saved to {output_path}")
    _print_summary(output_path)


def _print_summary(output_path: str):
    rows = []
    with open(output_path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return

    groups = defaultdict(list)
    for r in rows:
        key = (r['dataset'], r['model'], r['flags'], int(r['num_layers']))
        try:
            groups[key].append(float(r['test_acc']))
        except ValueError:
            pass

    best = defaultdict(lambda: (0.0, None, None))
    for (ds, model, flags, nl), accs in groups.items():
        mean_acc = sum(accs) / len(accs)
        key = (ds, model, flags)
        if mean_acc > best[key][0]:
            best[key] = (mean_acc, nl, accs)

    print("\n=== Summary (best depth per dataset×model) ===")
    print(f"{'Dataset':<20} {'Model':<25} {'BestL':>5} {'Mean':>7} {'Std':>6}")
    for (ds, model, flags), (mean_acc, nl, accs) in sorted(best.items()):
        std = math.sqrt(sum((a - mean_acc)**2 for a in accs) / len(accs))
        label = f"{model} {flags}" if flags != 'default' else model
        print(f"{ds:<20} {label:<25} {nl:>5} {mean_acc*100:>6.1f}% {std*100:>5.1f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Path to YAML config file')
    parser.add_argument('--output', default='results_realworld.csv', help='Output CSV path')
    parser.add_argument('--dataset', default=None, help='Run only this dataset')
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_experiments(cfg, args.output, dataset_filter=args.dataset)


if __name__ == '__main__':
    main()
