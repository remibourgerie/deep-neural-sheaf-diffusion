#!/usr/bin/env python3
"""
Synthetic community detection experiments (Table 2 in paper).

Usage:
    python train_synthetic.py --config config_synthetic.yaml
    python train_synthetic.py --config config_figure2.yaml   # Figure 2 depth sweep

Results are saved to results_synthetic.csv (or --output).
"""

import argparse
import os
import csv
import time

import torch
import yaml

from community_benchmark import get_or_generate_benchmark, get_graph, add_train_val_test_masks
from models import create_model, train_model


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_cache_path(cfg: dict, seed: int) -> str:
    gen = cfg
    cache_dir = cfg.get('cache_dir', './data/community_cache')
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(
        cache_dir,
        f"benchmark_s{seed}_n{gen['num_nodes']}_c{gen['num_communities']}"
        f"_f{gen.get('feature_type', 'gaussian')}.pt"
    )


def run_experiments(cfg: dict, output_path: str, device: str = None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    graph_type = cfg.get('graph_type', 'knn')
    policy = cfg.get('policy', 'inter_only')
    levels = cfg.get('levels', [cfg['level']]) if 'level' in cfg else cfg['levels']
    train_seeds = cfg['train_seeds']
    test_seeds = cfg['test_seeds']
    model_seeds = cfg['model_seeds']
    num_layers_list = cfg['num_layers'] if isinstance(cfg['num_layers'], list) else [cfg['num_layers']]
    models_cfg = cfg['models']
    hidden_dim = cfg['hidden_dim']
    num_stalks = cfg['num_stalks']
    epochs = cfg.get('epochs', 500)
    lr = cfg.get('lr', 0.01)

    # Pre-generate / load all benchmark graphs (cached)
    print("Loading/generating benchmark graphs...")
    train_benchmarks = {}
    for s in train_seeds:
        cache_path = get_cache_path(cfg, s)
        train_benchmarks[s] = get_or_generate_benchmark(
            cache_path,
            num_nodes=cfg['num_nodes'],
            num_communities=cfg['num_communities'],
            sigma=cfg.get('sigma', 3.0),
            k=cfg.get('k_neighbours', 8),
            p_intra=cfg.get('p_intra', 0.005),
            seed=s,
            feature_type=cfg.get('feature_type', 'gaussian'),
            graph_types=[graph_type],
            policies=[policy],
            levels=list(range(max(levels) + 1)),
        )

    test_benchmarks = {}
    for s in test_seeds:
        cache_path = get_cache_path(cfg, s)
        test_benchmarks[s] = get_or_generate_benchmark(
            cache_path,
            num_nodes=cfg['num_nodes'],
            num_communities=cfg['num_communities'],
            sigma=cfg.get('sigma', 3.0),
            k=cfg.get('k_neighbours', 8),
            p_intra=cfg.get('p_intra', 0.005),
            seed=s,
            feature_type=cfg.get('feature_type', 'gaussian'),
            graph_types=[graph_type],
            policies=[policy],
            levels=list(range(max(levels) + 1)),
        )

    # Build batched train/test graphs per level
    from torch_geometric.data import Batch

    def make_train_graph(level):
        graphs = []
        for s in train_seeds:
            g = get_graph(train_benchmarks[s], graph_type, policy, level, add_masks=False)
            graphs.append(g)
        batched = Batch.from_data_list(graphs)
        # 80/20 train/val split on batched graph
        n = batched.num_nodes
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(42))
        n_val = int(n * 0.2)
        batched.train_mask = torch.zeros(n, dtype=torch.bool)
        batched.val_mask = torch.zeros(n, dtype=torch.bool)
        batched.test_mask = torch.zeros(n, dtype=torch.bool)
        batched.val_mask[perm[:n_val]] = True
        batched.train_mask[perm[n_val:]] = True
        return batched

    def make_test_graph(level):
        graphs = [
            get_graph(test_benchmarks[s], graph_type, policy, level, add_masks=False)
            for s in test_seeds
        ]
        return Batch.from_data_list(graphs)

    input_dim = cfg['num_nodes'] // cfg['num_communities']  # placeholder
    # Infer input_dim from first graph
    sample = get_graph(train_benchmarks[train_seeds[0]], graph_type, policy, levels[0], add_masks=False)
    input_dim = sample.x.size(1)
    output_dim = cfg['num_communities']

    fieldnames = ['model', 'flags', 'level', 'num_layers', 'model_seed', 'test_acc', 'val_acc', 'epochs_run']
    write_header = not os.path.exists(output_path)

    print(f"\nRunning experiments → {output_path}")
    print(f"  levels={levels}, num_layers={num_layers_list}, models={len(models_cfg)}, model_seeds={model_seeds}")

    total = len(levels) * len(num_layers_list) * len(models_cfg) * len(model_seeds)
    done = 0
    t0 = time.time()

    with open(output_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        for level in levels:
            data_train = make_train_graph(level)
            data_test = make_test_graph(level)

            for model_cfg in models_cfg:
                model_name = model_cfg['name']
                flags = {k: v for k, v in model_cfg.items() if k != 'name'}
                flags_str = ','.join(f"{k}={v}" for k, v in flags.items()) if flags else 'default'

                for num_layers in num_layers_list:
                    for mseed in model_seeds:
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
                            model, data_train,
                            lr=lr, epochs=epochs, seed=mseed,
                            test_data=data_test,
                        )

                        best_epoch = history.get('stopped_epoch') or len(history['test_acc'])
                        best_val = max(history['val_acc']) if 'val_acc' in history else float('nan')
                        best_test = history['test_acc'][
                            history['val_acc'].index(best_val)
                            if 'val_acc' in history and history['val_acc']
                            else -1
                        ]

                        writer.writerow({
                            'model': model_name,
                            'flags': flags_str,
                            'level': level,
                            'num_layers': num_layers,
                            'model_seed': mseed,
                            'test_acc': f"{best_test:.4f}",
                            'val_acc': f"{best_val:.4f}",
                            'epochs_run': best_epoch,
                        })
                        f.flush()

                        done += 1
                        elapsed = time.time() - t0
                        eta = (elapsed / done) * (total - done) if done > 0 else 0
                        print(f"  [{done}/{total}] L{level} {model_name}({flags_str}) "
                              f"layers={num_layers} seed={mseed} "
                              f"test={best_test:.3f}  ETA {eta/60:.1f}min")

    print(f"\nDone. Results saved to {output_path}")
    _print_summary(output_path)


def _print_summary(output_path: str):
    import csv
    from collections import defaultdict

    rows = []
    with open(output_path) as f:
        rows = list(csv.DictReader(f))

    if not rows:
        return

    # Group by (model, flags, level, num_layers) → best_test_acc across model_seeds
    groups = defaultdict(list)
    for r in rows:
        key = (r['model'], r['flags'], int(r['level']), int(r['num_layers']))
        groups[key].append(float(r['test_acc']))

    # For each (model, flags, level): find best num_layers
    best = defaultdict(lambda: (0.0, None))
    for (model, flags, level, nl), accs in groups.items():
        mean_acc = sum(accs) / len(accs)
        key = (model, flags, level)
        if mean_acc > best[key][0]:
            best[key] = (mean_acc, nl, accs)

    print("\n=== Summary (best depth per model×level) ===")
    print(f"{'Model':<30} {'Level':>5} {'BestL':>6} {'Mean':>7} {'Std':>6}")
    import math
    for (model, flags, level), (mean_acc, nl, accs) in sorted(best.items()):
        std = math.sqrt(sum((a - mean_acc)**2 for a in accs) / len(accs))
        print(f"{model+' '+flags:<30} {level:>5} {nl:>6} {mean_acc*100:>6.1f}% {std*100:>5.1f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Path to YAML config file')
    parser.add_argument('--output', default='results_synthetic.csv', help='Output CSV path')
    parser.add_argument('--device', default=None, help='cuda or cpu (auto-detect if not set)')
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_experiments(cfg, args.output, device=args.device)


if __name__ == '__main__':
    main()
