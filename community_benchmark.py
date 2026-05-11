#!/usr/bin/env python3
"""
Community Detection Benchmark Generation

Generates synthetic community detection benchmarks with configurable
graph topology (KNN, SBM), perturbation policies, and feature types.

Usage:
    from community_benchmark import get_or_generate_benchmark, get_graph, add_train_val_test_masks

    benchmark = get_or_generate_benchmark(
        cache_path='./data/community_cache/benchmark_s42.pt',
        num_nodes=1500, num_communities=3, sigma=3.0,
        k=8, p_intra=0.005, seed=42, feature_type='gaussian',
        graph_types=['knn'], policies=['inter_only'], levels=list(range(11)),
    )
    data = get_graph(benchmark, 'knn', 'inter_only', level=5, add_masks=True, seed=42)
"""

import os
from pathlib import Path
from typing import Dict, Literal, Optional, Tuple

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_undirected, remove_self_loops


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)


# ---------------------------------------------------------------------------
# Node feature generation
# ---------------------------------------------------------------------------

def generate_node_features(
    num_nodes: int = 1500,
    num_communities: int = 3,
    sigma: float = 3.0,
    seed: int = 42,
    feature_type: str = 'gaussian'
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate node features from overlapping Gaussian distributions or constant features.

    Args:
        num_nodes: Total number of nodes (divided equally among communities)
        num_communities: Number of communities
        sigma: Standard deviation for Gaussian sampling
        seed: Random seed
        feature_type: 'gaussian' | 'random' | 'ones'

    Returns:
        x: Node features (N, 2)
        y: Community labels (N,)
    """
    set_seed(seed)

    nodes_per_community = num_nodes // num_communities
    N = nodes_per_community * num_communities

    labels_list = [torch.full((nodes_per_community,), c, dtype=torch.long)
                   for c in range(num_communities)]
    y = torch.cat(labels_list, dim=0)

    if feature_type == 'ones':
        x = torch.ones(N, 2)
    elif feature_type == 'gaussian':
        means = torch.tensor([
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ])
        features_list = []
        for c in range(num_communities):
            community_features = means[c] + sigma * torch.randn(nodes_per_community, 2)
            features_list.append(community_features)
        x = torch.cat(features_list, dim=0)
    elif feature_type == 'random':
        x = sigma * torch.randn(N, 2)
    else:
        raise ValueError(f"Unknown feature_type: {feature_type}")

    return x, y


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_knn_graph(
    x: torch.Tensor,
    y: torch.Tensor,
    k: int = 4
) -> torch.Tensor:
    """Build k-NN graph where nodes connect to k nearest neighbors WITHIN their community."""
    num_communities = y.max().item() + 1
    edge_list = []

    for c in range(num_communities):
        mask = (y == c)
        community_indices = torch.where(mask)[0].numpy()
        community_features = x[mask].numpy()

        n_neighbors = min(k + 1, len(community_indices))
        nn = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
        nn.fit(community_features)
        distances, indices = nn.kneighbors(community_features)

        for i, neighbors in enumerate(indices):
            global_src = community_indices[i]
            for j in neighbors:
                if i != j:
                    global_dst = community_indices[j]
                    edge_list.append([global_src, global_dst])

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    edge_index = to_undirected(edge_index)
    edge_index, _ = remove_self_loops(edge_index)
    return edge_index


def build_sbm_graph(
    y: torch.Tensor,
    p_intra: float = 0.005,
    p_inter: float = 0.0,
    seed: int = 42
) -> torch.Tensor:
    """Build Stochastic Block Model graph."""
    set_seed(seed)
    communities = torch.unique(y)
    comm_indices = {c.item(): (y == c).nonzero(as_tuple=True)[0] for c in communities}

    edge_src = []
    edge_dst = []

    for i, ci in enumerate(communities):
        idx_i = comm_indices[ci.item()]
        for j, cj in enumerate(communities):
            if j < i:
                continue
            idx_j = comm_indices[cj.item()]
            p = p_intra if ci == cj else p_inter
            if p == 0.0:
                continue

            ni, nj = len(idx_i), len(idx_j)
            rand_block = torch.rand(ni, nj)

            if ci == cj:
                mask = torch.triu(rand_block < p, diagonal=1)
            else:
                mask = rand_block < p

            local_src, local_dst = mask.nonzero(as_tuple=True)
            global_src = idx_i[local_src]
            global_dst = idx_j[local_dst]

            edge_src.append(global_src)
            edge_src.append(global_dst)
            edge_dst.append(global_dst)
            edge_dst.append(global_src)

    if len(edge_src) == 0:
        return torch.zeros((2, 0), dtype=torch.long)

    edge_index = torch.stack([torch.cat(edge_src), torch.cat(edge_dst)], dim=0)
    return edge_index


# ---------------------------------------------------------------------------
# Graph perturbation
# ---------------------------------------------------------------------------

def perturb_graph(
    edge_index: torch.Tensor,
    y: torch.Tensor,
    perturbation_level: int,
    policy: Literal['intra_only', 'inter_only', 'mixed'],
    seed: int = 42
) -> torch.Tensor:
    """
    Perturb graph by removing i*10% of edges and adding same number back per policy.
    """
    if perturbation_level == 0:
        return edge_index.clone()

    set_seed(seed + perturbation_level)

    num_nodes = len(y)
    num_communities = y.max().item() + 1

    edges_set = set()
    for i in range(edge_index.shape[1]):
        u, v = edge_index[0, i].item(), edge_index[1, i].item()
        edges_set.add((min(u, v), max(u, v)))

    total_edges = len(edges_set)
    num_to_perturb = int(total_edges * perturbation_level * 0.1)

    if num_to_perturb == 0:
        return edge_index.clone()

    edges_list = list(edges_set)
    remove_indices = np.random.choice(len(edges_list), size=num_to_perturb, replace=False)
    edges_to_remove = {edges_list[i] for i in remove_indices}
    remaining_edges = edges_set - edges_to_remove

    def sample_new_edges(num_edges, policy, existing_edges):
        new_edges = set()
        attempts = 0
        max_attempts = num_edges * 100

        while len(new_edges) < num_edges and attempts < max_attempts:
            attempts += 1

            if policy == 'intra_only':
                c = np.random.randint(0, num_communities)
                community_nodes = torch.where(y == c)[0].numpy()
                if len(community_nodes) < 2:
                    continue
                u, v = np.random.choice(community_nodes, size=2, replace=False)
            elif policy == 'inter_only':
                c1, c2 = np.random.choice(num_communities, size=2, replace=False)
                nodes_c1 = torch.where(y == c1)[0].numpy()
                nodes_c2 = torch.where(y == c2)[0].numpy()
                u = np.random.choice(nodes_c1)
                v = np.random.choice(nodes_c2)
            else:  # mixed
                u, v = np.random.choice(num_nodes, size=2, replace=False)

            edge = (min(u, v), max(u, v))
            if edge not in existing_edges and edge not in new_edges and u != v:
                new_edges.add(edge)

        return new_edges

    new_edges = sample_new_edges(num_to_perturb, policy, remaining_edges)
    final_edges = remaining_edges | new_edges

    edge_list = []
    for u, v in final_edges:
        edge_list.append([u, v])
        edge_list.append([v, u])

    return torch.tensor(edge_list, dtype=torch.long).t().contiguous()


def select_subset_by_feature(x: torch.Tensor, criterion: str = 'first_positive') -> torch.Tensor:
    if criterion == 'first_positive':
        return x[:, 0] > 0
    elif criterion == 'first_negative':
        return x[:, 0] < 0
    elif criterion == 'second_positive':
        return x[:, 1] > 0
    else:
        raise ValueError(f"Unknown criterion: {criterion}")


def perturb_graph_subset(
    edge_index: torch.Tensor,
    y: torch.Tensor,
    subset_mask: torch.Tensor,
    perturbation_level: int,
    policy: Literal['intra_only', 'inter_only', 'mixed'],
    seed: int = 42
) -> torch.Tensor:
    """Perturb only edges within a fixed subset of nodes."""
    if perturbation_level == 0:
        return edge_index.clone()

    set_seed(seed + perturbation_level)

    num_communities = y.max().item() + 1
    subset_indices = torch.where(subset_mask)[0].tolist()
    subset_set = set(subset_indices)

    all_edges = set()
    subset_edges = set()
    non_subset_edges = set()

    for i in range(edge_index.shape[1]):
        u, v = edge_index[0, i].item(), edge_index[1, i].item()
        edge = (min(u, v), max(u, v))
        if edge in all_edges:
            continue
        all_edges.add(edge)
        if u in subset_set and v in subset_set:
            subset_edges.add(edge)
        else:
            non_subset_edges.add(edge)

    total_subset_edges = len(subset_edges)
    num_to_perturb = int(total_subset_edges * perturbation_level * 0.1)

    if num_to_perturb == 0:
        return edge_index.clone()

    subset_list = list(subset_edges)
    remove_indices = np.random.choice(len(subset_list), size=num_to_perturb, replace=False)
    edges_to_remove = {subset_list[i] for i in remove_indices}
    remaining_subset_edges = subset_edges - edges_to_remove

    def sample_new_edges_in_subset(num_edges, policy, existing_edges, subset_indices, y):
        new_edges = set()
        attempts = 0
        max_attempts = num_edges * 100

        while len(new_edges) < num_edges and attempts < max_attempts:
            attempts += 1
            if policy == 'intra_only':
                c = np.random.randint(0, num_communities)
                subset_in_community = [idx for idx in subset_indices if y[idx] == c]
                if len(subset_in_community) < 2:
                    continue
                u, v = np.random.choice(subset_in_community, size=2, replace=False)
            elif policy == 'inter_only':
                c1, c2 = np.random.choice(num_communities, size=2, replace=False)
                subset_c1 = [idx for idx in subset_indices if y[idx] == c1]
                subset_c2 = [idx for idx in subset_indices if y[idx] == c2]
                if not subset_c1 or not subset_c2:
                    continue
                u = np.random.choice(subset_c1)
                v = np.random.choice(subset_c2)
            else:  # mixed
                if len(subset_indices) < 2:
                    break
                u, v = np.random.choice(subset_indices, size=2, replace=False)

            edge = (min(u, v), max(u, v))
            if edge not in existing_edges and edge not in new_edges and u != v:
                new_edges.add(edge)

        return new_edges

    new_subset_edges = sample_new_edges_in_subset(
        num_to_perturb, policy, remaining_subset_edges, subset_indices, y
    )

    final_edges = non_subset_edges | remaining_subset_edges | new_subset_edges

    edge_list = []
    for u, v in final_edges:
        edge_list.append([u, v])
        edge_list.append([v, u])

    return torch.tensor(edge_list, dtype=torch.long).t().contiguous()


# ---------------------------------------------------------------------------
# Benchmark generation and caching
# ---------------------------------------------------------------------------

def resolve_p_intra(p_intra, k: int, num_nodes: int, num_communities: int) -> float:
    """Resolve p_intra: if 'match_knn', compute from k to match average KNN degree."""
    if isinstance(p_intra, str) and p_intra == 'match_knn':
        nodes_per_community = num_nodes // num_communities
        p = k / (nodes_per_community - 1)
        print(f"  p_intra='match_knn' → p_intra={p:.6f} (k={k}, {nodes_per_community} nodes/community)")
        return p
    return float(p_intra)


def generate_community_benchmark(
    num_nodes: int = 1500,
    num_communities: int = 3,
    sigma: float = 3.0,
    k: int = 4,
    p_intra: float = 0.005,
    seed: int = 42,
    use_subset: bool = False,
    subset_criterion: str = 'first_positive',
    feature_type: str = 'gaussian',
    graph_types: list = None,
    policies: list = None,
    levels: list = None,
) -> dict:
    """
    Generate the community detection benchmark.

    Returns nested dict: {graph_type: {policy: [Data_or_None, ...]}}
    """
    if graph_types is None:
        graph_types = ['knn', 'sbm']
    if policies is None:
        policies = ['intra_only', 'inter_only', 'mixed']
    if levels is None:
        levels = list(range(11))
    max_level = max(levels)

    p_intra = resolve_p_intra(p_intra, k, num_nodes, num_communities)

    x, y = generate_node_features(num_nodes, num_communities, sigma, seed, feature_type=feature_type)

    if use_subset:
        subset_mask = select_subset_by_feature(x, criterion=subset_criterion)
        print(f"Subset perturbations enabled: {subset_mask.sum().item()}/{len(y)} nodes selected")

    all_policies = ['intra_only', 'inter_only', 'mixed']
    all_graph_types = ['knn', 'sbm']

    base_graphs = {}
    if 'knn' in graph_types:
        base_graphs['knn'] = build_knn_graph(x, y, k)
    if 'sbm' in graph_types:
        base_graphs['sbm'] = build_sbm_graph(y, p_intra, p_inter=0.0, seed=seed)

    datasets = {
        gt: {pol: [None] * (max_level + 1) for pol in all_policies}
        for gt in all_graph_types
    }

    for graph_type in graph_types:
        base_edge_index = base_graphs[graph_type]
        for policy in policies:
            for level in levels:
                if use_subset:
                    perturbed_edge_index = perturb_graph_subset(
                        base_edge_index, y, subset_mask, level, policy, seed
                    )
                else:
                    perturbed_edge_index = perturb_graph(
                        base_edge_index, y, level, policy, seed
                    )

                data = Data(
                    x=x.clone(),
                    edge_index=perturbed_edge_index,
                    y=y.clone(),
                    num_nodes=len(y),
                    num_classes=num_communities
                )
                data.graph_type = graph_type
                data.policy = policy
                data.perturbation_level = level
                if use_subset:
                    data.subset_mask = subset_mask.clone()

                datasets[graph_type][policy][level] = data

    return datasets


def add_train_val_test_masks(
    data: Data,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    seed: int = 42
) -> Data:
    """Add train/val/test masks to a Data object (70/20/10 split by default)."""
    set_seed(seed)

    num_nodes = data.num_nodes
    indices = np.random.permutation(num_nodes)

    train_size = int(num_nodes * train_ratio)
    val_size = int(num_nodes * val_ratio)

    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size + val_size]
    test_idx = indices[train_size + val_size:]

    data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    data.train_mask[train_idx] = True
    data.val_mask[val_idx] = True
    data.test_mask[test_idx] = True

    return data


def save_benchmark(benchmark_data: dict, filepath: str):
    """Save benchmark dataset to disk."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    torch.save(benchmark_data, filepath)
    print(f"Saved benchmark to {filepath}")


def load_benchmark(filepath: str) -> dict:
    """Load benchmark dataset from disk."""
    benchmark_data = torch.load(filepath, weights_only=False)
    print(f"Loaded benchmark from {filepath}")
    return benchmark_data


def _find_missing_entries(cached, graph_types, policies, levels):
    if graph_types is None:
        graph_types = ['knn', 'sbm']
    if policies is None:
        policies = ['intra_only', 'inter_only', 'mixed']
    if levels is None:
        levels = list(range(11))

    missing_gts = set()
    missing_pols = set()
    missing_lvls = set()

    for gt in graph_types:
        if gt not in cached:
            missing_gts.add(gt)
            missing_pols.update(policies)
            missing_lvls.update(levels)
            continue
        for pol in policies:
            if pol not in cached[gt]:
                missing_pols.add(pol)
                missing_lvls.update(levels)
                continue
            level_list = cached[gt][pol]
            for lvl in levels:
                if lvl >= len(level_list) or level_list[lvl] is None:
                    missing_gts.add(gt)
                    missing_pols.add(pol)
                    missing_lvls.add(lvl)

    if not missing_lvls:
        return {}

    return {
        'graph_types': sorted(missing_gts),
        'policies': sorted(missing_pols),
        'levels': sorted(missing_lvls),
    }


def _merge_benchmarks(base, new):
    for gt in new:
        if gt not in base:
            base[gt] = {}
        for pol in new[gt]:
            if pol not in base[gt]:
                base[gt][pol] = []
            new_list = new[gt][pol]
            base_list = base[gt][pol]
            if len(new_list) > len(base_list):
                base_list.extend([None] * (len(new_list) - len(base_list)))
            for i, entry in enumerate(new_list):
                if entry is not None:
                    base_list[i] = entry
    return base


def get_or_generate_benchmark(
    cache_path: str,
    force_regenerate: bool = False,
    **generation_kwargs
) -> dict:
    """Load benchmark from cache or generate (and cache) if not found.

    Supports incremental generation: if cache exists but is missing requested
    entries, only missing combinations are generated and merged.
    """
    graph_types = generation_kwargs.get('graph_types')
    policies = generation_kwargs.get('policies')
    levels = generation_kwargs.get('levels')

    if os.path.exists(cache_path) and not force_regenerate:
        cached = load_benchmark(cache_path)
        missing = _find_missing_entries(cached, graph_types, policies, levels)
        if not missing:
            return cached
        print(f"Generating missing entries: {missing}")
        gen_kwargs = {**generation_kwargs, **missing}
        new_data = generate_community_benchmark(**gen_kwargs)
        merged = _merge_benchmarks(cached, new_data)
        save_benchmark(merged, cache_path)
        return merged
    else:
        print(f"Generating benchmark...")
        benchmark_data = generate_community_benchmark(**generation_kwargs)
        save_benchmark(benchmark_data, cache_path)
        return benchmark_data


def get_graph(
    benchmark_data: dict,
    graph_type: Literal['knn', 'sbm'],
    policy: Literal['intra_only', 'inter_only', 'mixed'],
    level: int,
    add_masks: bool = True,
    seed: int = 42
) -> Data:
    """Get a specific graph from the benchmark dict."""
    data = benchmark_data[graph_type][policy][level].clone()
    if add_masks:
        data = add_train_val_test_masks(data, seed=seed)
    return data
