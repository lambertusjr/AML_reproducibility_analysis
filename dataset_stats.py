"""
Report the exact number of nodes and edges for every dataset in
``config.DATASET_PATHS`` (including all four IBM AML variants).

Requires the processed ``data.pt`` caches to already exist (i.e.
``pre_process_datasets.py`` has been run, or ``main.py`` has been run at
least once per dataset) since this simply loads each dataset the same way
the rest of the pipeline does.

Note on edge counts: every loader stores the graph as an undirected
edge_index (each original edge appears twice, once per direction). This
script reports both:
  - "edges (stored)"    -> edge_index.shape[1], i.e. what data.edge_index
                            actually contains (both directions).
  - "edges (directed)"  -> stored // 2, i.e. the number of original
                            transactions/relations before symmetrisation.

Usage:
    python dataset_stats.py
"""
from config import DATASET_PATHS


def _load(name):
    if name == "Elliptic":
        from pre_processing import EllipticDataset
        return EllipticDataset(root=DATASET_PATHS[name])[0]
    if name == "IBM_AML_HiSmall":
        from pre_processing import IBMAMLDataset_HiSmall
        return IBMAMLDataset_HiSmall(root=DATASET_PATHS[name])[0]
    if name == "IBM_AML_LiSmall":
        from pre_processing import IBMAMLDataset_LiSmall
        return IBMAMLDataset_LiSmall(root=DATASET_PATHS[name])[0]
    if name == "IBM_AML_HiMedium":
        from pre_processing import IBMAMLDataset_HiMedium
        return IBMAMLDataset_HiMedium(root=DATASET_PATHS[name])[0]
    if name == "IBM_AML_LiMedium":
        from pre_processing import IBMAMLDataset_LiMedium
        return IBMAMLDataset_LiMedium(root=DATASET_PATHS[name])[0]
    if name == "AMLSim":
        from pre_processing import AMLSimDataset
        return AMLSimDataset(root=DATASET_PATHS[name])[0]
    raise ValueError(f"Unknown dataset '{name}'")


def dataset_stats():
    rows = []
    for name in DATASET_PATHS:
        data = _load(name)
        num_nodes = data.num_nodes
        num_edges_stored = data.edge_index.shape[1]
        num_edges_directed = num_edges_stored // 2
        rows.append((name, num_nodes, num_edges_directed, num_edges_stored))
    return rows


def main():
    rows = dataset_stats()
    header = f"{'Dataset':<20}{'Nodes':>12}{'Edges (directed)':>20}{'Edges (stored)':>18}"
    print(header)
    print("-" * len(header))
    for name, num_nodes, num_edges_directed, num_edges_stored in rows:
        print(f"{name:<20}{num_nodes:>12,}{num_edges_directed:>20,}{num_edges_stored:>18,}")


if __name__ == "__main__":
    main()
