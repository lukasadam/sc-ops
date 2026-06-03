import numpy as np
import pandas as pd
import anndata as ad
import networkx as nx
from pynndescent import NNDescent
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import pairwise_distances

def knn_hungarian_matching(adata_combined: ad.AnnData,
                           condition_key: str,
                           source_condition: str,
                           target_condition: str,
                           latent_key: str = "X_scVI") -> pd.DataFrame:
    """
    Perform KNN matching between perturbed and control cells in latent space using the Hungarian algorithm.
    
    Parameters:
    -------------
    adata_combined: AnnData object containing both perturbed and control cells.
    condition_key: Key in adata.obs indicating the condition of each cell.
    source_condition: Value in adata.obs[condition_key] indicating the source condition (e.g., "Perturbed").
    target_condition: Value in adata.obs[condition_key] indicating the target condition (e.g., "Control").
    latent_key: Key in adata.obsm where the latent representation is stored.
    
    Returns:
    pairs_df: DataFrame with columns ["perturbed_idx", "control_idx", "perturbed_cell_id", "control_cell_id"] containing the matched pairs of cells.
    """
    # Get perturbed adata
    adata_perturbed = adata_combined[adata_combined.obs[condition_key] == source_condition]
    adata_controls = adata_combined[adata_combined.obs[condition_key] == target_condition]

    # Split latent space by condition
    Xp = adata_perturbed.obsm[latent_key] # (nP, d) perturbed cells in embedding
    Xc = adata_controls.obsm[latent_key] # (nC, d) control cells in embedding

    # Compute pairwise euclidean distances
    D = pairwise_distances(Xp, Xc, metric="euclidean") 
    # Get optimal matching using Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(D)
    pairs = list(zip(row_ind, col_ind))
    # Make a dataframe of matched pairs
    pairs_df = pd.DataFrame(pairs, columns=["perturbed_idx", "control_idx"])
    # Add cell barcodes for perturbed idx and control idx
    pairs_df["perturbed_cell_id"] = adata_perturbed.obs_names[pairs_df["perturbed_idx"]]
    pairs_df["control_cell_id"] = adata_controls.obs_names[pairs_df["control_idx"]]
    return pairs_df


def knn_mcmf_bipartite_matching(
    adata_combined: ad.AnnData,
    condition_key: str,
    source_condition: str,
    target_condition: str,
    latent_key: str = "X_scVI",
    k: int = 50,
    n_matches: int | None = None,
    cost_scale: float = 1_000.0,
    max_k_multiplier: int = 10,
    control_capacity: int = 1,
    random_state: int = 0,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """
    Minimum-cost maximum-flow (MCMF) bipartite matching between source and target cells
    in latent space, using pynndescent for KNN candidate edges.

    - One-to-one: control_capacity=1
    - Many-to-one: control_capacity>1
    
    Parameters:
    -------------
    adata_combined: AnnData with both source and target cells.
    condition_key: Key in adata.obs indicating the condition of each cell.
    source_condition: Value in adata.obs[condition_key] for source cells (e.g., "Perturbed").
    target_condition: Value in adata.obs[condition_key] for target cells (e.g., "Control").
    latent_key: Key in adata.obsm where the latent representation is stored.
    k: Number of nearest neighbors to consider for each source cell (will be multiplied by max_k_multiplier if needed).
    n_matches: Total number of matches (flow) desired. If None, will match as many as possible up to min(nS, nT*control_capacity).
    cost_scale: Scale factor to convert float distances to integers for MCMF.
    max_k_multiplier: Maximum multiplier for k if initial KNN does not yield a feasible flow.
    control_capacity: Maximum number of source cells that can be matched to the same target cell (capacity of target nodes).
    random_state: Random seed for KNN index.
    n_jobs: Number of parallel jobs for KNN index.
    """

    adata_src = adata_combined[adata_combined.obs[condition_key] == source_condition]
    adata_tgt = adata_combined[adata_combined.obs[condition_key] == target_condition]

    Xs = adata_src.obsm[latent_key]
    Xt = adata_tgt.obsm[latent_key]

    nS = Xs.shape[0]
    nT = Xt.shape[0]

    if nS == 0 or nT == 0:
        return pd.DataFrame(columns=["source_idx", "target_idx", "source_cell_id", "target_cell_id", "cost"])

    # Total matches (flow) desired
    max_possible = min(nS, nT * int(control_capacity))
    if n_matches is None:
        n_matches = max_possible
    n_matches = int(min(n_matches, max_possible))
    if n_matches <= 0:
        return pd.DataFrame(columns=["source_idx", "target_idx", "source_cell_id", "target_cell_id", "cost"])

    # Build ANN index on targets once
    # (metric "euclidean" returns Euclidean distances in the query)
    index = NNDescent(
        Xt,
        metric="euclidean",
        random_state=random_state,
        n_jobs=n_jobs,
    )

    SRC = "SRC"
    SNK = "SNK"

    def _build_and_solve(curr_k: int):
        curr_k = int(min(curr_k, nT))
        if curr_k <= 0:
            return None

        # Query KNN candidates for each source
        # nn_idx: (nS, k), nn_dist: (nS, k)
        nn_idx, nn_dist = index.query(Xs, k=curr_k)

        G = nx.DiGraph()
        G.add_node(SRC, demand=-n_matches)
        G.add_node(SNK, demand=+n_matches)

        # Source nodes (capacity 1 per source)
        for i in range(nS):
            si = f"s{i}"
            G.add_node(si, demand=0)
            G.add_edge(SRC, si, capacity=1, weight=0)

        # Target nodes (capacity = control_capacity per target)
        cap = int(control_capacity)
        for j in range(nT):
            tj = f"t{j}"
            G.add_node(tj, demand=0)
            G.add_edge(tj, SNK, capacity=cap, weight=0)

        # Candidate edges from KNN
        # networkx min_cost_flow expects integer weights
        for i in range(nS):
            si = f"s{i}"
            for j, dist in zip(nn_idx[i], nn_dist[i]):
                tj = f"t{int(j)}"
                w = int(np.round(float(dist) * float(cost_scale)))
                G.add_edge(si, tj, capacity=1, weight=w)

        try:
            flow = nx.min_cost_flow(G)
        except nx.NetworkXUnfeasible:
            return None

        # Extract used edges s{i} -> t{j}
        pairs = []
        for i in range(nS):
            si = f"s{i}"
            if si not in flow:
                continue
            for tj, f in flow[si].items():
                if f <= 0 or not tj.startswith("t"):
                    continue
                j = int(tj[1:])
                w = G[si][tj]["weight"]
                for _ in range(int(f)):  # usually 1
                    pairs.append((i, j, w))

        if len(pairs) != n_matches:
            return None

        return pairs

    pairs = None
    for mult in range(1, max_k_multiplier + 1):
        curr_k = k * mult
        pairs = _build_and_solve(curr_k)
        if pairs is not None:
            break

    if pairs is None:
        raise RuntimeError(
            f"MCMF matching infeasible with k up to {min(k*max_k_multiplier, nT)}. "
            f"Try increasing k/max_k_multiplier, or increase control_capacity."
        )

    pairs_df = pd.DataFrame(pairs, columns=["source_idx", "target_idx", "cost_int"])
    pairs_df["cost"] = pairs_df["cost_int"] / float(cost_scale)
    pairs_df.drop(columns=["cost_int"], inplace=True)

    pairs_df["source_cell_id"] = adata_src.obs_names[pairs_df["source_idx"]].to_numpy()
    pairs_df["target_cell_id"] = adata_tgt.obs_names[pairs_df["target_idx"]].to_numpy()

    return pairs_df