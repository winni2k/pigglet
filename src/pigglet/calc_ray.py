import h5py
import networkx as nx
import ray
import numpy as np
from pigglet.scipy_import import logsumexp


from pigglet.likelihoods import PhyloTreeLikelihoodCalculator
from pigglet.tree_interactor import GraphAnnotator


if not ray.is_initialized():
    ray.init()


@ray.remote
def calc_phylo_branch_lengths_for_tree(gls, g_edge_list):
    g = nx.DiGraph(list(g_edge_list))
    GraphAnnotator(g).annotate_all_nodes_with_descendant_leaves()
    calc = PhyloTreeLikelihoodCalculator(g, gls)
    site_like_total = calc.attachment_marginalized_log_likelihoods()
    # log_likes.shape = len(g) x m
    log_likes = calc.attachment_log_like - site_like_total
    expected_num_sites = np.exp(logsumexp(log_likes, axis=1))
    for n in range(len(expected_num_sites)):
        g.nodes[n]["expected_n_mut"] = expected_num_sites[n]
    return g


def calc_branch_lengths_samples(gls, h5_file, n_sampling_iterations):
    gls_id = ray.put(gls)
    futures = []
    with h5py.File(h5_file, "r") as fh:
        for idx in range(n_sampling_iterations):
            futures.append(
                calc_phylo_branch_lengths_for_tree.remote(
                    gls_id, fh[f"phylo_tree/samples/{idx}"][:]
                )
            )
    with h5py.File(h5_file, "a") as fh:
        for idx, fut in enumerate(futures):
            g = ray.get(fut)
            fh.create_dataset(
                f"phylo_tree/samples_br_lens/{idx}",
                data=np.array(
                    [g.nodes[n]["expected_n_mut"] for n in range(len(g))]
                ),
            )


def calc_branch_lengths_map(gls, h5_file):
    gls_id = ray.put(gls)
    with h5py.File(h5_file, "r") as fh:
        fut = calc_phylo_branch_lengths_for_tree.remote(
            gls_id, fh["map_phylo_tree/edge_list"][:]
        )
    with h5py.File(h5_file, "a") as fh:
        g = ray.get(fut)
        fh.create_dataset(
            "map_phylo_tree/br_lens",
            data=np.array(
                [g.nodes[n]["expected_n_mut"] for n in range(len(g))]
            ),
        )
