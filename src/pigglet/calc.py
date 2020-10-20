import math
from multiprocessing import Process, SimpleQueue

import h5py
import networkx as nx
import numpy as np
from tqdm import tqdm

from pigglet.aggregator import PhyloAttachmentAggregator, tree_to_newick
from pigglet.likelihoods import PhyloTreeLikelihoodCalculator
from pigglet.scipy_import import logsumexp
from pigglet.tree_interactor import GraphAnnotator


def calc_phylo_mutation_probs_for_tree(
    h5_file, gls, input_q, output_q, tracking_q
):

    agg = PhyloAttachmentAggregator()
    with h5py.File(h5_file, "r") as fh:
        while True:
            tree_idx = input_q.get()
            if tree_idx is None:
                output_q.put(agg.attachment_scores)
                return
            g = nx.DiGraph(list(fh[f"phylo_tree/samples/{tree_idx}"]))
            GraphAnnotator(g).annotate_all_nodes_with_descendant_leaves()
            agg.add_attachment_log_likes(PhyloTreeLikelihoodCalculator(g, gls))
            tracking_q.put(tree_idx)


def calc_mutation_probabilities(gls, h5_file, jobs, n_sampling_iterations):
    processes = []
    input_q = SimpleQueue()
    for idx in range(n_sampling_iterations):
        input_q.put(idx)
    output_q = SimpleQueue()
    tracking_q = SimpleQueue()
    for _ in range(jobs):
        p = Process(
            target=calc_phylo_mutation_probs_for_tree,
            args=(h5_file, gls, input_q, output_q, tracking_q),
        )
        p.start()
        processes.append(p)
        input_q.put(None)
    for _ in tqdm(
        range(n_sampling_iterations),
        unit="trees",
        desc="Mutation probabilities",
        total=n_sampling_iterations,
    ):
        tracking_q.get()
    summed_attach_probs = None
    for _ in range(jobs):
        output = output_q.get()
        if summed_attach_probs is None:
            summed_attach_probs = output
        else:
            summed_attach_probs = np.logaddexp(summed_attach_probs, output)
    for p in processes:
        p.join()
    assert input_q.empty()
    assert output_q.empty()
    sum_ll = summed_attach_probs - math.log(n_sampling_iterations)
    with h5py.File(h5_file, "r+") as fh:
        if "phylo_tree/mutation_probabilities" in fh:
            del fh["phylo_tree/mutation_probabilities"]
        fh.create_dataset(
            "phylo_tree/mutation_probabilities",
            data=sum_ll,
            compression="gzip",
        )


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


def calc_tree_stats_for_digraph(gls, g, leaf_lookup=None, one_base=True):
    g_with_exp_n_mut = calc_phylo_branch_lengths_for_tree(gls, g.edges)
    ll = PhyloTreeLikelihoodCalculator(g, gls).log_likelihood()
    node_branch_lengths = {
        k: v for k, v in g_with_exp_n_mut.nodes(data="expected_n_mut")
    }
    return (
        ll,
        tree_to_newick(
            g_with_exp_n_mut,
            one_base=one_base,
            leaf_lookup=leaf_lookup,
            node_branch_length_lookup=node_branch_lengths,
        ),
    )
