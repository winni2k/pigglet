import math
from multiprocessing import Process, SimpleQueue

import h5py
import networkx as nx
import numpy as np
from tqdm import tqdm

from pigglet.aggregator import PhyloAttachmentAggregator
from pigglet.likelihoods import PhyloTreeLikelihoodCalculator
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
