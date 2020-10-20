import h5py
import ray
import numpy as np

import pigglet.calc

if not ray.is_initialized():
    ray.init()

calc_phylo_branch_lengths_for_tree_ray = ray.remote(
    pigglet.calc.calc_phylo_branch_lengths_for_tree
)


def calc_branch_lengths_samples(gls, h5_file, n_sampling_iterations):
    gls_id = ray.put(gls)
    futures = []
    with h5py.File(h5_file, "r") as fh:
        for idx in range(n_sampling_iterations):
            futures.append(
                calc_phylo_branch_lengths_for_tree_ray.remote(
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
        fut = calc_phylo_branch_lengths_for_tree_ray.remote(
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
