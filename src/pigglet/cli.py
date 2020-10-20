import click


def configure_logger(log_level, log_file):
    import logging

    logging.basicConfig(
        level=log_level,
        format="%(process)d|pigglet|%(asctime)s|%(levelname)s  %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        handlers=[
            logging.FileHandler(log_file, mode="w"),
            logging.StreamHandler(),
        ],
    )


def get_version():
    import pkg_resources  # part of setuptools

    return pkg_resources.require("pigglet")[0].version


@click.group()
@click.version_option()
def cli():
    """The PIGGLET: The Phylogenetic Inference from Genotype Likelihoods Tool

    Author: Warren W. Kretzschmar
    """


def set_numexpr_threads():
    import os

    if "NUMEXPR_MAX_THREADS" not in os.environ:
        os.environ["NUMEXPR_MAX_THREADS"] = "8"


def create_store(output_store):
    import h5py

    with h5py.File(output_store, "w-"):
        pass


@cli.command(context_settings=dict(show_default=True))
@click.argument("gl_vcf", type=click.Path(exists=True))
@click.argument("out_prefix", type=click.Path())
@click.option(
    "--normal",
    "log_level",
    flag_value="INFO",
    default=True,
    help="Set normal log level",
)
@click.option(
    "--silent",
    "log_level",
    flag_value="WARNING",
    help="Only report warnings and errors",
)
@click.option(
    "--verbose",
    "log_level",
    flag_value="DEBUG",
    help="Report lots of debugging information",
)
@click.option(
    "-b",
    "--burnin",
    type=int,
    default=10,
    show_default=True,
    help="Number of burn-in iterations to run",
)
@click.option(
    "-s",
    "--sampling",
    type=int,
    default=10,
    show_default=True,
    help="Number of sampling iterations to run",
)
@click.option(
    "--reporting-interval",
    type=int,
    default=1000,
    show_default=True,
    help="Report MCMC progress after this number of iterations",
)
@click.option(
    "--store-gls/--no-store-gls",
    default=True,
    help="Store the input GLs in the output h5 file.",
)
@click.option(
    "--logsumexp-refresh-rate",
    default=1000,
    help="Refresh log sum exponent calculation every n calculations. ",
)
@click.option(
    "--check-logsumexp-accuracy/--no-check-logsumexp-accuracy", default=False
)
@click.option(
    "--mutation-tree/--no-mutation-tree",
    default=False,
    help="Use a mutation tree instead of a phylogenetic tree for inference",
)
@click.option("--seed", default=None, type=int, help="Set random seed")
@click.option(
    "--double-check-likelihood-calculation/"
    "--no-double-check-likelihood-calculation",
    default=False,
    help="This is slow and only meant for debugging.",
)
@click.option(
    "--defer-mutation-probability-calc/"
    "--no-defer-mutation-probability-calc",
    default=False,
    help="Don't calculate mutation probabilities while sampling trees. "
    "Mutation probabilities can later be estimated in parallel using "
    "`pigglet calc`",
)
@click.option(
    "--num-actors",
    default=2,
    type=int,
    help="Number of actors to use for phylogenetic tree inference.",
)
@click.option(
    "-f",
    "--force",
    is_flag=True,
    help="Delete any pre-existing output files before running.",
)
def infer(
    gl_vcf,
    out_prefix,
    burnin,
    sampling,
    log_level,
    reporting_interval,
    store_gls,
    logsumexp_refresh_rate,
    check_logsumexp_accuracy,
    mutation_tree,
    seed,
    double_check_likelihood_calculation,
    defer_mutation_probability_calc,
    num_actors,
    force,
):
    """Infer phylogenetic or mutation tree from genotype likelihoods
    stored in GL_VCF.

    Save the resulting tree and mutation probabilities to OUT_PREFIX.

    Mutations and samples are ordered in the output according to their order
    in GL_VCF.
    """

    configure_logger(log_level=log_level, log_file=out_prefix + ".log")
    import logging
    import sys

    logger = logging.getLogger(__name__)
    if logger.isEnabledFor(logging.INFO):
        version = get_version()
        from pyfiglet import Figlet

        f = Figlet(font="speed")
        print(f.renderText("The PIGGLET"), file=sys.stderr)
        logger.info(f"v{version}")

    set_numexpr_threads()

    import numpy as np
    import random

    from pigglet.constants import HET_NUM, HOM_REF_NUM
    from pigglet.gl_loader import LikelihoodLoader
    from pigglet.mcmc import MCMCRunner
    from pigglet.aggregator import TreeAggregator, NullAttachmentAggregator

    if seed is None:
        seed = random.randrange(sys.maxsize)
    logger.info(f"Random seed: {seed}")
    random.seed(seed)

    output_store = out_prefix + ".h5"
    if force:
        from pathlib import Path

        store_path = Path(output_store)
        if store_path.is_file():
            logger.info(f"Deleting previous output store: {output_store}")
            store_path.unlink()

    logger.info(f"Creating output store: {output_store}")
    create_store(output_store)

    logger.info("Loading GLs in %s", gl_vcf)
    loader = LikelihoodLoader(vcf_file=gl_vcf)
    gls = loader.load()

    logger.info("Storing input")
    store_input(gls, loader, output_store, store_gls)
    del loader

    logger.info("Loaded %s sites and %s samples", gls.shape[0], gls.shape[1])
    missingness = (
        np.sum(gls[:, :, HOM_REF_NUM] == gls[:, :, HET_NUM])
        / gls.shape[0]
        / gls.shape[1]
    )
    logger.info(f"Proportion missing sites: {missingness}")

    logger.info(
        "Running MCMC with %s burnin and %s sampling iterations",
        burnin,
        sampling,
    )
    if mutation_tree:
        runner = MCMCRunner.mutation_tree_from_gls(gls)
    else:
        runner = MCMCRunner.phylogenetic_tree_from_gls(
            gls,
            tree_move_weights=[int(gls.shape[1] != 3), 1],
            double_check_ll_calculation=double_check_likelihood_calculation,
            num_actors=num_actors,
        )
    runner.num_burnin_iter = burnin
    runner.num_sampling_iter = sampling
    runner.reporting_interval = reporting_interval

    runner.like_mover.check_logsumexp_accuracy_on = check_logsumexp_accuracy
    runner.like_mover.logsumexp_refresh_rate = logsumexp_refresh_rate
    if mutation_tree:
        logger.info("Using a mutation tree")
    else:
        runner.tree_aggregator = TreeAggregator()
        if defer_mutation_probability_calc or num_actors > 1:
            runner.agg = NullAttachmentAggregator()
        logger.info("Using a phylogenetic tree")
    runner.run()

    logger.info("Storing results")
    if mutation_tree:
        store_mutation_tree_results(gls, out_prefix, output_store, runner)
    else:
        store_phylo_tree_results(out_prefix, output_store, runner)
        if not defer_mutation_probability_calc and num_actors > 1:
            import subprocess

            subprocess.run(
                [
                    "pigglet",
                    "calc-mutation-probabilities",
                    output_store,
                    "--jobs",
                    str(num_actors),
                ]
            )


@cli.command()
@click.argument("out_prefix", type=click.Path(resolve_path=True))
@click.option("--mutation-tree", type=click.Path(exists=True), required=True)
@click.option("--hdf5", type=click.Path(exists=True), required=True)
@click.option("--hdf5-sample-attachment-descriptor", type=str, required=True)
@click.option("--hdf5-mutation-attachment-descriptor", type=str, required=True)
def convert(
    out_prefix,
    mutation_tree,
    hdf5,
    hdf5_sample_attachment_descriptor,
    hdf5_mutation_attachment_descriptor,
):
    """Convert mutation tree to phylogenetic tree"""
    import h5py
    import networkx as nx
    import numpy as np
    import random

    from pigglet.tree import strip_tree
    from pigglet.tree_converter import MutationToPhylogeneticTreeConverter

    g = nx.read_gml(mutation_tree)
    g = nx.relabel_nodes(g, int)
    converter = MutationToPhylogeneticTreeConverter(g, prng=random)
    with h5py.File(hdf5, "r") as fh:
        sample_attachments = fh[hdf5_sample_attachment_descriptor][:]
    phylo_g = converter.convert(sample_attachments)
    phylo_g = strip_tree(phylo_g)
    with h5py.File(out_prefix + ".h5", "a") as fh:
        mut_ids = np.zeros(len(converter.mutation_attachments), dtype=np.int64)
        attachments = np.zeros(
            len(converter.mutation_attachments), dtype=np.int64
        )
        for idx, key_val in enumerate(
            sorted(converter.mutation_attachments.items())
        ):
            mut_ids[idx] = key_val[0]
            attachments[idx] = key_val[1]
        mut_ids -= len(converter.sample_ids)
        fh.create_dataset(
            hdf5_mutation_attachment_descriptor + "/mutation_ids", data=mut_ids
        )
        fh.create_dataset(
            hdf5_mutation_attachment_descriptor + "/attachments",
            data=attachments,
        )
    nx.write_gml(phylo_g, out_prefix + ".gml")


@cli.command()
@click.argument("gl_vcf", type=click.Path(exists=True))
@click.argument("output_h5", type=click.Path(writable=True))
def store_gls(gl_vcf, output_h5):
    """Store GLs in GL_VCF into OUTPUT_H5"""

    import logging
    from pigglet.gl_loader import LikelihoodLoader

    logger = logging.getLogger(__name__)
    logger.info("Loading GLs in %s", gl_vcf)

    loader = LikelihoodLoader(vcf_file=gl_vcf)
    gls = loader.load()
    logger.info("Storing input")
    store_input(gls, loader, output_h5, store_gls=True)


@cli.command()
@click.argument("h5_file")
@click.option(
    "--jobs",
    type=int,
    default=None,
    help="Number of processes to parallelize over",
)
def calc_mutation_probabilities(h5_file, jobs=1):
    """Calculate mutation probabilities for posterior phylogenetic tree samples

    H5_FILE is the output file of pigglet infer ending in .h5
    """

    configure_logger("INFO", h5_file + ".calc_phylo_mutation_probs.log")
    import logging

    logger = logging.getLogger(__name__)
    logger.info(
        "Calculating mutation probabilities for stored phylogenetic trees"
    )
    logger.info(f"Opening store: {h5_file}")
    import h5py

    with h5py.File(h5_file, "r") as fh:
        n_sampling_iterations = len(list(fh["phylo_tree/samples"]))
        if "input/gls" not in fh:
            raise Exception("Please run pigglet infer with --store-gls.")
        gls = fh["input/gls"][:]

    from pigglet.calc import calc_mutation_probabilities

    calc_mutation_probabilities(gls, h5_file, jobs, n_sampling_iterations)


@cli.command()
@click.argument("h5_file")
@click.option(
    "--jobs",
    type=int,
    default=None,
    help="Number of processes to parallelize over",
)
def calc_branch_lengths(h5_file, jobs=1):
    """Calculate branch lengths for posterior sampled and MAP phylogenetic
    trees.

    H5_FILE is the output file of pigglet infer ending in .h5
    """

    configure_logger("INFO", h5_file + ".calc_branch_lengths_probs.log")
    import logging

    logger = logging.getLogger(__name__)
    logger.info("Calculating branch lengths for stored phylogenetic trees")
    logger.info(f"Opening store: {h5_file}")
    import h5py

    with h5py.File(h5_file, "r") as fh:
        n_sampling_iterations = len(list(fh["phylo_tree/samples"]))
        if "input/gls" not in fh:
            raise Exception("Please run pigglet infer with --store-gls.")
        gls = fh["input/gls"][:]

    import ray

    if not ray.is_initialized():
        ray.init(num_cpus=jobs)
    from pigglet.calc_ray import (
        calc_branch_lengths_samples,
        calc_branch_lengths_map,
    )

    calc_branch_lengths_samples(gls, h5_file, n_sampling_iterations)
    calc_branch_lengths_map(gls, h5_file)


@cli.command()
@click.argument("h5_file")
@click.argument("newick_tree", type=click.File())
@click.option(
    "--label-leaves/--no-label-leaves",
    default=False,
    help="Use sample labels on output trees",
)
@click.option(
    "--one-based/--no-one-based",
    default=True,
    help="Input node IDs are 1-based (default for Newick format)",
)
def calc_tree_stats(h5_file, newick_tree, label_leaves, one_based):
    """Calculate likelihood and branch lengths for a tree in Newick format
    from genotype likelihoods stored in H5_FILE.

    H5_FILE is the output file of pigglet infer ending in .h5
    """
    import h5py
    from pigglet.extract import extract_newick_file_as_digraph
    from pigglet.calc import calc_tree_stats_for_digraph
    import sys

    with h5py.File(h5_file, "r") as fh:
        if "input/gls" not in fh:
            raise Exception("Please run pigglet store-gls.")
        gls = fh["input/gls"][:]
        leaf_labels = None
        if label_leaves:
            print("Annotating nodes with leaf labels", file=sys.stderr)
            leaf_labels = list(fh["input/samples"])
        else:
            print("Newick output leaf nodes are zero-based", file=sys.stderr)

    for g in extract_newick_file_as_digraph(
        newick_tree, zero_based=not one_based
    ):
        ll, nw = calc_tree_stats_for_digraph(gls, g, leaf_lookup=leaf_labels)
        print(f"tree_log_likelihood={ll}\tnewick_branch_lengths={nw}")


@cli.command()
@click.argument("h5_file")
@click.option(
    "--phylo-nexus",
    type=click.File("w"),
    help="Extract posterior phylogenetic trees to NEXUS format "
    "(list of newick trees)",
)
@click.option(
    "--phylo-newicks",
    type=click.File("w"),
    help="Extract posterior phylogenetic trees to list of NEWICK trees",
)
@click.option(
    "--phylo-map-newick",
    type=click.File("w"),
    help="Extract maximum a posteriori phylogenetic tree as NEWICK tree",
)
@click.option(
    "--label-leaves/--no-label-leaves",
    default=True,
    help="Use sample labels on output trees",
)
@click.option(
    "--branch-lengths/--no-branch-lengths",
    default=True,
    help="Annotate Newick trees with branch lengths",
)
def extract(
    h5_file,
    phylo_nexus,
    phylo_newicks,
    phylo_map_newick,
    label_leaves,
    branch_lengths,
):
    """Extract trees (etc.) from output h5 file"""

    configure_logger("INFO", h5_file + ".extract.log")
    import logging

    logger = logging.getLogger(__name__)

    from pigglet.extract import NewickTreeConverter

    converter = NewickTreeConverter(
        h5_file, label_leaves, branch_lengths=branch_lengths
    )
    if phylo_nexus:
        logger.info(f"Extracting posterior trees to NEXUS file: {phylo_nexus}")
        converter.to_phylo_nexus(phylo_nexus)
    if phylo_newicks:
        logger.info(
            f"Extracting posterior trees to newicks file: {phylo_newicks}"
        )
        converter.to_phylo_newicks(phylo_newicks)
    if phylo_map_newick:
        logger.info(f"Extracting MAP tree to newick file: {phylo_map_newick}")
        converter.to_phylo_map_newick(phylo_map_newick)


def store_input(gls, loader, output_store, store_gls):
    import h5py
    import numpy as np

    with h5py.File(output_store, "a") as fh:
        if "input/site_info" in fh:
            assert loader.infos == "input/site_info"
        else:
            fh.create_dataset(
                "input/site_info",
                data=np.array(
                    loader.infos, dtype=h5py.string_dtype(encoding="utf-8")
                ),
                compression="gzip",
            )
        if "input/samples" in fh:
            assert loader.bcf_in.header.samples == fh["input/samples"]
        else:
            fh.create_dataset(
                "input/samples",
                data=np.array(
                    loader.bcf_in.header.samples,
                    dtype=h5py.string_dtype(encoding="utf-8"),
                ),
                compression="gzip",
            )
        if store_gls:
            fh.create_dataset("input/gls", data=gls, compression="gzip")


def store_mutation_tree_results(gls, out_prefix, output_store, runner):
    import h5py
    import networkx as nx
    import numpy as np

    from pigglet.likelihoods import MutationTreeLikelihoodCalculator

    calc = MutationTreeLikelihoodCalculator(runner.map_g, gls)
    map_tree_mut_probs = calc.mutation_probabilites(
        runner.agg.normalized_attachment_probabilities()
    )
    map_tree_map_attachments = calc.ml_sample_attachments() - 1
    with h5py.File(output_store, mode="a") as fh:
        fh.create_dataset(
            "map_tree/mutation_probabilities",
            data=map_tree_mut_probs,
            compression="gzip",
        )
        fh.create_dataset(
            "map_tree/map_sample_attachments",
            data=map_tree_map_attachments,
            compression="gzip",
        )
        fh.create_dataset(
            "map_tree/edge_list",
            data=np.array(
                [edge[0:2] for edge in nx.to_edgelist(runner.map_g)]
            ),
        )
    output_graph = out_prefix + ".map_tree.gml"
    nx.write_gml(runner.map_g, output_graph)


def store_phylo_tree_results(out_prefix, output_store, runner):
    import h5py
    import networkx as nx
    import numpy as np

    from pigglet.tree import strip_tree
    from pigglet.aggregator import NullAttachmentAggregator

    with h5py.File(output_store, mode="a") as fh:
        if not isinstance(runner.agg, NullAttachmentAggregator):
            fh.create_dataset(
                "phylo_tree/mutation_probabilities",
                data=runner.agg.averaged_mutation_probabilities(),
                compression="gzip",
            )
        fh.create_dataset(
            "map_phylo_tree/edge_list",
            data=np.array(
                [edge[0:2] for edge in nx.to_edgelist(runner.map_g)]
            ),
        )
        for idx, g in enumerate(runner.tree_aggregator.trees):
            fh.create_dataset(
                f"phylo_tree/samples/{idx}",
                data=np.array([edge[0:2] for edge in nx.to_edgelist(g)]),
            )
    output_graph = out_prefix + ".map_tree.gml"
    nx.write_gml(strip_tree(runner.map_g), output_graph)
    with open(out_prefix + ".t", "w") as fh:
        nw_trees = runner.tree_aggregator.to_newick()
        fh.write("\n".join(nw_trees) + "\n")
