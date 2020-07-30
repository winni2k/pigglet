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
    help="This is slow and only meant for debugging. ",
)
@click.option(
    "--defer-mutation-probability-calc/"
    "--no-defer-mutation-probability-calc",
    default=False,
    help="Don't calculate mutation probabilities while sampling trees. "
    "Mutation probabilities can later be estimated in parallel using "
    "`pigglet calc`",
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
        )
    runner.num_burnin_iter = burnin
    runner.num_sampling_iter = sampling
    runner.reporting_interval = reporting_interval
    runner.mover.calc.summer.check_calc = check_logsumexp_accuracy
    runner.mover.calc.summer.max_diffs = logsumexp_refresh_rate
    if mutation_tree:
        logger.info("Using a mutation tree")
    else:
        runner.tree_aggregator = TreeAggregator()
        if defer_mutation_probability_calc:
            runner.agg = NullAttachmentAggregator()
        logger.info("Using a phylogenetic tree")
    runner.run()

    logger.info("Storing results")
    if mutation_tree:
        store_mutation_tree_results(gls, out_prefix, output_store, runner)
    else:
        store_phylo_tree_results(out_prefix, output_store, runner)


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
    from pigglet.tree_converter import PhylogeneticTreeConverter

    g = nx.read_gml(mutation_tree)
    g = nx.relabel_nodes(g, int)
    converter = PhylogeneticTreeConverter(g, prng=random)
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


def calc_phylo_mutation_probs_for_tree(
    h5_file, shared_gls_name, gls_shape, gls_dtype, tree_idx
):
    import h5py
    import numpy as np
    from pigglet.aggregator import PhyloAttachmentAggregator
    from pigglet.likelihoods import PhyloTreeLikelihoodCalculator
    from pigglet.tree_interactor import GraphAnnotator

    import networkx as nx
    from multiprocessing import shared_memory

    existing_shm = shared_memory.SharedMemory(name=shared_gls_name)
    gls = np.ndarray(gls_shape, dtype=gls_dtype, buffer=existing_shm.buf)
    with h5py.File(h5_file, "r") as fh:
        g = nx.DiGraph(list(fh[f"phylo_tree/samples/{tree_idx}"]))
    GraphAnnotator(g).annotate_all_nodes_with_descendant_leaves()
    agg = PhyloAttachmentAggregator()
    agg.add_attachment_log_likes(PhyloTreeLikelihoodCalculator(g, gls))
    return agg.averaged_mutation_probabilities()


@cli.command()
@click.argument("h5_file")
@click.option(
    "--jobs",
    type=int,
    default=None,
    help="Number of processes to parallelize over",
)
def calc_phylo_mutation_probs(h5_file, jobs=1):
    """Calculate mutation probabilities for posterior phylogenetic tree samples

    H5_FILE is the output file of pigglet infer ending in .h5
    """

    import h5py
    import numpy as np
    import math
    from tqdm import tqdm
    from multiprocessing import Pool, shared_memory
    import functools as ft
    from pigglet.scipy_import import logsumexp

    with h5py.File(h5_file, "r") as fh:
        n_sampling_iterations = len(list(fh["phylo_tree/samples"]))
        if "input/gls" not in fh:
            raise Exception("Please run pigglet infer with --store-gls.")
        n_sites, n_samples = (
            fh["input/site_info"].shape[0],
            fh["input/samples"].shape[0],
        )
        gls = fh["input/gls"][:]
    shm = shared_memory.SharedMemory(create=True, size=gls.nbytes)
    shared_gls = np.ndarray(gls.shape, dtype=gls.dtype, buffer=shm.buf)
    shared_gls[:] = gls[:]
    func = ft.partial(
        calc_phylo_mutation_probs_for_tree,
        h5_file,
        shm.name,
        shared_gls.shape,
        shared_gls.dtype,
    )
    attachment_probs = np.zeros(
        shape=(n_sampling_iterations, n_sites, n_samples)
    )
    with Pool(jobs) as p:
        for idx, attach_probs in enumerate(
            tqdm(
                p.imap_unordered(func, range(n_sampling_iterations)),
                unit="trees",
                desc="Mutation probabilities",
                total=n_sampling_iterations,
            )
        ):
            attachment_probs[idx] = attach_probs
    sum_ll = logsumexp(np.array(attachment_probs), 0) - math.log(
        n_sampling_iterations
    )
    with h5py.File(h5_file, "r+") as fh:
        fh.create_dataset(
            "phylo_tree/mutation_probabilities",
            data=sum_ll,
            compression="gzip",
        )


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
