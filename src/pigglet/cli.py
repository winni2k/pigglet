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


@click.group()
@click.version_option()
def cli():
    pass


@cli.command()
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
    default=False,
    help="Store the input GLs in the output h5 file."
    " Probably only useful for debugging purposes",
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
):
    """Infer phylogenetic or mutation tree from genotype likelihoods stored in
    GL_VCF.

    Save the resulting tree and mutation probabilities to OUT_PREFIX.

    Mutations and samples are ordered in the output according to their order
    in GL_VCF.
    """

    configure_logger(log_level=log_level, log_file=out_prefix + ".log")
    import logging

    from pigglet.gl_loader import LikelihoodLoader
    from pigglet.mcmc import MCMCRunner

    version = get_version()
    logging.info(f"The PIGGLET v{version}")

    logging.info("Loading GLs in %s", gl_vcf)
    loader = LikelihoodLoader(vcf_file=gl_vcf)
    gls = loader.load()

    logging.info("Storing input")
    output_store = out_prefix + ".h5"
    store_input(gls, loader, output_store, store_gls)
    del loader

    logging.info("Loaded %s sites and %s samples", gls.shape[0], gls.shape[1])
    logging.info(
        "Running MCMC with %s burnin and %s sampling iterations",
        burnin,
        sampling,
    )
    if mutation_tree:
        runner = MCMCRunner.mutation_tree_from_gls(gls)
    else:
        runner = MCMCRunner.phylogenetic_tree_from_gls(gls)

    runner.num_burnin_iter = burnin
    runner.num_sampling_iter = sampling
    runner.reporting_interval = reporting_interval
    runner.mover.calc.summer.check_calc = check_logsumexp_accuracy
    runner.mover.calc.summer.max_diffs = logsumexp_refresh_rate
    runner.run()

    logging.info("Storing results")
    store_results(gls, out_prefix, output_store, runner)


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

    from pigglet.tree import strip_tree
    from pigglet.tree_converter import PhylogeneticTreeConverter

    g = nx.read_gml(mutation_tree)
    g = nx.relabel_nodes(g, int)
    converter = PhylogeneticTreeConverter(g)
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


def get_version():
    import pkg_resources  # part of setuptools

    return pkg_resources.require("pigglet")[0].version


def store_input(gls, loader, output_store, store_gls):
    import h5py
    import numpy as np

    with h5py.File(output_store, "w") as fh:
        fh.create_dataset(
            "input/site_info",
            data=np.array(
                loader.infos, dtype=h5py.string_dtype(encoding="utf-8")
            ),
            compression="gzip",
        )
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


def store_results(gls, out_prefix, output_store, runner):
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
