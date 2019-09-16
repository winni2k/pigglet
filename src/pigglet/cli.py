import click


@click.command()
@click.version_option()
@click.argument('gl_vcf', type=click.Path(exists=True))
@click.argument('out_prefix', type=click.Path())
@click.option('-b', '--burnin', type=int, default=10, show_default=True,
              help='Number of burn-in iterations to run')
@click.option('-s', '--sampling', type=int, default=10, show_default=True,
              help='Number of sampling iterations to run')
@click.option('--normal', 'log_level', flag_value='INFO', default=True)
@click.option('--silent', 'log_level', flag_value='WARNING')
@click.option('--verbose', 'log_level', flag_value='DEBUG')
@click.option('--reporting-interval', type=int, default=1000, show_default=True,
              help='Report MCMC progress after this number of iterations')
@click.option('--store-gls/--no-store-gls', default=False,
              help='Store the input GLs in the output h5 file.'
                   ' Probably only useful for debugging purposes')
def cli(gl_vcf, out_prefix, burnin, sampling, log_level, reporting_interval, store_gls):
    """Impute mutation tree from genotype likelihoods stored in GL_VCF and save the
    resulting tree and mutation probabilities to OUT_PREFIX.

    Mutations and samples are ordered in the output according to their order in GL_VCF.
    """

    import logging
    logging.basicConfig(level=getattr(logging, log_level),
                        format='%(process)d-pigglet-%(levelname)s: %(message)s')

    from pigglet.mcmc import MCMCRunner
    from pigglet.gl_loader import LikelihoodLoader

    version = get_version()
    logging.info(f'The PIGGLET v{version}')

    logging.info('Loading GLs in %s', gl_vcf)
    loader = LikelihoodLoader(vcf_file=gl_vcf)
    gls = loader.load()

    logging.info('Storing input')
    output_store = out_prefix + '.h5'
    store_input(gls, loader, output_store, store_gls)
    del loader

    logging.info('Loaded %s sites and %s samples', gls.shape[0], gls.shape[1])

    logging.info('Running MCMC with %s burnin and %s sampling iterations', burnin,
                 sampling)
    runner = MCMCRunner.from_gls(gls=gls,
                                 num_burnin_iter=burnin,
                                 num_sampling_iter=sampling,
                                 reporting_interval=reporting_interval)
    runner.run()

    logging.info('Storing results')
    store_results(gls, out_prefix, output_store, runner)


def get_version():
    import pkg_resources  # part of setuptools
    return pkg_resources.require("pigglet")[0].version


def store_input(gls, loader, output_store, store_gls):
    import h5py
    import numpy as np

    with h5py.File(output_store, 'w') as fh:
        fh.create_dataset('input/site_info',
                          data=np.array(loader.infos,
                                        dtype=h5py.string_dtype(encoding='utf-8')),
                          compression="gzip")
        fh.create_dataset('input/samples',
                          data=np.array(loader.bcf_in.header.samples,
                                        dtype=h5py.string_dtype(encoding='utf-8')),
                          compression="gzip")
        if store_gls:
            fh.create_dataset('input/gls',
                              data=gls,
                              compression="gzip")


def store_results(gls, out_prefix, output_store, runner):
    import networkx as nx
    from pigglet.likelihoods import TreeLikelihoodCalculator
    import h5py
    import numpy as np

    calc = TreeLikelihoodCalculator(runner.map_g, gls)
    map_tree_mut_probs = calc \
        .mutation_probabilites(runner.agg.normalized_attachment_probabilities())
    map_tree_map_attachments = calc.ml_sample_attachments() - 1
    with h5py.File(output_store, mode='a') as fh:
        fh.create_dataset('map_tree/mutation_probabilities',
                          data=map_tree_mut_probs,
                          compression="gzip")
        fh.create_dataset('map_tree/map_sample_attachments',
                          data=map_tree_map_attachments,
                          compression="gzip")
        fh.create_dataset('map_tree/edge_list',
                          data=np.array(
                              [edge[0:2] for edge in nx.to_edgelist(runner.map_g)]
                          ))
    output_graph = out_prefix + '.map_tree.gml'
    nx.write_gml(runner.map_g, output_graph)
