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
def cli(gl_vcf, out_prefix, burnin, sampling, log_level):
    """Impute mutation tree from genotype likelihoods stored in GL_VCF and save the
    resulting tree and mutation probabilities to OUT_PREFIX.

    Mutations and samples are ordered in the output according to their order in GL_VCF.
    """

    import logging
    logging.basicConfig(level=getattr(logging, log_level),
                        format='%(process)d-pigglet-%(levelname)s: %(message)s')

    from pigglet.mcmc import MCMCRunner
    from pigglet.gl_loader import LikelihoodLoader
    import networkx as nx
    import h5py
    import numpy as np
    from pigglet.likelihoods import TreeLikelihoodCalculator

    logging.info('Loading GLs in %s', gl_vcf)
    gls = LikelihoodLoader(vcf_file=gl_vcf).load()
    logging.info('Loaded %s sites and %s samples', gls.shape[0], gls.shape[1])

    logging.info('Running MCMC with %s burnin and %s sampling iterations', burnin,
                 sampling)
    runner = MCMCRunner.from_gls(gls=gls,
                                 num_burnin_iter=burnin,
                                 num_sampling_iter=sampling)
    runner.run()

    logging.info('Storing results')
    calc = TreeLikelihoodCalculator(runner.map_g, gls)
    map_tree_mut_probs = calc \
        .mutation_probabilites(runner.agg.normalized_attachment_probabilities())
    map_tree_map_attachments = calc.ml_sample_attachments() - 1
    output_store = out_prefix + '.h5'
    with h5py.File(output_store, 'w') as fh:
        fh.create_dataset('map_tree/mutation_probabilities', data=map_tree_mut_probs,
                          compression="gzip")
        fh.create_dataset('map_tree/map_sample_attachments',
                          data=map_tree_map_attachments,
                          compression="gzip")
        fh.create_dataset('map_tree/edge_list', data=np.array(
            [edge[0:2] for edge in nx.to_edgelist(runner.map_g)]))

    output_graph = out_prefix + '.map_tree.gml'
    nx.write_gml(runner.map_g, output_graph)
