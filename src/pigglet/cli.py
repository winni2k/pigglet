import click


@click.command()
@click.version_option()
@click.argument('genotype_likelihood_VCF', type=click.Path(exists=True))
@click.argument('output_prefix', type=click.Path())
def cli(genotype_likelihood_vcf, output_prefix):
    """pigglet main entry point"""
    return run(genotype_likelihood_vcf, output_prefix)


def run(gl_vcf, out_prefix):
    from pigglet.mcmc import MCMCRunner
    from pigglet.gl_loader import LikelihoodLoader
    import networkx as nx
    import h5py
    import numpy as np
    from pigglet.likelihoods import TreeLikelihoodCalculator

    gls = LikelihoodLoader(vcf_file=gl_vcf).load()
    runner = MCMCRunner.from_gls(gls=gls)
    runner.run()

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
