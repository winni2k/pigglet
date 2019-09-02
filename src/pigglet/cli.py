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
    gls = LikelihoodLoader(vcf_file=gl_vcf).load()
    runner = MCMCRunner.from_gls(gls=gls)
    runner.run()

    output_graph = out_prefix + '.gml'
    nx.write_gml(runner.g, output_graph)
