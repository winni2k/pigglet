import h5py
import networkx as nx
from pigglet.aggregator import tree_to_newick


def phylo_nexus_impl(h5_file, phylo_nexus):

    nexus_out = []
    with h5py.File(h5_file, "r") as fh:
        taxa = list(fh["input/samples"])
        nexus_out.append(
            f"#NEXUS\n"
            f"Begin TAXA;\n"
            f"  Dimensions ntax={len(taxa)};\n"
            f"  TaxLabels {' '.join(taxa)};\n"
            "End;\n"
            "BEGIN TREES;\n"
        )
        for idx, nw in enumerate(get_newicks(fh, "phylo_tree/samples")):
            nexus_out.append(f"Tree tree{idx} = {nw};\n")
        nexus_out.append("END;\n")
    with open(phylo_nexus, "w") as fh:
        fh.writelines(nexus_out)


def get_newicks(h5_fh, trees: str):
    for sample in list(h5_fh[trees]):
        g = nx.DiGraph(list(h5_fh[f"{trees}/{sample}"]))
        yield tree_to_newick(g, one_base=True)


def phylo_newicks_impl(h5_file, newicks):
    with h5py.File(h5_file, "r") as fh5:
        with open(newicks, "w") as fh:
            for nw in get_newicks(fh5, "phylo_tree/samples"):
                fh.write(f"{nw};\n")
