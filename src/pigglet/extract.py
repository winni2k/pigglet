import h5py
import networkx as nx
from pigglet.aggregator import tree_to_newick


def phylo_nexus_impl(h5_file, phylo_nexus, label_leaves=False):

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
        for idx, nw in enumerate(get_newicks(fh, label_leaves=label_leaves)):
            nexus_out.append(f"Tree tree{idx} = {nw};\n")
        nexus_out.append("END;\n")
    phylo_nexus.writelines(nexus_out)


def get_newicks(h5_fh, trees="phylo_tree/samples", label_leaves=False):
    leaf_labels = None
    if label_leaves:
        leaf_labels = list(h5_fh["input/samples"])
    for sample in list(h5_fh[trees]):
        g = nx.DiGraph(list(h5_fh[f"{trees}/{sample}"]))
        yield tree_to_newick(g, one_base=True, leaf_lookup=leaf_labels)


def phylo_newicks_impl(h5_file, newicks, label_leaves=False):
    with h5py.File(h5_file, "r") as fh5:
        for nw in get_newicks(fh5, label_leaves=label_leaves):
            newicks.write(f"{nw};\n")
