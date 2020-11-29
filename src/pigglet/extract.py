import h5py
import networkx as nx
import newick

from pigglet.aggregator import tree_to_newick

from dataclasses import dataclass


def get_input_samples(h5_fh):
    return [s.decode() for s in h5_fh["input/samples"]]


@dataclass
class NewickTreeConverter:
    h5_file: str
    label_leaves: bool = False
    branch_lengths: bool = False

    def to_phylo_nexus(self, phylo_nexus):
        """Convert posterior phylogenetic tree samples to NEXUS format"""
        nexus_out = []
        with h5py.File(self.h5_file, "r") as h5_fh:
            taxa = get_input_samples(h5_fh)
            nexus_out.append(
                f"#NEXUS\n"
                f"Begin TAXA;\n"
                f"  Dimensions ntax={len(taxa)};\n"
                f"  TaxLabels {' '.join(taxa)};\n"
                "End;\n"
                "BEGIN TREES;\n"
            )
            for idx, nw in enumerate(self._get_newicks(h5_fh)):
                nexus_out.append(f"Tree tree{idx} = {nw};\n")
            nexus_out.append("END;\n")
        phylo_nexus.writelines(nexus_out)

    def to_phylo_newicks(self, newicks):
        """Convert posterior phylogenetic tree samples
        to list of NEWICK trees."""
        with h5py.File(self.h5_file, "r") as fh5:
            for nw in self._get_newicks(fh5):
                newicks.write(f"{nw};\n")

    def to_phylo_map_newick(self, newick):
        """Convert MAP phylogenetic tree to NEWICK tree"""
        with h5py.File(self.h5_file, "r") as fh5:
            nw = self._get_newick(fh5)
            newick.write(f"{nw};\n")

    def _get_newicks(
        self,
        h5_fh,
        trees="phylo_tree/samples",
        br_lens="phylo_tree/samples_br_lens",
    ):
        leaf_labels = node_branch_lengths = None
        if self.label_leaves:
            leaf_labels = get_input_samples(h5_fh)
        for sample in list(h5_fh[trees]):
            if self.branch_lengths:
                node_branch_lengths = list(h5_fh[f"{br_lens}/{sample}"])
            g = nx.DiGraph(list(h5_fh[f"{trees}/{sample}"]))
            yield tree_to_newick(
                g,
                one_base=True,
                leaf_lookup=leaf_labels,
                node_branch_length_lookup=node_branch_lengths,
            )

    def _get_newick(
        self,
        h5_fh,
        tree="map_phylo_tree/edge_list",
        br_lens="map_phylo_tree/br_lens",
    ):
        leaf_labels = node_branch_lengths = None
        if self.label_leaves:
            leaf_labels = get_input_samples(h5_fh)
        if self.branch_lengths:
            node_branch_lengths = list(h5_fh[br_lens])
        g = nx.DiGraph(list(h5_fh[tree]))
        return tree_to_newick(
            g,
            one_base=True,
            leaf_lookup=leaf_labels,
            node_branch_length_lookup=node_branch_lengths,
        )


def extract_newick_file_as_digraph(newick_tree_file, zero_based=False):
    for t in newick.load(newick_tree_file):
        g = nx.DiGraph()
        node_names = {}
        for idx, node in enumerate(t.walk()):
            if node.name is None:
                node_names[id(node)] = -1
            else:
                node_id = int(node.name)
                node_names[id(node)] = node_id
            for dec in node.descendants:
                g.add_edge(id(node), id(dec))
        first_leaf_id = sorted(set(node_names.values()))[1]
        if zero_based:
            assert first_leaf_id == 0
        else:
            assert first_leaf_id == 1
        last_leaf_id = max(node_names.values())
        num_missing_nodes = sum(
            1 if v == -1 else 0 for v in node_names.values()
        )
        assert len(g.nodes) == last_leaf_id + num_missing_nodes + int(
            zero_based
        ), "Assuming leaf nodes have ids 1:n_leaves"
        next_node_id = last_leaf_id + 1
        for n in g:
            if node_names[n] == -1:
                node_names[n] = next_node_id
                next_node_id += 1
        if not zero_based:
            node_names = {k: v - 1 for k, v in node_names.items()}
        g = nx.relabel_nodes(g, node_names)
        yield g
