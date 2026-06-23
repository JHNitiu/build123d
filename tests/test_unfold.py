import unittest

from build123d.objects_part import Box
from build123d.topology import build_graph, _unfold_2
from build123d import BuildPart, BuildLine, Plane, Line, fillet, Side, make_brake_formed, export_stl, import_step
import networkx as nx
import matplotlib.pyplot as plt
from ocp_vscode import show

class UnfoldTests(unittest.TestCase):
    
    def test_build_graph(self):

        l_bend = import_step("/home/johan/Dev/b123d-fork/tests/FOLDED_CUT.stp")
        show(l_bend)
        reF_face = l_bend.faces()[0]

        _unfold_2(l_bend, reF_face, 1.0)
    
    
        graph = build_graph(l_bend, reF_face)
        node_labels = nx.get_node_attributes(graph, "type")

        
        nx.draw(graph, labels=node_labels, with_labels=True)
        plt.show()