import unittest

from build123d.objects_part import Box
from build123d.topology import build_graph
from build123d import BuildPart, BuildLine, Plane, Line, fillet, Side, make_brake_formed, export_stl, import_step
import networkx as nx
import matplotlib.pyplot as plt
from ocp_vscode import show

class UnfoldTests(unittest.TestCase):
    
    def test_build_graph(self):

        l_bend = import_step("/home/johan/Dev/b123d-fork/tests/L-bend.step")
        show(l_bend)
        graph = build_graph(l_bend, l_bend.faces()[0])
        nx.draw(graph, with_labels=True)
        plt.show()