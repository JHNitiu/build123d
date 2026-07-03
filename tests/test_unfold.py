import unittest

from build123d.objects_part import Box
from build123d.topology import build_graph, _unfold
from build123d import BuildPart, BuildLine, Plane, Line, fillet, Side, make_brake_formed, export_stl, import_step
import networkx as nx
import matplotlib.pyplot as plt
from ocp_vscode import show

class UnfoldTests(unittest.TestCase):
    
    def test_build_graph(self):

        bend = import_step("/home/gretaspals/projects/sommarprojekt/Unfold/build123d/tests/FOLDED_CUT.stp")
        show(bend)
        reF_face = bend.faces()[1]
        show([bend, reF_face], colors=["yellow", "purple"])

        _unfold(bend, reF_face, 1.0) 