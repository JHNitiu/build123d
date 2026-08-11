import unittest

from build123d.objects_part import Box
from build123d.topology import build_graph, _unfold
from build123d import BuildPart, BuildLine, Plane, Line, fillet, Side, make_brake_formed, export_stl, import_step
import networkx as nx
import matplotlib.pyplot as plt
from ocp_vscode import show

class UnfoldTests(unittest.TestCase):
    
    def test_build_graph(self):

        bend = import_step("/home/johan/Dev/b123d-fork/tests/item6.step")
        show(bend)
        reF_face = bend.faces()[9]
 
        _unfold(bend, reF_face, 1.0) 


    #41 på hat, 2 på U, 20 på folded cut, item1, item2, 3 på item3, 6 på item4, 5 på item5, 9 på item6

