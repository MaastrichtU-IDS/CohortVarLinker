import networkx as nx
from src.omop_graph import OmopGraphNX



def make_graph():
    g = nx.DiGraph()
    g.add_edge(2, 1, relation="is a")
    g.add_edge(3, 2, relation="is a")
    g.add_edge(1, 2, relation="subsumes")
    g.add_edge(2, 3, relation="subsumes")
    obj = OmopGraphNX.__new__(OmopGraphNX)
    obj.graph = g
    return obj


def test_bfs_upward_reachable():
    g = make_graph()
    assert g.bfs_upward_reachable(3, [1], max_depth=3) == [1]


def test_bfs_downward_reachable():
    g = make_graph()
    assert g.bfs_downward_reachable(1, [3], max_depth=3) == [3]


def test_bfs_bidirectional_reachable():
    g = make_graph()
    assert g.bfs_bidirectional_reachable(3, [1], max_depth=3) == [1]


def test_only_upward_or_downward():
    g = make_graph()
    result = g.only_upward_or_downward(3, {1}, max_depth=3)
    assert result == {1}