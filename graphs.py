from graphviz import Digraph
from transliterate import translit

from evaluation.labels import Label, PositiveLabel, NegativeLabel
from source.synonyms import SynonymsCollection
from runtime.relations import RelationCollection


class CollectionGraph:

    def __init__(self):
        self.graph = {}
        self.dot = None

    def add_edge(self, v1, v2, label):
        assert(isinstance(label, Label))
        if v1 not in self.graph:
            self.graph[v1] = []
        self.graph[v1].append((v2, label))


    def get_edge_label(self, v1, v2):
        assert(type(v1) == int and v1 in self.graph)
        assert(type(v2) == int)
        edges = self.graph[v1]
        for u, label in edges:
            if u == v2:
                return label
        return None


    def has_edge(self, v1, v2, expected_label=None):
        assert(type(self.graph) == dict)
        assert(expected_label is None or isinstance(expected_label, Label))

        if v1 not in self.graph:
            return False

        edge_label = self.get_edge_label(v1, v2)
        if edge_label is None:
            return False

        if expected_label is not None:
            return edge_label == expected_label

        return True


    @classmethod
    def create_graph_by_opinions(cls, triplets, synonyms,
                                 default_color='blue', dot_format='dot'):
        """
        Create Grpaph from list of opinions collections

        triplets: list of (news, opinion_collections, colorize)
        """
        assert(isinstance(triplets, list))
        assert(isinstance(synonyms, SynonymsCollection))

        def add_edge(graph_collection, v1, v2, label, color=None,
                     edges_count=0, style='solid', add=False, colorize=True,
                     avoid_existed=True, display_edge_label=False):
            assert(isinstance(graph_collection, CollectionGraph))
            assert(isinstance(v1, int))
            assert(isinstance(v2, int))
            assert(isinstance(color, str) or color is None)
            assert(isinstance(label, Label))

            if ((graph_collection.has_edge(v1, v2) or
                 graph_collection.has_edge(v2, v1)) and avoid_existed):
                return

            if color is None:
                color = default_color
                if colorize:
                    if label == PositiveLabel():
                        color = 'green'
                    elif label == NegativeLabel():
                        color = 'red'

            v_l = synonyms.get_group_by_index(v1)[0]
            v_r = synonyms.get_group_by_index(v2)[0]

            dot.edge(translit(v_l, "ru", reversed=True),
                     translit(v_r, "ru", reversed=True),
                     color=color,
                     label='' if edges_count == 0 or not display_edge_label else str(edges_count),
                     style=style)

            graph_collection.add_edge(v1, v2, label)
            vertices.add(v1)
            vertices.add(v2)

            # if not add:
            #     return

            # o = Opinion(value_by_vertex[v1], value_by_vertex[v2], label)
            # # if not constraint_opinions.has_opinion_by_values(o):
            # constraint_opinions.add_opinion(o)


        dot = Digraph()
        vertices = set()
        graph = CollectionGraph()
        value_by_vertex = {}

        for news, opinion_collections, colorize in triplets:
            for opinions in opinion_collections:
                for o in opinions:
                    relations = RelationCollection.from_news_opinion(news, o, synonyms)

                    if len(relations) == 0:
                        continue

                    left_v = relations[0].get_left_entity_value()
                    right_v = relations[0].get_right_entity_value()

                    if not synonyms.has_synonym(left_v):
                        synonyms.add_synonym(left_v)

                    if not synonyms.has_synonym(right_v):
                        synonyms.add_synonym(right_v)

                    left_node_id = synonyms.get_synonym_group_index(left_v)
                    right_node_id = synonyms.get_synonym_group_index(right_v)

                    value_left_node = synonyms.get_group_by_index(left_node_id)[0]
                    value_right_node = synonyms.get_group_by_index(right_node_id)[0]

                    assert(isinstance(value_left_node, unicode))
                    assert(isinstance(value_right_node, unicode))

                    add_edge(graph, left_node_id, right_node_id, o.sentiment,
                             edges_count=len(relations), colorize=colorize)

                    if left_node_id not in value_by_vertex:
                        value_by_vertex[left_node_id] = o.value_left
                    if right_node_id not in value_by_vertex:
                        value_by_vertex[right_node_id] = o.value_right

        graph.dot = dot
        return graph
