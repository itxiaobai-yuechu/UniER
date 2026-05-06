


import numpy as np
from EduSim.Envs.meta import ItemBase
from networkx import Graph, DiGraph

__all__ = ["KSSItemBase"]


class KSSItemBase(ItemBase):
    def __init__(self, knowledge_structure: (Graph, DiGraph), learning_order=None, items=None, seed=None,
                 reset_attributes=True):
        self.random_state = np.random.RandomState(seed)
        if items is None or reset_attributes:
            assert learning_order is not None
            _difficulties = list(
                sorted([self.random_state.randint(0, 5) for _ in range(len(knowledge_structure.nodes))])
            )
            difficulties = {}
            for i, node in enumerate(knowledge_structure.nodes):
                difficulties[node] = _difficulties[i]

            if items is None:
                items = [
                    {
                        "knowledge": node,
                        "attribute": {
                            "difficulty": difficulties[node]
                        }
                    } for node in knowledge_structure.nodes
                ]
            elif isinstance(items, list):
                for item in items:
                    item["attribute"] = {"difficulty": difficulties[item["knowledge"]]}
            elif isinstance(items, dict):
                for item in items.values():
                    item["attribute"] = {"difficulty": difficulties[item["knowledge"]]}
            else:
                raise TypeError()

        super(KSSItemBase, self).__init__(
            items, knowledge_structure=knowledge_structure,
        )
        self.knowledge2item = dict()
        for item in self.items:
            self.knowledge2item[item.knowledge] = item
