import networkx as nx
from typing import List, Dict, Any, Optional

class GraphMemory:
    """
    Lightweight Graph Memory for Phase C using NetworkX.
    Stores entities as nodes and relationships as edges.
    """
    def __init__(self):
        self.graph = nx.DiGraph()
        
    def add_fact(self, subject: str, relation: str, object_: str, metadata: Optional[Dict] = None):
        """
        Adds a knowledge triple to the graph.
        """
        if metadata is None:
            metadata = {}
            
        self.graph.add_node(subject)
        self.graph.add_node(object_)
        self.graph.add_edge(subject, object_, relation=relation, **metadata)
        
    def query(self, entity: str, hop: int = 1) -> List[Dict[str, Any]]:
        """
        Retrieves local neighborhood for an entity up to `hop` steps.
        """
        if not self.graph.has_node(entity):
            return []
            
        results = []
        # Breadth-first search up to depth `hop`
        edges = nx.bfs_edges(self.graph, source=entity, depth_limit=hop)
        for u, v in edges:
            edge_data = self.graph.get_edge_data(u, v)
            results.append({
                "subject": u,
                "relation": edge_data.get("relation", "unknown"),
                "object": v
            })
            
        return results
