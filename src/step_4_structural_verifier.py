import networkx as nx
import numpy as np

class StructuralVerifier:
    def __init__(self, graph):
        self.graph = graph  # ReasoningGraph instance
    
    def verify_structure(self):
        """
        Structural Verification:
        Tinh chỉnh lại trọng số của các node dựa trên cấu trúc toàn cục.
        """
        # 1. Tính toán PageRank hoặc Centrality
        # Trong paper họ dùng Laplacian relaxation, nhưng PageRank là một approximation tốt
        # để tìm ra các "trụ cột" trong luồng suy luận.
        centrality_scores = nx.pagerank(self.graph, weight='weight')

        for node in self.graph.nodes():
            # Lấy điểm cục bộ (từ Atomic/Logical step)
            local_conf = self.graph.nodes[node].get('total_confidence', 0)
            count = self.graph.nodes[node].get('count', 1)

            # 2. Kết hợp với Structural Score (Centrality)
            # Một node đúng thường nằm trên các luồng chính (centrality cao)
            struct_score = centrality_scores.get(node, 0)

            # Cập nhật lại trọng số node
            # Công thức lai ghép: Local * Structural
            final_node_weight = (local_conf /count) * (1 + struct_score)

            self.graph.nodes[node]['final_score'] = final_node_weight

        return self.graph