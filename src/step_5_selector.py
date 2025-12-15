import numpy as np
import networkx as nx
from .step_3_graph_builder import IsomorphismEngine # Tái sử dụng engine so sánh

class EntropySelector:
    def __init__(self):
        self.iso_engine = IsomorphismEngine()
    
    def select_answer(self, graph):
        """
        Input: Reasoning Graph đã được verify và re-weight (có 'final_score').
        Output: (best_answer_content, confidence_score, entropy)
        """
        # 1. Tìm tất cả các node lá (Leaf nodes - Out degree = 0)
        leaf_nodes = [n for n in graph.nodes() if graph.out_degree(n) == 0]

        if not leaf_nodes:
            return None, 0.0, float('inf')
        
        # 2. Gom nhóm các đáp án tương đương (Grouping Answer Clusters)
        # Structure: { "canonical_answer_text": { "score": float, "original_texts": [] } }

        clusters = {}
        for node in leaf_nodes:
            content = graph.nodes[node].get('content', '')
            # Lấy điểm số đã được tính từ bước Structural Verification
            # Đây chính là thành phần cốt lõi của công thức SC(global)
            score = graph.nodes[node].get('final_score', 0.0)

            # Kiểm tra xem đáp án này đã có trong cluster nào chưa
            found_cluster = False
            for key in clusters.keys():
                if self.iso_engine.are_equivalent(content, key):
                    # Nếu tương đương, gộp vào cluster đó
                    clusters[key]['score'] += score
                    clusters[key]['original_texts'].append(content)
                    found_cluster = True
                    break
            if not found_cluster:
                # Tạo cluster mới
                clusters[content] = {
                    'score': score,
                    'original_texts': [content]
                }
        # 3. Chuẩn hóa thành xác suất (Probability Distribution)
        # P(a) = SC(a) / Sum(SC(all))
        candidates = []
        total_sc_score = sum(c['score'] for c in clusters.values())
        if total_sc_score == 0:
            return None, 0.0, float('inf')
        
        for answer, data in clusters.items():
            prob = data['score'] / total_sc_score
            candidates.append({
                'answer': answer,
                'prob': prob,
                'raw_score': data['score']
            })

        # 4. Tính entropy của phân phối này
        # H(X) = - sum(p * log(p))
        probs = np.array([c['prob'] for c in candidates])
        # Cộng thêm epsilon để tránh log(0)
        entropy = -np.sum(probs * np.log(probs + 1e-10))

        # 5. Chọn đáp án tốt nhất (Max Probability = Minimized Risk)
        best_candidate = max(candidates, key=lambda x: x['prob'])

        return {
            'final_anser': best_candidate['answer'],
            'confidence': best_candidate['prob'],
            'entropy': entropy,
            'all_candidates': candidates
        }