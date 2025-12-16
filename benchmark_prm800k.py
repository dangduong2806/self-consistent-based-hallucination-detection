from src.metrics import DeepMathMetrics
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import logging
import yaml
import time
import torch

from models.llm_engine import LLMEngine
from src.step_1_sampler import AdaptiveSampler
from src.step_2_verifier import LocalVerifier
# from src.step_3_graph_builder import ReasoningGraphBuilder
from src.step_4_graph import ReasoningGraph
from src.step_4_structural_verifier import StructuralVerifier
from src.step_5_selector import EntropySelector

logger = logging.getLogger(__name__)

def load_prm800k_test_set(file_path, limit=50):
    """
    Load dataset PRM800K. 
    File thường định dạng jsonl. Mỗi dòng là 1 json object.
    """

    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= limit: break
            item = json.loads(line)
            # PRM800K format mapping (tuỳ chỉnh theo file thật bạn tải)
            
            question = item.get('question', {})
            data.append({
                'problem': question.get('problem', ''),
                'ground_truth': question.get('ground_truth_answer', '') # Hoặc final_answer
            })
    return data

def run_benchmark():
    # 1. Setup
    config_path = "configs/main_config.yaml"
    # Load Config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Load model
    logger.info("Initializing Llama-3-8B (4-bit)...")
    llm = LLMEngine(config['model']['name'])
    
    # 3. Initialize Pipeline Components
    sampler = AdaptiveSampler(llm, config)
    local_verifier = LocalVerifier(config)
    graph_builder = ReasoningGraph() # Có chứa IsomorphismEngine bên trong
    struct_verifier = StructuralVerifier(config)
    selector = EntropySelector()

    math_eval = DeepMathMetrics()
    
    # Load Data (Thay đường dẫn file thật của bạn vào đây)
    data_path = "data/raw/phase1_test.jsonl"
    test_data = load_prm800k_test_set(data_path, limit=20) 
    
    results = []
    
    print(f"Starting benchmark on {len(test_data)} samples...")
    for item in tqdm(test_data):
        problem = item['problem']
        gt = item['ground_truth']

        try:
            start_time = time.time()
            torch.cuda.reset_peak_memory_stats()

            # Gọi pipeline
            logger.info(">>> Step 1: Adaptive Sampling...")
            raw_paths = sampler.sample(problem)
            sample_count = len(raw_paths)
            logger.info(">>> Step 2: Atomic & Logical Verification (SymPy + Logprobs)...")
            verified_paths = []
            valid_path_count = 0
            
            for idx, path in enumerate(raw_paths):
                # path là list các object/dict chứa text và logprobs của từng bước
                verified_steps = local_verifier.verify_path(path)
                
                # Chỉ giữ lại các path có ít nhất 1 bước đúng
                if verified_steps:
                    verified_paths.append(verified_steps)
                    valid_path_count += 1
                else:
                    logger.debug(f"    Path {idx} rejected completely.")
            
            logger.info(f"    Retained {valid_path_count} valid paths after local filtering.")

            if not verified_paths:
                logger.warning("!!! No valid paths found. Pipeline aborted.")
                return None
            
            # ---------------------------------------------------------
            # BƯỚC 3: Graph Construction (Isomorphism Isomorphism)
            # ---------------------------------------------------------
            logger.info(">>> Step 3: Building Reasoning Graph (SymPy Isomorphism)...")
            raw_graph = graph_builder.build_graph(verified_paths)
            logger.info(f"    Graph built with {raw_graph.number_of_nodes()} nodes and {raw_graph.number_of_edges()} edges.")

            # ---------------------------------------------------------
            # BƯỚC 4: Structural Verification (Global Dependency)
            # ---------------------------------------------------------
            logger.info(">>> Step 4: Structural Verification (Centrality Reweighting)...")
            refined_graph = struct_verifier.verify_structure(raw_graph)

            # Debug: In ra một vài node quan trọng
            top_nodes = sorted(refined_graph.nodes(data=True), key=lambda x: x[1].get('final_score', 0), reverse=True)[:3]
            logger.debug(f"    Top robust nodes: {[n[1].get('content') for n in top_nodes]}")

            # ---------------------------------------------------------
            # BƯỚC 5: Global Selection (Entropy Minimization)
            # ---------------------------------------------------------
            logger.info(">>> Step 5: Final Selection (Entropy Minimization)...")
            result = selector.select_answer(refined_graph)

            end_time = time.time()
            peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2) # MB

            pred = result['final_answer']
            best_path_text = result['final_path_content']
            metrics = math_eval.compute_all_metrics(
                generated_path_text= best_path_text,
                ground_truth_value_str= gt
            )
            print(f"EE: {metrics['EE']}, ASS: {metrics['ASS']}, TSA: {metrics['TSA']}")
            record = {
                "problem": problem[:30] + "...",
                "EE": metrics['EE'],
                "ASS": metrics['ASS'],
                "TSA": metrics['TSA'],
                "SE": sample_count,
                "CO": end_time - start_time,
                "MF": peak_memory
            }
            results.append(record)
        except Exception as e:
            print("❌ Failed to generate a valid answer.")
            
    # 3. Aggregation (Tính trung bình)
    df = pd.DataFrame(results)
    print("FINAL BENCHMARK RESULTS (PRM800K)")
    print("="*50)
    print(f"1. Expression Equivalence (EE - Accuracy): {df['EE'].mean():.4f}")
    print(f"2. Algebraic Simplification (ASS):         {df['ASS'].mean():.4f}")
    print(f"3. Transformation Step Acc (TSA):          {df['TSA'].mean():.4f}")
    print("-" * 30)
    print(f"4. Sample Efficiency (SE - Avg Samples):   {df['SE'].mean():.2f}")
    print(f"5. Comp Overhead (CO - Avg Time s):        {df['CO'].mean():.4f}")
    print(f"6. Memory Footprint (MF - Peak MB):        {df['MF'].max():.2f}") # Lấy Max thay vì Mean cho RAM
    print("="*50)

    # Lưu kết quả chi tiết
    # df.to_csv("benchmark_results.csv", index=False)
    # print("Detailed results saved to benchmark_results.csv")

if __name__ == "__main__":
    run_benchmark()

            


