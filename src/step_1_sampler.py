import collections

form collections import Counter

class AdaptiveSampler:
    def __init__(self, llm_engine, config):
        self.llm = llm_engine
        self.min_k = config['adaptive_sampling']['min_samples']
        self.max_k = config['adaptive_sampling']['max_samples']
        self.threshold = config['adaptive_sampling']['consistency_threshold']

    def sample(self, prompt):
        """
        Sinh các lời giải (paths) theo cơ chế thích ứng.
        """
        samples = []

        # 1. Sinh lô đầu tiên (min_k)
        initial_batch = self.llm.generate(prompt, num_return_sequences=self.min_k)
        samples.extend(initial_batch)

        # 2. Vòng lặp thích ứng
        while len(samples) < self.max_k:
            # Trích xuất đáp án cuối (final anser) để check hội tụ
            answers = [self.extract_answer(s) for s in samples]
            if not answers: break

            # Tính độ nhất quán (consistency score)
            most_common, count = Counter(answers).most_common(1)[0]
            consistency_score = count / len(samples)

            # Điều kiện dừng sớm
            if consistency_score >= self.threshold:
                print(f"--> Converged early with score {consistency_score}")
                break
                
            # nếu chưa đủ tự tin, sinh thêm 1 mẫu nữa/ batch nhỏ
            new_sample = self.llm.generate(prompt, num_return_sequences=1)
            samples.extend(new_sample)

        return samples
    
    def extract_answer(self, text):
        # Logic Regex để lấy text trong \boxed{}
        # (Implement chi tiết sau)
        return "temp_answer"
    
