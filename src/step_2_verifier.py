import numpy as np
from sympy import sympify, SympifyError
class LocalVerifier:
    def verify_step(self, context, step_content, step_logprob):
        """
        Kết hợp Atomic và Logical check.
        """
        # 1. Atomic Check (SymPy): Bước này có vô lý về toán học không?
        # Ví dụ: "1 + 1 = 3" -> Atomic Error
        atomic_score = self.sympy_check(step_content)
        # 2. Logical Dependency (Model Confidence):
        # Model có chắc chắn bước này suy ra từ context không?
        # Dùng logprob (xác suất) làm thước đo dependency.
        logical_score = np.exp(step_logprob)  # Chuyển logprob về prob bình thường

        # Kết hợp hai thước đo
        final_score = atomic_score * logical_score
        return final_score