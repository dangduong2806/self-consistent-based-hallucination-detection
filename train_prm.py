import json
import torch
from torch.utils.data import Dataset, random_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
import yaml
import gc

# Cấu hình
config_path = "configs/main_config.yaml"

class PRMDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_len=512):    
        self.data = []
        self.tokenized_data = []
        self.tokenizer = tokenizer
        self.max_len = max_len

        # Mapping từ Rating của dataset sang Label của Model (0, 1, 2)
        # Rating -1 -> Label 0 (Sai)
        # Rating  0 -> Label 1 (Trung tính)
        # Rating  1 -> Label 2 (Đúng)
        self.label_map = {-1: 0, 0: 1, 1: 2}

        # Load data PRM800K (Phase 2 - label từng bước)
        with open(jsonl_path, 'r') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    
                    # Kiểm tra dữ liệu đầu vào cơ bản
                    if 'question' not in item or 'label' not in item or 'steps' not in item['label']:
                        continue
                    
                    problem = item['question']['problem']
                    steps = item['label']['steps']

                    # History khởi đầu bằng đề bài
                    history_context = problem

                    for step in steps:
                        # 1. Tạo Dữ liệu Training (Học cả Đúng và Sai)
                        if step.get('completions') is None: continue

                        for comp in step['completions']:
                            if comp['rating'] is None: continue

                            rating = comp.get('rating')
                            label = self.label_map[rating]
                            # Input cho DeBERTa: Context + [SEP] + Candidate
                            text = f"{history_context} [SEP] {comp['text']}"

                            self.data.append({
                                    'text': text,
                                    'label': label
                                })
                        # 2. Cập nhật History cho bước sau (Chỉ lấy bước được chọn làm mạch chính)
                        # Sample của bạn có trường 'chosen_completion' (VD: 0, 1, 2...)
                        # Đây là chỉ mục của bước đi tiếp theo mạch truyện
                        chosen_idx = step.get('chosen_completion')
                        if chosen_idx is not None and chosen_idx < len(step['completions']):
                            # Lấy text của bước được chọn để cộng vào lịch sử
                            chosen_step_text = step['completions'][chosen_idx]['text']
                            history_context += " " + chosen_step_text
                        else:
                            # Fallback: Nếu không có chosen_completion, mới dùng logic tìm cái đúng đầu tiên
                            for comp in step['completions']:
                                if comp['rating'] == 1:
                                    history_context += " " + comp['text']
                                    break
                except Exception as e:
                    print(f"Error details: {str(e)}")
                    import traceback
                    traceback.print_exc()

        print(f"✅ Đã đọc xong {len(self.data)} mẫu raw. Bắt đầu Tokenize hàng loạt...")

        # BƯỚC 2: Tokenize Hàng loạt (Batch Tokenization) - Tốc độ X100 lần
        # Thay vì tokenize từng dòng, ta gom lại xử lý luôn
        batch_size = 2000
        for i in range(0, len(self.data), batch_size):
            batch = self.data[i : i + batch_size]
            texts = [x['text'] for x in batch]
            labels = [x['label'] for x in batch]
            
            # Tokenize batch này
            encodings = tokenizer(
                texts,
                truncation=True,
                max_length=max_len,
                add_special_tokens=True,
                return_attention_mask=True,
                # KHÔNG padding ở đây để tiết kiệm RAM, để DataCollator lo
            )
            
            # Lưu vào self.data
            for j in range(len(texts)):
                self.tokenized_data.append({
                    'input_ids': encodings['input_ids'][j],
                    'attention_mask': encodings['attention_mask'][j],
                    'label': labels[j]
                })

        # BƯỚC 3: DỌN DẸP RAM (QUAN TRỌNG NHẤT)
        del self.data
        gc.collect() # Ép Python giải phóng RAM ngay lập tức
        n = len(self.tokenized_data)
        self.tokenized_data = self.tokenized_data[: 1 * n // 6]
        print(f"🎉 Sẵn sàng train! Tổng số mẫu: {len(self.tokenized_data)}")

    def __len__(self):
        return len(self.tokenized_data)

    # def __getitem__(self, idx):
    #     item = self.data[idx]
    #     encoding = self.tokenizer(
    #         item['text'],
    #         truncation=True,
    #         max_length=self.max_len,
    #         # padding="max_length",
    #         # return_tensors="pt"
    #     )
    #     return {
    #         "input_ids": encoding["input_ids"],
    #         "attention_mask": encoding["attention_mask"],
    #         # "labels": torch.tensor(item['label'], dtype=torch.long)
    #         "labels": item['label']
    #     }
    def __getitem__(self, idx):
        # Hàm này giờ chỉ việc lấy dữ liệu có sẵn, không cần tính toán gì cả -> Siêu nhanh
        return {
            "input_ids": self.tokenized_data[idx]['input_ids'],
            "attention_mask": self.tokenized_data[idx]['attention_mask'],
            "labels": self.tokenized_data[idx]['label']
        }

def train():
    # 1. Load Config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    MODEL_NAME = config['tool']['name1']
    OUTPUT_DIR = config['tool']['output_dir1']
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True) # tokenize nhanh hơn  
    
    # --- SỬA QUAN TRỌNG: num_labels=3 ---
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=3,  # Model sẽ có 3 đầu ra
        id2label={0: "Bad", 1: "Neutral", 2: "Good"}, # Gán nhãn cho dễ hiểu khi infer
        label2id={"Bad": 0, "Neutral": 1, "Good": 2},
        ignore_mismatched_sizes=True
    )

    # # --- [CHÈN ĐOẠN NÀY ĐỂ FIX LỖI BACKWARD & OOM] ---
    # print("Applying DeBERTa Gradient Checkpointing Fix...")
    
    # # 1. Tắt cache (Bắt buộc khi train)
    # model.config.use_cache = False 
    
    # # 2. Bật checkpointing thủ công trên model
    # model.gradient_checkpointing_enable()
    
    # # 3. THUỐC ĐẶC TRỊ: Đảm bảo Input Embeddings nhận Gradient
    # # Nếu không có dòng này, đồ thị tính toán sẽ bị ngắt quãng gây ra lỗi "backward second time"
    # if hasattr(model, "enable_input_require_grads"):
    #     model.enable_input_require_grads()
    # else:
    #     def make_inputs_require_grad(module, input, output):
    #         output.requires_grad_(True)
    #     model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    # # -----------------------------------------------------
    
    # Load dataset (Bạn cần trỏ đúng file phase2_train.jsonl)
    full_dataset = PRMDataset("data/raw/phase1_train.jsonl", tokenizer, max_len=256)
    
    # Chia 90% train, 10% validation
    train_size = int(0.7 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")

    # 5. Training Arguments
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,              # DeBERTa cần train kỹ hơn chút (3-5 epochs)
        per_device_train_batch_size=8,   # 4 hoặc 8 tùy VRAM (4 là an toàn cho GPU 8-12GB)
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,   # Tích lũy gradient để batch size thực tế = 16
        gradient_checkpointing=False,     # <--- CỰC KỲ QUAN TRỌNG: Tiết kiệm 50-70% VRAM (Đổi lại tốc độ train sẽ chậm hơn khoảng 20%)
        # gradient_checkpointing_kwargs={"use_reentrant": False}, # <--- THÊM DÒNG NÀY (Thuốc đặc trị)
        learning_rate=2e-5,              # QUAN TRỌNG: LR thấp cho DeBERTa
        weight_decay=0.01,
        warmup_ratio=0.1,                # Warmup giúp ổn định training đầu
        lr_scheduler_type="cosine",      # <--- QUAN TRỌNG: Giảm LR theo hình Cosine (tốt hơn Linear)
        fp16=True,                       # Tăng tốc trên GPU
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        save_total_limit=2,              # Chỉ giữ 2 checkpoint tốt nhất để tiết kiệm ổ cứng
        report_to="none",
        # Thêm dòng này để DataLoader load dữ liệu nhanh hơn
        dataloader_num_workers=4
    )

    # Collator padding động
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = torch.argmax(torch.tensor(logits), dim=-1)
        acc = (predictions == torch.tensor(labels)).float().mean().item()
        return {"accuracy": acc}
    
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )
    
    print("Bắt đầu huán luyện")
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("PRM Specialist Model trained and saved!")

if __name__ == "__main__":
    train() # Uncomment để chạy
    pass