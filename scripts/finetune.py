import yaml
import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    TrainingArguments, Trainer
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

# 加载配置
with open("configs/config.yaml") as f:
    cfg = yaml.safe_load(f)

# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    cfg["MODEL_PATH"],
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(cfg["MODEL_PATH"], trust_remote_code=True)

# LoRA配置
lora_config = LoraConfig(
    r=8, lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05
)
model = get_peft_model(model, lora_config)

# 加载数据
dataset = load_dataset("json", data_files=cfg["DATA_PATH"])

# 训练参数
args = TrainingArguments(
    output_dir=cfg["LORA_OUTPUT"],
    per_device_train_batch_size=1,
    num_train_epochs=cfg["EPOCHS"],
    learning_rate=1e-4,
    fp16=True,
    logging_steps=10
)

# 启动训练
trainer = Trainer(model=model, args=args, train_dataset=dataset["train"])
trainer.train()
model.save_pretrained(cfg["LORA_OUTPUT"])
print("✅ LoRA 微调完成！")
