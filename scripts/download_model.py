from modelscope import snapshot_download
import yaml

# 读取配置
with open("configs/config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

model_dir = snapshot_download(
    config["MODEL_NAME"],
    cache_dir="./models"
)
print(f"✅ 模型下载完成：{model_dir}")
