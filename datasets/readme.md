# 关于数据集的下载：

使用hf-mirror
```shell
pip install -U huggingface_hub
export HF_ENDPOINT=https://hf-mirror.com

cd /PATH/Quantized-Reasoning-Models
# 以下数据集如有不能直接下载，请在网页上手动下载后上传到指定文件夹
# wikitext
huggingface-cli download --repo-type dataset --resume-download Salesforce/wikitext --local-dir ./datasets/wikitext
# pile-val-backup
huggingface-cli download --repo-type dataset --resume-download mit-han-lab/pile-val-backup --local-dir ./datasets/pile-val-backup
# code_generation_lite
huggingface-cli download --repo-type dataset --resume-download livecodebench/code_generation_lite  --local-dir ./datasets/code_generation_lite
# AIME_2024
huggingface-cli download --repo-type dataset --resume-download Maxwell-Jia/AIME_2024 --local-dir ./datasets/AIME_2024
# AIME_90
huggingface-cli download --repo-type dataset --resume-download zwhe99/aime90 --local-dir ./datasets/AIME90
# AIME_25
huggingface-cli download --repo-type dataset --resume-download yentinglin/aime_2025 --local-dir ./datasets/aime_2025
# MATH-500
huggingface-cli download --repo-type dataset --resume-download HuggingFaceH4/MATH-500 --local-dir ./datasets/MATH-500
# NuminaMath-1.5
huggingface-cli download --repo-type dataset --resume-download AI-MO/NuminaMath-1.5 --local-dir ./datasets/NuminaMath-1.5
# GSM8K
huggingface-cli download --repo-type dataset --resume-download openai/gsm8k --local-dir ./datasets/gsm8k
# gpqa:diamond
huggingface-cli download --repo-type dataset --resume-download Idavidrein/gpqa --local-dir ./datasets/gpqa
```
gpqa:diamond数据文件需要先进行预处理转换成.parquet格式才能被正常读取：
  ```python
  from datasets import load_dataset
  dataset = load_dataset("csv", data_files="./datasets/gpqa/gpqa_diamond.csv")
  dataset["train"].to_parquet("./datasets/gpqa/train.parquet") 
  ```

## Q & A 测试更新
现在添加了FlatQuant库中进行qa测试的数据集和相关lighteval代码补充，具体的数据集下载请参考：
```shell
# ARC-E and ARC-C
huggingface-cli download --repo-type dataset --resume-download allenai/ai2_arc --local-dir ./datasets/ai2_arc
# HellaSwag
huggingface-cli download --repo-type dataset --resume-download Rowan/hellaswag --local-dir ./datasets/hellaswag
# LAMBADA
huggingface-cli download --repo-type dataset --resume-download EleutherAI/lambada_openai --local-dir ./datasets/lambada_openai
# PIQA
huggingface-cli download --repo-type dataset --resume-download nthngdy/piqa --local-dir ./datasets/piqa
# WinoGrande
huggingface-cli download --repo-type dataset --resume-download allenai/winogrande --local-dir ./datasets/winogrande
```

# 关于模型文件
评测部分支持原readme.md里提到的模型，自行下载到方便使用的任意路径均可，代码均可接收本地路径。但方便起见建议下载到项目目录下的./modelzoo文件夹下。
