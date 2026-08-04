# 软件与模块名称重复性检测程序

本项目包含两个可独立运行、也可串联运行的 Python 3.10+ 程序：

- `formtrans.py`：将“序号 / 应用系统名称”两列层级表格转换为标准层级路径表。
- `detectduplication.py`：使用 `BAAI/bge-m3` 的 dense embedding 和余弦相似度，召回软件名称、模块名称的文字语义相似候选。

程序只判断名称文本层面的相似性。平台、系统、软件等路径字段仅用于限定比较范围、对象定位和结果追溯，不会拼接进入模型编码文本。程序不调用大语言模型，不执行人工同义词替换，也不分析功能、业务流程、输入输出或技术实现。

## 目录结构

```text
interduplication/
├── formtrans.py
├── detectduplication.py
├── run_directional_cross.sh
├── .gitignore
├── requirements.txt
├── README.md
├── pytest.ini
├── src/
│   ├── __init__.py
│   ├── excel_schema.py
│   ├── hierarchy_parser.py
│   ├── name_normalizer.py
│   ├── embedding_model.py
│   ├── similarity_engine.py
│   └── xlsx_writer.py
└── tests/
    ├── test_formtrans.py
    ├── test_name_normalizer.py
    ├── test_similarity_engine.py
    ├── test_detectduplication_cli.py
    └── test_integration_bge.py
```

## 安装

```bash
git clone https://github.com/DYmello/interduplication.git
cd interduplication
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

运行环境要求：

| 依赖 | 用途 |
|---|---|
| Python 3.10+ | 程序运行环境 |
| PyTorch 2.0+ | BGE-M3 模型推理；GPU 环境应安装与 CUDA 匹配的版本 |
| FlagEmbedding 1.3.5+ | 加载 BGE-M3 dense embedding |
| NumPy 1.24+ | 向量归一化和余弦相似度计算 |
| openpyxl 3.1+ | XLSX 读写 |
| pytest 7.4+ | 单元测试与集成测试 |

完整 Python 依赖已写入 `requirements.txt`。`FlagEmbedding` 还会安装 Transformers 等间接依赖。GPU 用户如果需要特定 CUDA 版本，建议先按照 PyTorch 官方说明安装匹配的 PyTorch，再执行 `pip install -r requirements.txt`。

首次指定 `BAAI/bge-m3` 时通常会从模型仓库下载权重。离线环境应提前准备完整模型目录，并使用：

```bash
--model_name_or_path ./models/bge-m3
```

本地目录需要包含模型配置、分词器和权重文件，不能只提供 Hugging Face 缓存中的不完整子目录。

## 程序 A：formtrans

### 输入

原始 XLSX 至少包含两列：

```text
序号 | 应用系统名称
```

识别规则：

| 序号形式 | 层级 |
|---|---|
| `一`、`四`、`六` | 平台 |
| `（一）`、`（二）` | 系统 |
| `1`、`10`、`25` | 软件 |
| `（1）`、`（2）` | 模块 |
| 序号为空、名称非空 | 被排除的分类标题 |

空行和名称为“合计”的行会被忽略。序号匹配前进行 Unicode NFKC 与空格规范化；输出名称只清除首尾和连续无意义空白，不改写原始名称。

### 运行

```bash
python formtrans.py \
  --input_file ../systemsheet.xlsx \
  --output_file ./normalized_input.xlsx \
  --group_id A \
  --overwrite
```

可选参数：

- `--sheet_name Sheet1`：不传时读取第一个工作表。
- `--strict`：层级缺失或未知序号立即失败；默认记录警告并按规则跳过或保留空上级字段。
- `--overwrite`：允许覆盖已有输出，默认拒绝。

### 输出

`normalized_input.xlsx` 包含：

1. `Input`：严格按以下顺序保存标准路径字段：

   ```text
   组编号, 平台编号, 平台名称, 系统编号, 系统名称,
   软件编号, 软件名称, 模块编号, 模块名称
   ```

2. `Config`：保存默认比较参数。
3. `Excluded_Categories`：保存被排除的分类标题及原始行号。

每个软件有一条模块字段为空的软件实体记录；每个模块另占一行并携带完整平台、系统、软件路径。所有编号按原表出现顺序全局递增，同一输入重复运行可得到相同编号。输出不使用合并单元格、“同上”或省略路径。

当前 `systemsheet.xlsx` 的回归基线是 3 个平台、5 个系统、102 个软件、312 个模块和 3 个分类标题。数量只用于验收，没有硬编码在业务逻辑中。

## 程序 B：detectduplication

### 输入

输入必须是一个或多个 `formtrans` 生成的标准 XLSX，且包含 `Input` 工作表和严格顺序的九个字段。多个文件会纵向合并；若相同“组编号 + 对象编号”对应不同名称或路径，程序直接报错。

软件对象按“组编号 + 软件编号”去重，模块对象按“组编号 + 模块编号”去重。软件仅与软件比较，模块仅与模块比较。

### 名称规范化与模型

送入 BGE-M3 前只执行：

1. Unicode NFKC；
2. 去除首尾空白并压缩连续空白；
3. 移除中文字符之间无意义空格；
4. 英文字母统一大小写用于计算。

默认不删除“评测”“检测”“验证”“软件”“模块”等词，不拼接上级路径，不扩写名称。`--remove_module_suffix` 可选择只删除模块名称末尾的“模块”，默认关闭，且实际配置会写入结果文件。

模型调用固定请求 dense embedding：

```python
return_dense=True
return_sparse=False
return_colbert_vecs=False
```

向量由 NumPy 显式执行 L2 归一化，随后以点积计算余弦相似度。该分数不是概率或重复率。

### 运行

同组内部比较：

```bash
python detectduplication.py \
  --input_files ./normalized_input.xlsx \
  --output_file ./duplication_results.xlsx \
  --comparison_mode within \
  --levels software,module \
  --similarity_threshold 0.85 \
  --top_k 10 \
  --overwrite
```

跨组比较：

```bash
python detectduplication.py \
  --input_files ./group_A.xlsx ./group_B.xlsx \
  --output_file ./duplication_results.xlsx \
  --comparison_mode cross
```

### 定向跨组比较

在 `cross` 模式下，可以指定一个基准组以及一个或多个目标组：

```bash
python detectduplication.py \
  --input_files group_A.xlsx group_B.xlsx group_C.xlsx group_D.xlsx \
  --output_file duplication_results.xlsx \
  --comparison_mode cross \
  --anchor_group A \
  --target_groups B,C,D \
  --similarity_threshold 0.85 \
  --top_k 10 \
  --overwrite
```

其中 A 是基准组，B、C、D 是目标组。程序只生成 A-B、A-C、A-D，不生成目标组之间或 B-A、C-A、D-A 等反向比较。每个 A 组对象会合并全部目标组候选，根据余弦相似度统一降序排名，`top_k` 是跨全部目标组的合计上限。启用 `--include_unmatched` 时，也只为没有匹配结果的 A 组对象输出“无匹配”记录。

`--anchor_group` 与 `--target_groups` 必须同时提供，而且只适用于 `cross` 模式。`--target_groups` 使用逗号分隔，解析时会去除空格、自动去重并保留首次出现顺序。不提供这两个参数时，`cross` 保持原有行为，比较输入数据中所有不同组的组合。

日常运行可使用项目提供的简化脚本：

```bash
./run_directional_cross.sh \
  A \
  B,C,D \
  group_A.xlsx group_B.xlsx group_C.xlsx group_D.xlsx
```

脚本会自动设置 `comparison_mode=cross`、模型、默认输出文件 `duplication_results.xlsx` 和覆盖选项。模型会依次查找仓库内的 `./models/bge-m3`、仓库同级的 `../models/bge-m3`；均不存在时使用 Hugging Face 模型名 `BAAI/bge-m3`。需要自定义输出文件时，可在命令前设置 `OUTPUT_FILE`：

```bash
OUTPUT_FILE=results_A_to_BCD.xlsx \
  ./run_directional_cross.sh A B,C,D group_A.xlsx group_B.xlsx group_C.xlsx group_D.xlsx
```

运行 `./run_directional_cross.sh --help` 可查看完整说明。

本地模型与 GPU：

```bash
python detectduplication.py \
  --input_files ./normalized_input.xlsx \
  --output_file ./duplication_results.xlsx \
  --model_name_or_path ./models/bge-m3 \
  --device cuda \
  --use_fp16 \
  --overwrite
```

CPU：

```bash
python detectduplication.py \
  --input_files ./normalized_input.xlsx \
  --device cpu \
  --no-use_fp16 \
  --overwrite
```

主要参数：

- `--comparison_mode within|cross|all`，默认 `within`。
- `--anchor_group A`：定向跨组比较的基准组，只适用于 `cross`。
- `--target_groups B,C,D`：定向跨组比较的目标组，必须与 `--anchor_group` 同时提供。
- `--levels software,module`，可只选择一个层级。
- `--similarity_threshold`，默认 `0.85`。
- `--top_k`，默认每个对象最多 10 个候选。
- `--include_unmatched`，输出没有达到阈值的对象。
- `--batch_size 64`、`--max_length 64`。
- `--device auto|cuda|cpu`；`auto` 检测到 CUDA 时使用 GPU。
- `--use_fp16` / `--no-use_fp16`；GPU 默认启用，CPU 强制关闭。
- `--block_size 2048`：相似度分块行数，避免一次创建完整 N×N 矩阵。

### 输出

`duplication_results.xlsx` 包含：

1. `SimilarityResults`：一行一个 A-B 候选，字段顺序严格符合验收要求；相似度显示 6 位小数。
2. `InputSnapshot`：本次实际使用的去重后标准输入。
3. `Config`：模型、模式、阈值、设备、FP16 与余弦指标等实际参数。

结果按“比较层级 → A组编号 → A对象编号 → 相似度排名”排序。组内比较不会输出自身或 A-B/B-A 对称重复；跨组比较只比较不同组，A 优先使用输入顺序较前的组。

### 相似等级

| 条件 | 相似等级 |
|---|---|
| 规范化名称完全一致 | 名称完全相同 |
| 分数 ≥ 0.90 | 极高相似 |
| 0.85 ≤ 分数 < 0.90 | 高相似 |
| 0.75 ≤ 分数 < 0.85 | 中等相似 |
| 分数 < 0.75 | 低相似 |

这些阈值只是初始工程阈值，不是 BGE-M3 的通用标准。正式使用前必须人工标注一批相似/不相似名称对，并基于 Precision、Recall、F1 与实际误报成本校准阈值。

## 测试

无需下载模型的单元测试：

```bash
pytest -q -m "not integration"
```

真实模型集成测试由用户显式启用：

```bash
RUN_BGE_INTEGRATION=1 \
BGE_M3_MODEL=./models/bge-m3 \
pytest -q -m integration
```

## 性能与限制

- 软件和模块分开编码、分开计算，不会合并为同一矩阵。
- 余弦计算按行分块，不保留低于阈值的全量结果。
- 几百到几千对象可直接运行；超过 10,000 个对象时日志会提示评估 FAISS，本基础版本不依赖 FAISS。
- 结果只能解释为“检测到名称文字语义高度相似的候选软件或模块”，不能直接推断功能相同、重复建设或应当合并。
- 最终业务结论仍需要人工复核。
