#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
用法：
  ./run_directional_cross.sh <基准组> <目标组列表> <输入文件...>

示例：
  ./run_directional_cross.sh A B,C,D group_A.xlsx group_B.xlsx group_C.xlsx group_D.xlsx

默认行为：
  - comparison_mode：cross
  - output_file：项目目录/duplication_results.xlsx
  - model：依次查找 ./models/bge-m3、../models/bge-m3；均不存在时使用 BAAI/bge-m3
  - 自动覆盖同名输出文件

可选环境变量：
  OUTPUT_FILE   自定义输出文件路径
  BGE_M3_MODEL  自定义模型目录或 Hugging Face 模型名
  PYTHON_BIN    自定义 Python 解释器路径
EOF
}

if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
  usage
  exit 0
fi

if (( $# < 3 )); then
  usage >&2
  exit 2
fi

anchor_group=$1
target_groups=$2
shift 2
input_files=("$@")

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
python_bin=${PYTHON_BIN:-"$script_dir/.venv/bin/python"}
output_file=${OUTPUT_FILE:-"$script_dir/duplication_results.xlsx"}

if [[ -n ${BGE_M3_MODEL:-} ]]; then
  model_path=$BGE_M3_MODEL
elif [[ -d "$script_dir/models/bge-m3" ]]; then
  model_path="$script_dir/models/bge-m3"
elif [[ -d "$script_dir/../models/bge-m3" ]]; then
  model_path="$script_dir/../models/bge-m3"
else
  model_path=BAAI/bge-m3
fi

if [[ ! -x "$python_bin" ]]; then
  echo "错误：Python 解释器不存在或不可执行：$python_bin" >&2
  exit 2
fi
for input_file in "${input_files[@]}"; do
  if [[ ! -f "$input_file" ]]; then
    echo "错误：输入文件不存在：$input_file" >&2
    exit 2
  fi
done

exec "$python_bin" "$script_dir/detectduplication.py" \
  --input_files "${input_files[@]}" \
  --output_file "$output_file" \
  --comparison_mode cross \
  --anchor_group "$anchor_group" \
  --target_groups "$target_groups" \
  --model_name_or_path "$model_path" \
  --overwrite
