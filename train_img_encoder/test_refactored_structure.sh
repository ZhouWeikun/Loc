#!/bin/bash

echo "=========================================="
echo "重构后的项目结构验证"
echo "=========================================="
echo ""

echo "1. 目录结构检查"
echo "---"
for dir in core/base core/config trainers legacy; do
    if [ -d "$dir" ]; then
        echo "✅ $dir/ 存在"
    else
        echo "❌ $dir/ 不存在"
    fi
done
echo ""

echo "2. 核心文件检查"
echo "---"
files=(
    "core/base/trainer_base.py"
    "core/base/components.py"
    "core/config/parser.py"
    "trainers/stage1_visual_encoder.py"
    "trainers/stage2_grid_hashfit.py"
    "trainers/stage3_metric_net.py"
)

for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo "✅ $file"
    else
        echo "❌ $file 不存在"
    fi
done
echo ""

echo "3. Legacy文件检查"
echo "---"
legacy_count=$(ls -1 legacy/trainer_*.py 2>/dev/null | wc -l)
echo "✅ Legacy目录中有 $legacy_count 个旧trainer文件"
echo ""

echo "4. Python语法检查"
echo "---"
python -m py_compile trainers/stage1_visual_encoder.py 2>&1
if [ $? -eq 0 ]; then
    echo "✅ stage1_visual_encoder.py 语法正确"
else
    echo "❌ stage1_visual_encoder.py 语法错误"
fi

python -m py_compile trainers/stage2_grid_hashfit.py 2>&1
if [ $? -eq 0 ]; then
    echo "✅ stage2_grid_hashfit.py 语法正确"
else
    echo "❌ stage2_grid_hashfit.py 语法错误"
fi

python -m py_compile trainers/stage3_metric_net.py 2>&1
if [ $? -eq 0 ]; then
    echo "✅ stage3_metric_net.py 语法正确"
else
    echo "❌ stage3_metric_net.py 语法错误"
fi
echo ""

echo "5. 代码统计"
echo "---"
echo "新代码:"
for file in core/base/trainer_base.py core/base/components.py core/config/parser.py \
            trainers/stage1_visual_encoder.py trainers/stage2_grid_hashfit.py trainers/stage3_metric_net.py; do
    lines=$(wc -l < $file)
    echo "  $file: $lines 行"
done
echo ""

new_lines=$(cat core/base/trainer_base.py core/base/components.py core/config/parser.py \
                trainers/stage1_visual_encoder.py trainers/stage2_grid_hashfit.py trainers/stage3_metric_net.py | wc -l)

old_lines=$(cat legacy/trainer_neuloc_CL_multi_scene.py \
                legacy/trainer_neuloc_CL_multi_scene_hashfit.py \
                legacy/trainer_neuloc_CL_multi_scene_hashfit_metricnet.py 2>/dev/null | wc -l)

echo "总计:"
echo "  新代码: $new_lines 行"
echo "  旧代码(3个主要文件): $old_lines 行"
if [ $old_lines -gt 0 ]; then
    reduction=$(echo "scale=1; ($old_lines - $new_lines) * 100 / $old_lines" | bc)
    echo "  代码减少: ${reduction}%"
fi

echo ""
echo "=========================================="
echo "✅ 重构验证完成"
echo "=========================================="
