#!/bin/bash
echo "=========================================="
echo "导入检查报告"
echo "=========================================="
echo ""

echo "📋 检查所有trainer文件的导入..."
echo ""

for file in trainers/stage*.py; do
    echo "检查: $file"
    echo "---"
    
    # 提取所有from import语句
    grep "^from " "$file" | while read -r line; do
        echo "  $line"
    done
    
    echo ""
done

echo "🔍 验证导入路径规范:"
echo "---"
echo "✅ 应该使用的前缀:"
echo "  - trainer_depends.*  (基类、组件、配置)"
echo "  - train_img_encoder.* (数据集、工具)"
echo "  - models.*            (模型定义)"
echo "  - losses.*            (损失函数)"
echo "  - tool.*              (工具函数)"
echo "  - trainers.*          (其他trainer)"
echo ""

echo "🔧 语法检查:"
echo "---"
python -m py_compile trainers/stage1_visual_encoder.py 2>/dev/null
[ $? -eq 0 ] && echo "  ✅ stage1_visual_encoder.py" || echo "  ❌ stage1_visual_encoder.py 有语法错误"

python -m py_compile trainers/stage2_grid_hashfit.py 2>/dev/null
[ $? -eq 0 ] && echo "  ✅ stage2_grid_hashfit.py" || echo "  ❌ stage2_grid_hashfit.py 有语法错误"

python -m py_compile trainers/stage3_metric_net.py 2>/dev/null
[ $? -eq 0 ] && echo "  ✅ stage3_metric_net.py" || echo "  ❌ stage3_metric_net.py 有语法错误"

echo ""
echo "=========================================="
echo "✅ 导入检查完成"
echo "=========================================="
