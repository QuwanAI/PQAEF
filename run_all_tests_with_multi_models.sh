#!/bin/bash

# 自动化测试运行脚本
# 功能：循环使用不同的模型配置运行测试

echo "🚀 开始自动化测试流程..."
echo "======================================"

# 配置文件路径
CONFIG_FILE="./model_configs.json"
CONVERT_SCRIPT="./convert_models.py"
TEST_SCRIPT="./run_all_tests.sh"
TEMP_CONFIG="./temp_config.json"

# 检查必要文件是否存在
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ 配置文件不存在: $CONFIG_FILE"
    exit 1
fi

if [ ! -f "$CONVERT_SCRIPT" ]; then
    echo "❌ 转换脚本不存在: $CONVERT_SCRIPT"
    exit 1
fi

if [ ! -f "$TEST_SCRIPT" ]; then
    echo "❌ 测试脚本不存在: $TEST_SCRIPT"
    exit 1
fi

# 确保测试脚本有执行权限
chmod +x "$TEST_SCRIPT"

# 读取配置数量
config_count=$(python3 -c "import json; data=json.load(open('$CONFIG_FILE')); print(len(data))")
echo "📋 找到 $config_count 个模型配置"

# 记录开始时间
start_time=$(date +%s)
total_success=0
total_failed=0
failed_configs=()

# 循环处理每个配置
for i in $(seq 0 $((config_count - 1))); do
    echo ""
    echo "======================================"
    echo "🔄 处理配置 $((i + 1))/$config_count"
    echo "======================================"
    
    # 提取当前配置
    python3 -c "
import json
with open('$CONFIG_FILE', 'r') as f:
    configs = json.load(f)
with open('$TEMP_CONFIG', 'w') as f:
    json.dump(configs[$i], f, indent=2)
" 2>/dev/null
    
    if [ $? -ne 0 ]; then
        echo "❌ 提取配置 $((i + 1)) 失败"
        total_failed=$((total_failed + 1))
        failed_configs+=("配置$((i + 1)): 提取失败")
        continue
    fi
    
    # 获取当前配置的模型信息
    model_info=$(python3 -c "
import json
data = json.load(open('$TEMP_CONFIG'))
model_type = data.get('model_type', 'unknown')
model_name = data.get('model_name', 'unknown')
model_id = data.get('config', {}).get('model_identifier', data.get('config', {}).get('model_path', 'unknown'))
print(f'{model_type}|{model_name}|{model_id}')
" 2>/dev/null)
    
    IFS='|' read -r model_type model_name model_id <<< "$model_info"
    echo "📝 当前模型类型: $model_type"
    echo "📝 当前模型名称: $model_name"
    echo "📝 当前模型标识: $model_id"
    
    # 步骤1：更新配置文件
    echo "🔧 步骤1: 更新配置文件..."
    if python3 "$CONVERT_SCRIPT" "$TEMP_CONFIG"; then
        echo "✅ 配置文件更新成功"
    else
        echo "❌ 配置文件更新失败"
        total_failed=$((total_failed + 1))
        failed_configs+=("配置$((i + 1)): 配置更新失败")
        continue
    fi
    
    # 步骤2：运行测试
    echo "🧪 步骤2: 运行测试脚本..."
    if bash "$TEST_SCRIPT"; then
        echo "✅ 测试执行成功 - 模型: $model_name ($model_type)"
        total_success=$((total_success + 1))
    else
        echo "❌ 测试执行失败 - 模型: $model_name ($model_type)"
        total_failed=$((total_failed + 1))
        failed_configs+=("配置$((i + 1)): 测试执行失败 (模型: $model_name)")
    fi
    
    # 清理临时文件
    rm -f "$TEMP_CONFIG"
    
    echo "📊 当前进度: 成功 $total_success, 失败 $total_failed"
done

# 记录结束时间并计算耗时
end_time=$(date +%s)
elapsed_time=$((end_time - start_time))
elapsed_hours=$((elapsed_time / 3600))
elapsed_minutes=$(((elapsed_time % 3600) / 60))
elapsed_seconds=$((elapsed_time % 60))

# 输出最终总结
echo ""
echo "======================================"
echo "🎯 自动化测试流程完成！"
echo "======================================"
echo "📊 执行统计:"
echo "   总配置数: $config_count"
echo "   成功: $total_success"
echo "   失败: $total_failed"
echo "   总耗时: ${elapsed_hours}小时 ${elapsed_minutes}分钟 ${elapsed_seconds}秒"

if [ $total_failed -gt 0 ]; then
    echo ""
    echo "❌ 失败的配置:"
    for failed_config in "${failed_configs[@]}"; do
        echo "   - $failed_config"
    done
    echo ""
    echo "⚠️  建议检查失败的配置和日志"
else
    echo ""
    echo "🎉 所有配置都执行成功！"
    
    # 步骤3：计算加权分数
    echo ""
    echo "====================================="
    echo "📊 步骤3: 计算加权分数..."
    echo "====================================="
    if python calculate_weighted_scores.py; then
        echo "✅ 加权分数计算成功"
        
        # 步骤4：生成报告
        echo ""
        echo "====================================="
        echo "📋 步骤4: 生成分析报告..."
        echo "====================================="
        cd result_analyze
        if python generate_report.py result.xlsx scores.json report.html; then
            echo "✅ 分析报告生成成功"
            echo "📄 报告文件: result_analyze/report.html"
        else
            echo "❌ 分析报告生成失败"
        fi
        cd ..
    else
        echo "❌ 加权分数计算失败"
    fi
fi

if [ $total_failed -gt 0 ]; then
    exit 1
else
    exit 0
fi