echo "开始执行所有测试配置文件..."
echo "======================================"

# 定义配置文件列表
config_files=(
    "test_AGNews.yaml"
)

# 记录开始时间
start_time=$(date +%s)
total_files=${#config_files[@]}
current=0
success_count=0
failed_count=0
failed_files=()

# 逐个执行配置文件
for config_file in "${config_files[@]}"; do
    current=$((current + 1))
    echo "[$current/$total_files] 正在执行: $config_file"
    echo "命令: pqaef-runner --config ./test/$config_file"
    
    # 执行命令并捕获返回值
    if pqaef-runner --config "./test/$config_file"; then
        echo "✅ $config_file 执行成功"
        success_count=$((success_count + 1))
    else
        echo "❌ $config_file 执行失败"
        failed_count=$((failed_count + 1))
        failed_files+=("$config_file")
    fi
    
    echo "--------------------------------------"
done

# 记录结束时间并计算耗时
end_time=$(date +%s)
elapsed_time=$((end_time - start_time))

# 输出执行总结
echo "======================================"
echo "执行完成！"
echo "总文件数: $total_files"
echo "成功: $success_count"
echo "失败: $failed_count"
echo "总耗时: ${elapsed_time}秒"

if [ $failed_count -gt 0 ]; then
    echo ""
    echo "失败的配置文件:"
    for failed_file in "${failed_files[@]}"; do
        echo "  - $failed_file"
    done
    exit 1
else
    echo "🎉 所有配置文件执行成功！"
    exit 0
fi