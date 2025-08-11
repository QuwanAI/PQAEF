echo "Starting to execute all test configuration files..."
echo "======================================"

# Define configuration file list
config_files=(
    "test_AGNews.yaml"
)

# Record start time
start_time=$(date +%s)
total_files=${#config_files[@]}
current=0
success_count=0
failed_count=0
failed_files=()

# Execute configuration files one by one
for config_file in "${config_files[@]}"; do
    current=$((current + 1))
    echo "[$current/$total_files] Executing: $config_file"
    echo "Command: pqaef-runner --config ./test/$config_file"
    
    # Execute command and capture return value
    if pqaef-runner --config "./test/$config_file"; then
        echo "✅ $config_file executed successfully"
        success_count=$((success_count + 1))
    else
        echo "❌ $config_file execution failed"
        failed_count=$((failed_count + 1))
        failed_files+=("$config_file")
    fi
    
    echo "--------------------------------------"
done

# Record end time and calculate elapsed time
end_time=$(date +%s)
elapsed_time=$((end_time - start_time))

# Output execution summary
echo "======================================"
echo "Execution completed!"
echo "Total files: $total_files"
echo "Success: $success_count"
echo "Failed: $failed_count"
echo "Total time: ${elapsed_time}s"

if [ $failed_count -gt 0 ]; then
    echo ""
    echo "Failed configuration files:"
    for failed_file in "${failed_files[@]}"; do
        echo "  - $failed_file"
    done
    exit 1
else
    echo "🎉 All configuration files executed successfully!"
    exit 0
fi