import json
import yaml
import pandas as pd
from jinja2 import Environment
from collections import OrderedDict
import os

# User-specified HTML template, embedded in the script
HTML_TEMPLATE_MERGED = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>{{ report_title }}</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; margin: 40px; }
        h1 { color: #333; }
        .report-links { margin-bottom: 25px; font-size: 1.1em; }
        .report-links a { margin-right: 15px; text-decoration: none; color: #007bff; }
        .report-links a.active { font-weight: bold; color: #000; text-decoration: underline; }
        table { border-collapse: collapse; width: 100%; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
        th, td { border: 1px solid #e0e0e0; padding: 12px 15px; text-align: left; vertical-align: middle; }
        th { background-color: #f8f8f8; font-weight: 600; }
        tr:nth-child(even) { background-color: #fdfdfd; }
        tr:hover { background-color: #f5f5f5; }
        .score-cell { text-align: right; font-weight: 500; font-family: "Menlo", "Consolas", monospace; }
        .score-na { color: #aaa; }
        .rank-1 { background-color: #ffd700 !important; font-weight: bold; color: #333; }
        .rank-2 { background-color: #c0c0c0 !important; font-weight: bold; color: #333; }
    </style>
</head>
<body>
    <h1>{{ report_title }}</h1>
    <div class="report-links">
        <a href="report_level_1.html" class="{% if current_level == 1 %}active{% endif %}">L1 Capability Dimensions</a>
        <a href="report_level_2.html" class="{% if current_level == 2 %}active{% endif %}">L2 Capability Names</a>
        <a href="report_level_3.html" class="{% if current_level == 3 %}active{% endif %}">L3 Task Difficulty</a>
        <a href="report_level_4.html" class="{% if current_level == 4 %}active{% endif %}">L4 Evaluation Tasks</a>
        <a href="report_level_5_datasets.html" class="{% if current_level == 5 %}active{% endif %}">L5 Datasets</a>
    </div>
    <table>
        <thead>
            <tr>
                {% for header in headers %}
                <th>{{ header }}</th>
                {% endfor %}
            </tr>
        </thead>
        <tbody>
            {% for row in data %}
            <tr>
                {% for cell in row.cells %}
                <td rowspan="{{ cell.rowspan }}">{{ cell.value }}</td>
                {% endfor %}
                {% for score in row.scores %}
                <td class="score-cell {% if score.value == 'N/A' %}score-na{% endif %}{% if score.rank == 1 %} rank-1{% elif score.rank == 2 %} rank-2{% endif %}">{{ score.value }}</td>
                {% endfor %}
            </tr>
            {% endfor %}
        </tbody>
    </table>
</body>
</html>
"""

def load_data(scores_file='scores.json', weights_file='../weight_config.yaml'):
    """Loads scores and weights config files."""
    try:
        with open(scores_file, 'r', encoding='utf-8') as f:
            scores = json.load(f)
        with open(weights_file, 'r', encoding='utf-8') as f:
            weights = yaml.safe_load(f)
        model_names = sorted(scores.keys())
        return scores, weights, model_names
    except FileNotFoundError as e:
        print(f"❌ Error: File not found: {e.filename}")
        exit()

def calculate_all_level_scores(weights_data, scores_data, model_names):
    """
    【Core Function】Recursively calculates weighted average scores for all levels.
    """
    all_scores = {}
    def recurse(sub_weights, path=[]):
        for name, node_data in sub_weights.items():
            current_path = path + [name]
            path_str = " / ".join(current_path)
            
            if isinstance(node_data, dict) and 'children' in node_data and node_data['children']:
                recurse(node_data['children'], current_path)
                all_scores[path_str] = {}
                for model in model_names:
                    weighted_sum, total_weight = 0, 0
                    for child_name, child_data in node_data['children'].items():
                        child_path_str = path_str + " / " + child_name
                        child_score = all_scores.get(child_path_str, {}).get(model)
                        if child_score is not None and pd.notna(child_score):
                            child_weight = child_data.get('weight', 1.0)
                            weighted_sum += child_score * child_weight
                            total_weight += child_weight
                    all_scores[path_str][model] = weighted_sum / total_weight if total_weight > 0 else None
            else:
                # Leaf node (dataset)
                all_scores[path_str] = {model: scores_data.get(model, {}).get(name) for model in model_names}
    
    recurse(weights_data)
    print("✅ All weighted average scores have been calculated.")
    return all_scores

def prepare_and_generate_report(level_depth, all_level_scores, weights_data, model_names, template, output_dir):
    """
    Prepares data for a specific level and generates its HTML report.
    """
    hierarchy_headers = ["Capability Dimension", "Capability Name", "Task Difficulty", "Evaluation Task", "Dataset"]
    report_headers = hierarchy_headers[:level_depth] + model_names
    
    # 1. Prepare data rows
    table_rows = []
    
    def build_rows_recursive(sub_weights, path=[]):
        current_level = len(path)
        if current_level >= level_depth:
            return

        for name, node_data in sub_weights.items():
            current_path = path + [name]
            path_str = " / ".join(current_path)
            
            if len(current_path) == level_depth:
                scores_for_this_row = all_level_scores.get(path_str, {})
                table_rows.append({'path': current_path, 'scores_dict': scores_for_this_row})
            
            elif isinstance(node_data, dict) and 'children' in node_data and node_data['children']:
                build_rows_recursive(node_data['children'], current_path)

    build_rows_recursive(weights_data)
    
    # 2. **NEW SORTING LOGIC**: Move rows with all-zero scores to the bottom,
    #    but keep rows with all-missing scores in their original place.
    def sort_key(row):
        scores = row['scores_dict'].values()
        all_scores_are_none = all(s is None for s in scores)
        
        # If all scores are None (dataset not in scores.json), treat it as a "valid" row to keep its position.
        if all_scores_are_none:
            return 0 # Belongs with the valid group

        # If there's at least one score > 0, it's a valid row.
        has_positive_score = any(isinstance(s, (int, float)) and s > 0 for s in scores)
        if has_positive_score:
            return 0 # Belongs with the valid group

        # Otherwise, all scores are zero or None (but not all None). This row should be moved to the bottom.
        return 1 # Belongs with the empty group

    sorted_rows = sorted(table_rows, key=sort_key)

    # 3. Calculate Rowspan and format the final data for the template
    final_data = []
    rowspan_cache = {}
    
    for i in range(len(sorted_rows)):
        row_data = sorted_rows[i]
        path = row_data['path']
        scores_dict = row_data['scores_dict']

        # Format scores and calculate ranks
        valid_scores = sorted([s for s in scores_dict.values() if isinstance(s, (int, float)) and s > 0], reverse=True)
        first_place = valid_scores[0] if valid_scores else None
        second_place = valid_scores[1] if len(valid_scores) > 1 else None

        scores_list = []
        for model in model_names:
            score = scores_dict.get(model)
            rank = 0
            if isinstance(score, (int, float)) and score > 0:
                if score == first_place: rank = 1
                elif score == second_place: rank = 2
                score_value = f"{score:.2f}"
            else:
                score_value = "N/A" if score is None else f"{float(score):.2f}"
            scores_list.append({'value': score_value, 'rank': rank})
        
        # Calculate rowspans for hierarchy cells
        row_cells = []
        for j in range(level_depth):
            cell_value = path[j]
            if rowspan_cache.get(j, {}).get('value') != cell_value:
                span = 0
                for k in range(i, len(sorted_rows)):
                    if sorted_rows[k]['path'][j] == cell_value:
                        span += 1
                    else:
                        break
                rowspan_cache[j] = {'value': cell_value, 'span': span}
                row_cells.append({'value': cell_value, 'rowspan': span})
        
        final_data.append({'cells': row_cells, 'scores': scores_list})

    # 4. Render and save the HTML file
    level_titles = {1: "L1 Capability Dimensions", 2: "L2 Capability Names", 3: "L3 Task Difficulty", 4: "L4 Evaluation Tasks", 5: "L5 Datasets"}
    base_filename = f"report_level_{level_depth}_datasets.html" if level_depth == 5 else f"report_level_{level_depth}.html"
    output_path = os.path.join(output_dir, base_filename) # <--- 使用 os.path.join 构建完整路径

    html_content = template.render(
        report_title=f"Model Evaluation Report - {level_titles[level_depth]}",
        current_level=level_depth,
        headers=report_headers,
        data=final_data
    )

    with open(output_path, 'w', encoding='utf-8') as f: # <--- 使用新的 output_path
        f.write(html_content)
    print(f"✅ Report generated: {output_path}") # <--- 打印正确的路径

if __name__ == "__main__":
    # 1. 加载数据
    scores, weights, models = load_data()

    if scores and weights:
        # 2. 计算所有层级的加权平均分数
        all_level_scores = calculate_all_level_scores(weights, scores, models)
        
        # 3. 准备Jinja2环境和模板
        env = Environment()
        template = env.from_string(HTML_TEMPLATE_MERGED)

        # 4. 定义并创建输出文件夹 <--- 新增逻辑
        output_dir = 'reports'
        os.makedirs(output_dir, exist_ok=True)

        # 5. 循环生成5个级别的报告，并传入输出路径 <--- 修改循环
        for level in range(1, 6):
            prepare_and_generate_report(level, all_level_scores, weights, models, template, output_dir)
        
        print(f"\n🎉 All 5 level reports have been successfully generated in the '{output_dir}' folder!")