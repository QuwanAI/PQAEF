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
        <a href="report_level_1.html" class="{% if current_level == 1 %}active{% endif %}">L1 能力维度</a>
        <a href="report_level_2.html" class="{% if current_level == 2 %}active{% endif %}">L2 能力名称</a>
        <a href="report_level_3.html" class="{% if current_level == 3 %}active{% endif %}">L3 任务难度</a>
        <a href="report_level_4.html" class="{% if current_level == 4 %}active{% endif %}">L4 评价任务</a>
        <a href="report_level_5_datasets.html" class="{% if current_level == 5 %}active{% endif %}">L5 数据集</a>
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

def load_data(scores_file='scores_scale.json', weights_file='../weight_config.yaml'):
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

def min_max_scale_scores(input_file, output_file):
    """
    Perform min-max normalization on model scores
    
    Args:
        input_file: Path to the input scores.json file
        output_file: Path to the output scores_scale.json file
    """
    # Read original data
    with open(input_file, 'r', encoding='utf-8') as f:
        scores_data = json.load(f)
    
    # Get all model names and dataset names
    models = list(scores_data.keys())
    datasets = set()
    for model_scores in scores_data.values():
        datasets.update(model_scores.keys())
    datasets = sorted(list(datasets))
    
    print(f"Found {len(models)} models and {len(datasets)} datasets")
    
    # Initialize normalized data structure
    scaled_scores = {model: {} for model in models}
    
    # Normalize each dataset
    for dataset in datasets:
        # Collect scores for this dataset across all models
        dataset_scores = []
        valid_models = []
        
        for model in models:
            if dataset in scores_data[model]:
                score = scores_data[model][dataset]
                if score is not None:  # Handle possible None values
                    dataset_scores.append(score)
                    valid_models.append(model)
        
        if len(dataset_scores) < 2:
            # If there's only one or no valid scores, skip normalization
            print(f"Warning: Dataset {dataset} has only {len(dataset_scores)} valid scores, skipping normalization")
            for model in valid_models:
                scaled_scores[model][dataset] = scores_data[model][dataset]
            continue
        
        # Calculate minimum and maximum values
        min_score = min(dataset_scores)
        max_score = max(dataset_scores)
        
        print(f"Dataset {dataset}: min={min_score:.2f}, max={max_score:.2f}")
        
        # Perform min-max normalization
        if max_score == min_score:
            # If all scores are the same, set to 50 (middle value)
            for model in valid_models:
                scaled_scores[model][dataset] = 50.0
        else:
            for model in valid_models:
                original_score = scores_data[model][dataset]
                # Normalization formula: (x - min) / (max - min) * 100
                scaled_score = (original_score - min_score) / (max_score - min_score) * 100
                scaled_scores[model][dataset] = round(scaled_score, 2)
        
        # Handle models that don't have scores for this dataset
        for model in models:
            if model not in valid_models:
                if dataset in scores_data[model]:
                    scaled_scores[model][dataset] = scores_data[model][dataset]  # Keep original value (usually None)
    
    # Save normalized data
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(scaled_scores, f, ensure_ascii=False, indent=4)
    
    print(f"\nNormalization completed! Results saved to {output_file}")
    
    # Display some statistics
    print("\nPost-normalization statistics:")
    for dataset in datasets[:5]:  # Only show statistics for the first 5 datasets
        dataset_scores = []
        for model in models:
            if dataset in scaled_scores[model] and scaled_scores[model][dataset] is not None:
                dataset_scores.append(scaled_scores[model][dataset])
        
        if dataset_scores:
            print(f"{dataset}: min={min(dataset_scores):.2f}, max={max(dataset_scores):.2f}, avg={sum(dataset_scores)/len(dataset_scores):.2f}")

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
    hierarchy_headers = ["能力维度", "能力名称", "任务难度", "评价任务", "数据集"]
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
    level_titles = {1: "L1 能力维度", 2: "L2 能力名称", 3: "L3 任务难度", 4: "L4 评价任务", 5: "L5 数据集"}
    base_filename = f"report_level_{level_depth}_datasets.html" if level_depth == 5 else f"report_level_{level_depth}.html"
    output_path = os.path.join(output_dir, base_filename)

    html_content = template.render(
        report_title=f"Model Evaluation Report - {level_titles[level_depth]}",
        current_level=level_depth,
        headers=report_headers,
        data=final_data
    )

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"✅ Report generated: {output_path}")

def calculate_final_ranking(all_level_scores, model_names, output_dir):
    """
    计算最终排名：基础能力(30%) + 情感能力(40%) + 陪伴能力(30%)
    价值观与安全不参与排名，低于70分标为灰色
    """
    # Define weights
    weights = {
        '基础能力': 0.3,
        '情感能力': 0.4,
        '陪伴能力': 0.3
    }
    
    # Calculate total score for each model
    final_scores = {}
    safety_scores = {}  # Values and Safety scores, used for marking
    
    for model in model_names:
        total_score = 0
        valid_dimensions = 0
        
        # Calculate weighted scores for the three dimensions participating in ranking
        for dimension, weight in weights.items():
            score = all_level_scores.get(dimension, {}).get(model)
            if score is not None and isinstance(score, (int, float)):
                total_score += score * weight
                valid_dimensions += 1
        
        # Get Values and Safety score
        safety_score = all_level_scores.get('价值观与安全', {}).get(model)
        safety_scores[model] = safety_score
        
        # Only calculate total score when all three dimensions have scores
        if valid_dimensions == 3:
            final_scores[model] = total_score
        else:
            final_scores[model] = None
    
    # Sort by total score (descending)
    sorted_models = sorted(
        [(model, score) for model, score in final_scores.items() if score is not None],
        key=lambda x: x[1],
        reverse=True
    )
    
    # Generate ranking file
    output_path = os.path.join(output_dir, 'final_ranking.txt')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("Final Model Ranking\n")
        f.write("=" * 50 + "\n\n")
        f.write("Ranking Rules: Basic Ability (30%) + Emotional Ability (40%) + Companionship Ability (30%)\n")
        f.write("Note: Values and Safety do not participate in ranking, scores below 70 are marked in gray\n\n")
        
        f.write(f"{'排名':<4} {'模型名称':<20} {'总分':<8} {'基础能力':<8} {'情感能力':<8} {'陪伴能力':<8} {'价值观与安全':<10} {'状态':<6}\n")
        f.write("-" * 80 + "\n")
        
        # Write models with total scores (by ranking)
        for rank, (model, total_score) in enumerate(sorted_models, 1):
            basic_score = all_level_scores.get('基础能力', {}).get(model)
            emotion_score = all_level_scores.get('情感能力', {}).get(model)
            companion_score = all_level_scores.get('陪伴能力', {}).get(model)
            safety_score = safety_scores.get(model)
            
            # Safely handle None values, convert to string format
            basic_str = f"{basic_score:.2f}" if basic_score is not None else "N/A"
            emotion_str = f"{emotion_score:.2f}" if emotion_score is not None else "N/A"
            companion_str = f"{companion_score:.2f}" if companion_score is not None else "N/A"
            safety_str = f"{safety_score:.2f}" if safety_score is not None else "N/A"
            total_str = f"{total_score:.2f}" if total_score is not None else "N/A"
            
            # Determine if it needs to be marked gray (Values and Safety < 70)
            status = "Gray" if safety_score is not None and safety_score < 70 else "Normal"
            
            f.write(f"{rank:<4} {model:<20} {total_str:<8} {basic_str:<8} {emotion_str:<8} {companion_str:<8} {safety_str:<10} {status:<6}\n")
        
        # Write models without complete scores
        incomplete_models = [model for model, score in final_scores.items() if score is None]
        if incomplete_models:
            f.write("\n未参与排名的模型（缺少完整分数）：\n")
            f.write("-" * 40 + "\n")
            for model in incomplete_models:
                basic_score = all_level_scores.get('基础能力', {}).get(model)
                emotion_score = all_level_scores.get('情感能力', {}).get(model)
                companion_score = all_level_scores.get('陪伴能力', {}).get(model)
                safety_score = safety_scores.get(model)
                
                basic_str = f"{basic_score:.2f}" if basic_score is not None else "N/A"
                emotion_str = f"{emotion_score:.2f}" if emotion_score is not None else "N/A"
                companion_str = f"{companion_score:.2f}" if companion_score is not None else "N/A"
                safety_str = f"{safety_score:.2f}" if safety_score is not None else "N/A"
                
                f.write(f"     {model:<20} {'N/A':<8} {basic_str:<8} {emotion_str:<8} {companion_str:<8} {safety_str:<10} {'缺分':<6}\n")
    
    print(f"✅ Final ranking saved: {output_path}")
    return final_scores, sorted_models

def generate_reports_for_scores(scores_file, output_dir, report_type="Original"):
    """
    Generate reports for specified score file
    
    Args:
        scores_file: Score file path
        output_dir: Output directory
        report_type: Report type ("Original" or "Normalized")
    """
    print(f"\n🔄 Starting to generate {report_type} reports...")
    
    # 1. Load data
    scores, weights, models = load_data(scores_file)

    if scores and weights:
        # 2. Calculate weighted average scores for all levels
        all_level_scores = calculate_all_level_scores(weights, scores, models)
        
        # 3. Prepare Jinja2 environment and template
        env = Environment()
        template = env.from_string(HTML_TEMPLATE_MERGED)

        # 4. Create output folder
        os.makedirs(output_dir, exist_ok=True)

        # 5. Generate reports for 5 levels in loop
        for level in range(1, 6):
            prepare_and_generate_report(level, all_level_scores, weights, models, template, output_dir)
        
        print(f"\n🎉 All 5 levels of {report_type} reports have been successfully generated in '{output_dir}' folder!")
        
        # 6. Calculate and save final ranking
        final_scores, ranking = calculate_final_ranking(all_level_scores, models, output_dir)
        print(f"\n🏆 {report_type} final ranking has been saved to '{output_dir}/final_ranking.txt'")

if __name__ == "__main__":
    # Set file paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    scores_file = os.path.join(script_dir, 'scores.json')
    scores_scale_file = os.path.join(script_dir, 'scores_scale.json')
    
    # Check if input file exists
    if not os.path.exists(scores_file):
        print(f"❌ Error: Cannot find input file {scores_file}")
        exit()
    
    print("🚀 Starting complete report generation process...")
    
    # 1. Generate original reports (read scores.json)
    generate_reports_for_scores(scores_file, 'reports', "Original")
    
    # 2. Generate normalized score file
    print("\n🔄 Starting to generate normalized score file...")
    min_max_scale_scores(scores_file, scores_scale_file)
    
    # 3. Generate normalized reports (read scores_scale.json)
    generate_reports_for_scores(scores_scale_file, 'reports_scale', "Normalized")
    
    print("\n🎉 All report generation completed!")
    print(f"📁 Original reports saved in: reports/")
    print(f"📁 Normalized reports saved in: reports_scale/")
    print(f"📄 Normalized score file: {scores_scale_file}")