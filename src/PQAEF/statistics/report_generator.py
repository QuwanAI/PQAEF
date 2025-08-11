import os
import datetime
from pyexpat import model
import re
import json
from typing import Dict, Any, List
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.font_manager import FontProperties
from ..constant import constant

# Importing this module will load all analyzers and populate the registry
from .analysis import __init__ 
from .analysis.registry import ANALYSIS_REGISTRY

class ReportGenerator:
    """
    Analyzes processed data and generates comprehensive reports by dynamically loading and running configurable analyzer modules.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.output_dir = self.config['output_dir']
        self.file_prefix = self.config.get('file_prefix', 'statistics')
        # Support both field names
        self.analyses_to_run = self.config.get(
            'analyses_to_run', 
            self.config.get('analyzers', list(ANALYSIS_REGISTRY.keys()))
        )

        os.makedirs(self.output_dir, exist_ok=True)
        self.font_props = self._get_font_properties()
        plt.rcParams['axes.unicode_minus'] = False
        print(f"INFO: ReportGenerator configured to run the following analyses: {self.analyses_to_run}")
        print(f"INFO: ReportGenerator config: {self.config}")

    def _get_font_properties(self) -> FontProperties:
        """Load bundled font file for supporting CJK characters in plots."""
        try:
            font_path = os.path.join(
                os.path.dirname(__file__), '..', 'resources', 'fonts', 'NotoSansSC-Regular.ttf'
            )
            if os.path.exists(font_path):
                print(f"INFO: Found CJK font at '{font_path}'.")
                return FontProperties(fname=font_path)
            else:
                print("WARNING: CJK font not found. Chinese characters in plots may appear as squares.")
                return None
        except Exception as e:
            print(f"ERROR: An error occurred while loading the font: {e}")
            return None

    def _flatten_sample_annotations(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Flatten nested annotation dictionaries and add round counting."""
        flat_data = {}
        
        def flatten_dict(d, parent_key='', sep='.'):
            items = []
            for k, v in d.items():
                new_key = parent_key + sep + k if parent_key else k
                if isinstance(v, dict):
                    items.extend(flatten_dict(v, new_key, sep=sep).items())
                else:
                    items.append((new_key, v))
            return dict(items)

        dialogue_ann = sample.get(constant.KEY_DIALOGUE_ANNOTATION, {})
        flat_data.update(flatten_dict(dialogue_ann))
        flat_data['turn_count'] = len(sample.get(constant.KEY_DIALOGUES, []))
        return flat_data

    def _unpack_list_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Find columns containing lists and unpack them into separate columns."""
        print("INFO: Checking for list-based columns to unpack...")
        cols_to_unpack = []
        for col in df.columns:
            # Check if non-null values are lists
            first_val = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
            if isinstance(first_val, list):
                cols_to_unpack.append(col)

        for col in cols_to_unpack:
            print(f"INFO: Unpacking list-based column: '{col}'")
            unpacked_cols = pd.DataFrame(df[col].tolist(), index=df.index).add_prefix(f'{col}_')
            df = df.join(unpacked_cols)
            df = df.drop(columns=[col])
        return df

    def analyze(self, data: List[Dict[str, Any]], run_metadata: Dict[str, Any]):
        """Main method to execute all analyses and generate reports."""
        if not data:
            print("WARNING: No data provided for analysis.")
            return

        # self._write_json_file(data)

        print("\n--- Starting Statistical Analysis ---")
        
        
        # 1. Data preprocessing
        print("[Step 1/3] Pre-processing data...")
        flattened_data = [self._flatten_sample_annotations(s) for s in data]
        df = pd.DataFrame(flattened_data)
        df = self._unpack_list_columns(df)
        

    
        # 2. Run configured analyzers
        print("[Step 2/3] Running configured analyzers...")
        all_results = []
        for name in self.analyses_to_run:
            if name not in ANALYSIS_REGISTRY:
                print(f"WARNING: Analyzer '{name}' is specified in config but not registered. Skipping.")
                continue
            
            AnalyzerClass = ANALYSIS_REGISTRY[name]
            analyzer_instance = AnalyzerClass(self.config, self.output_dir, self.font_props, self.file_prefix)
            result = analyzer_instance.analyze(df, data)
            
            all_results.append(result)
    
        # 3. Write unified report
        print("[Step 3/3] Compiling final report...")
        self._write_report_file(data, run_metadata, all_results)
        
        print("--- Statistical Analysis Finished ---")

    def _write_report_file(self, data, run_metadata, all_results: List[Dict[str, Any]]):
        """Compile all analysis results into a single text file."""
        report_path = os.path.join(self.output_dir, f"{self.file_prefix}_report.txt")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            # Report header
            f.write("="*80 + "\n")
            f.write(" PQAEF Modular Statistical Report\n")
            f.write("="*80 + "\n")
            f.write(f"Report Generated On: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Samples Analyzed: {len(data)}\n\n")

            # Run metadata
            f.write("-" * 30 + " Tasks Performed " + "-"*31 + "\n")
            for i, task_info in enumerate(run_metadata.get('tasks_run', [])):
                f.write(f"  {i+1}. {task_info['task_class']}\n")
            f.write("\n")

            # Summarize all plots
            all_plots = [plot for res in all_results for plot in res.get("plots", [])]
            if all_plots:
                f.write("-" * 32 + " Generated Plots " + "-"*31 + "\n")
                f.write(f"  Plots are saved in: {os.path.abspath(self.output_dir)}\n")
                for path in all_plots:
                    f.write(f"  - {os.path.basename(path)}\n")
                f.write("\n")

            # Dynamically write results from each analyzer
            for i, result in enumerate(all_results):
                f.write("="*80 + "\n")
                f.write(f" Part {i+1}: {result.get('title', 'Untitled Analysis')}\n")
                f.write("="*80 + "\n")
                
                if result.get('summary'):
                    f.write(result['summary'])
                    f.write("\n\n")

                if result.get('data_tables'):
                    for table_name, table_df in result['data_tables'].items():
                        f.write(f"--- {table_name} ---\n")
                        with pd.option_context('display.width', 120, 'display.max_rows', 100, 'display.float_format', '{:.4f}'.format):
                           f.write(table_df.to_string())
                        f.write("\n\n")

        print(f"INFO: Full statistical report saved to {report_path}")
