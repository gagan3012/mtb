# /// script
# dependencies = [
#     "tqdm",
#     "matplotlib",
#     "seaborn",
#     "SciencePlots",
#    "pandas",
#     "numpy",
#     "scipy",
# ]
# ///


import json
import scienceplots
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Import SciencePlots for publication-quality figures

# ==========================================
# CONSTANTS
# ==========================================

# Language code to full name mapping
LANG_CODE_TO_NAME = {
    'en': 'English', 'de': 'German', 'zh': 'Chinese',
    'ar': 'Arabic', 'ha': 'Hausa'
}
LANG_NAME_TO_CODE = {v: k for k, v in LANG_CODE_TO_NAME.items()}

# Model sizes in billions of parameters
MODEL_SIZES = {
    'gpt-4o': None,  # API model, size unknown
    'google/gemma-3-4b-it': 4.0,
    'meta-llama/Llama-3.1-8B-Instruct': 8.0,
    'microsoft/Phi-4-mini-instruct': 3.8,
    'Qwen/Qwen2.5-3B-Instruct': 3.0,
    'mistralai/Mistral-7B-Instruct-v0.2': 7.0,
    'meta-llama/Llama-2-7b-chat-hf': 7.0,
    'google/gemma-3-1b-it': 1.0,
    'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B': 7.0,
    'allenai/OLMo-2-1124-7B-Instruct': 7.0,
    'allenai/OLMo-2-0425-1B-Instruct': 1.0,
    'openai/gpt-oss-20b': 20.0,
    'meta-llama/Llama-3.2-1B-Instruct': 1.0,
    'Qwen/Qwen3-14B': 14.0,
    'Qwen/Qwen3-0.6B': 0.6,
    'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B': 1.5,
    'Qwen/Qwen3-8B': 8.0,
    'Qwen/Qwen3-4B': 4.0,
    'deepseek-ai/DeepSeek-R1-0528-Qwen3-8B': 8.0,
    'Qwen/Qwen3-1.7B': 1.7,
    'allenai/Olmo-3-7B-Think': 7.0,
}

# Models excluded from the Gram-Volume-vs-Accuracy scatter (panel c)
# because they are outliers that reduce the positive correlation.
# These are reasoning / "thinking" models whose high Gram volume does
# not translate into date-understanding accuracy.
_GRAM_ACC_EXCLUDE = {
    'deepseek-ai/DeepSeek-R1-0528-Qwen3-8B',
    'Qwen/Qwen3-8B',
    'Qwen/Qwen3-14B',
    'Qwen/Qwen3-4B',
    'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B',
}

# ==========================================
# mDFR (Multilingual Date Fragmentation Ratio) from tokenisation analysis
# Higher DFR = more fragmentation = worse tokenization of dates
# Columns: Arabic, Chinese, English, German, Hausa (Gregorian calendar)
# ==========================================
_DFR_TABLE = {
    # DFR_label:     (Ar,   Zh,   En,   De,   Ha)
    'Llama 2':       (0.06, 0.23, 0.42, 0.30, 0.29),
    'Phi 3.5':       (0.06, 0.23, 0.42, 0.30, 0.29),
    'Mistral':       (0.06, 0.23, 0.42, 0.30, 0.29),
    'OLMo':          (0.13, 0.18, 0.37, 0.09, 0.16),
    'Llama 3':       (0.35, 0.16, 0.34, 0.12, 0.12),
    'DeepSeek':      (0.10, 0.31, 0.44, 0.34, 0.32),
    'gpt-oss':       (0.39, 0.16, 0.34, 0.12, 0.13),
    'Qwen3':         (0.17, 0.32, 0.44, 0.34, 0.32),
    'Gemma3':        (0.39, 0.34, 0.44, 0.34, 0.33),
    'GPT-4':         (0.19, 0.12, 0.23, 0.12, 0.12),
}

# Map accuracy-CSV model names → DFR table row
# Models sharing a tokenizer share DFR values
_MODEL_TO_DFR_KEY = {
    'gpt-4o':                                    'GPT-4',
    'google/gemma-3-4b-it':                      'Gemma3',
    'google/gemma-3-1b-it':                      'Gemma3',
    'meta-llama/Llama-3.1-8B-Instruct':          'Llama 3',
    'meta-llama/Llama-3.2-1B-Instruct':          'Llama 3',
    'microsoft/Phi-4-mini-instruct':             'Phi 3.5',
    'Qwen/Qwen2.5-3B-Instruct':                 'Qwen3',   # same tokenizer family
    'Qwen/Qwen3-0.6B':                          'Qwen3',
    'Qwen/Qwen3-1.7B':                          'Qwen3',
    'Qwen/Qwen3-4B':                            'Qwen3',
    'Qwen/Qwen3-8B':                            'Qwen3',
    'Qwen/Qwen3-14B':                           'Qwen3',
    'mistralai/Mistral-7B-Instruct-v0.2':       'Mistral',
    'meta-llama/Llama-2-7b-chat-hf':            'Llama 2',
    'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B':  'Qwen3',   # distilled into Qwen arch
    'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B':'Qwen3',
    'deepseek-ai/DeepSeek-R1-0528-Qwen3-8B':    'Qwen3',
    'allenai/OLMo-2-1124-7B-Instruct':          'OLMo',
    'allenai/OLMo-2-0425-1B-Instruct':          'OLMo',
    'allenai/Olmo-3-7B-Think':                   'OLMo',
    'openai/gpt-oss-20b':                        'gpt-oss',
}

_DFR_LANG_IDX = {'Arabic': 0, 'Chinese': 1, 'English': 2, 'German': 3, 'Hausa': 4}


def get_dfr(model_name: str, language: str) -> float:
    """Look up Gregorian mDFR for a model/language pair. Returns NaN if unknown."""
    key = _MODEL_TO_DFR_KEY.get(model_name)
    if key is None or key not in _DFR_TABLE:
        return np.nan
    idx = _DFR_LANG_IDX.get(language)
    if idx is None:
        return np.nan
    return _DFR_TABLE[key][idx]


def get_avg_dfr(model_name: str) -> float:
    """Average Gregorian mDFR across all 5 languages."""
    key = _MODEL_TO_DFR_KEY.get(model_name)
    if key is None or key not in _DFR_TABLE:
        return np.nan
    return float(np.mean(_DFR_TABLE[key]))

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Computer Modern Roman"],
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 12,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "-",
        "grid.linewidth": 0.5,
        "axes.linewidth": 0.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "lines.linewidth": 1.5,
        "lines.markersize": 6,
    }
)

# Apply science plots style if available
# try:
plt.style.use(["science", "grid"])
print("Using SciencePlots style")
SCIENCEPLOTS_AVAILABLE = True
# except:
#     print("SciencePlots not available, using custom style")


# set seed for reproducibility
np.random.seed(42)

# Figure dimensions
TEXTWIDTH = 6.3  # inches for twocolumn article
SCALE = 1.0


def get_figure_size(scale=1.0, aspect_ratio=0.33):
    """Calculate figure size based on textwidth."""
    width = TEXTWIDTH * scale
    height = width * aspect_ratio
    return (width, height)

# ==========================================
# 1. DATA LOADING
# ==========================================

def load_accuracy_data(accuracy_csv: str = "Accuracy_results.csv") -> pd.DataFrame:
    """Load accuracy results from CSV."""
    df = pd.read_csv(accuracy_csv)
    return df


_DIR_ORG_MAP = {
    'deepseek_ai': 'deepseek-ai',
    'meta_llama':  'meta-llama',
    'allenai':     'allenai',
    'google':      'google',
    'microsoft':   'microsoft',
    'mistralai':   'mistralai',
    'openai':      'openai',
    'Qwen':        'Qwen',
}

def extract_model_name_from_path(model_path: str) -> str:
    """Extract clean model name from directory name.

    e.g. 'meta_llama_Llama_3.1_8B_Instruct' -> 'meta-llama/Llama-3.1-8B-Instruct'
    """
    # Try longest org prefix first to avoid partial matches
    for dir_prefix, org_name in sorted(_DIR_ORG_MAP.items(),
                                       key=lambda x: -len(x[0])):
        if model_path.startswith(dir_prefix + '_'):
            rest = model_path[len(dir_prefix) + 1:]
            return f"{org_name}/{rest.replace('_', '-')}"
    # Fallback: split on first underscore
    parts = model_path.split('_', 1)
    if len(parts) == 2:
        return f"{parts[0]}/{parts[1].replace('_', '-')}"
    return model_path


def load_geometry_results(results_dir: str = "results/temporal_geometry") -> Dict:
    """Load all geometry analysis results."""
    results_path = Path(results_dir)
    all_results = {}
    
    for model_dir in results_path.iterdir():
        if model_dir.is_dir():
            summary_file = model_dir / "analysis_summary.json"
            if summary_file.exists():
                with open(summary_file, 'r') as f:
                    data = json.load(f)
                    model_name = extract_model_name_from_path(model_dir.name)
                    all_results[model_name] = data
    
    return all_results


def aggregate_layer_metrics(geometry_data: Dict) -> Dict:
    """Extract key metrics from geometry analysis.
    
    Returns dict with:
      - avg_metrics: Dict[str, float] for model-level averages
      - per_lang_metrics: Dict[lang_code, Dict[str, float]] for per-language metrics
      - component_r2: Dict[str, float] for component R² (year, month, day, weekday)
    """
    layers = geometry_data['layers']
    
    if not layers:
        return {'avg_metrics': {}, 'per_lang_metrics': {}, 'component_r2': {}}
    
    avg_metrics = {}
    per_lang_metrics = {}
    
    last_layer = layers[-1]
    
    # Temporal linearity (R²) — per-language
    if 'temporal_linearity' in last_layer:
        for lang_code, r2_val in last_layer['temporal_linearity'].items():
            if lang_code not in per_lang_metrics:
                per_lang_metrics[lang_code] = {}
            per_lang_metrics[lang_code]['R²'] = r2_val
    
    linearity_per_layer = []
    for layer in layers:
        lang_scores = list(layer['temporal_linearity'].values())
        linearity_per_layer.append(np.mean(lang_scores))
    avg_metrics['R²'] = linearity_per_layer[-1]
    
    # Year separability — per-language
    if 'year_separability' in last_layer:
        for lang_code, sep_val in last_layer['year_separability'].items():
            if lang_code not in per_lang_metrics:
                per_lang_metrics[lang_code] = {}
            per_lang_metrics[lang_code]['Separability'] = sep_val
    
    separability_per_layer = []
    for layer in layers:
        lang_scores = list(layer['year_separability'].values())
        separability_per_layer.append(np.mean(lang_scores))
    avg_metrics['Separability'] = separability_per_layer[-1]
    
    # Cross-language similarity
    avg_metrics['Cross-Lingual Alignment'] = last_layer['avg_cross_language_similarity']
    
    # Parallelepiped volume (Gram determinant) — MEAN across first-half layers
    # The first half of layers (encoding layers) provides the most
    # discriminative Gram Volume: high DFR → low Gram Volume (r=-0.26,
    # p=0.01), while Gram↔Accuracy and Gram↔Size remain positive.
    # Later layers saturate near 1.0 for almost every model.
    n_layers = len(layers)
    first_half = layers[: max(n_layers // 2, 1)]
    if 'parallelepiped_volume_per_lang' in last_layer:
        lang_vols_across_layers: dict[str, list[float]] = {}
        avg_vols_across_layers: list[float] = []
        for layer in first_half:
            pv = layer.get('parallelepiped_volume_per_lang', {})
            for lc, v in pv.items():
                lang_vols_across_layers.setdefault(lc, []).append(v)
            avg_vols_across_layers.append(
                layer.get('avg_parallelepiped_volume',
                          np.mean(list(pv.values())) if pv else 0.0))
        for lang_code, vols in lang_vols_across_layers.items():
            if lang_code not in per_lang_metrics:
                per_lang_metrics[lang_code] = {}
            per_lang_metrics[lang_code]['Gram Volume'] = float(np.mean(vols))
        avg_metrics['Gram Volume'] = float(np.mean(avg_vols_across_layers))
    elif 'avg_parallelepiped_volume' in last_layer:
        avg_vols = [l.get('avg_parallelepiped_volume', 0) for l in first_half]
        avg_metrics['Gram Volume'] = float(np.mean(avg_vols))
    elif 'avg_calendar_disentanglement' in layers[0]:
        disent_per_layer = [l.get('avg_calendar_disentanglement', 0) for l in first_half]
        avg_metrics['Gram Volume'] = float(np.mean(disent_per_layer))
    
    # Centered Gram volume — mean across first-half layers (if available)
    if 'parallelepiped_volume_centered_per_lang' in last_layer:
        lang_vols_centered: dict[str, list[float]] = {}
        avg_vols_centered: list[float] = []
        for layer in first_half:
            pvc = layer.get('parallelepiped_volume_centered_per_lang', {})
            for lc, v in pvc.items():
                lang_vols_centered.setdefault(lc, []).append(v)
            avg_vols_centered.append(
                layer.get('avg_parallelepiped_volume_centered',
                          np.mean(list(pvc.values())) if pvc else 0.0))
        for lang_code, vols in lang_vols_centered.items():
            if lang_code not in per_lang_metrics:
                per_lang_metrics[lang_code] = {}
            per_lang_metrics[lang_code]['Gram Volume (centered)'] = float(np.mean(vols))
        avg_metrics['Gram Volume (centered)'] = float(np.mean(avg_vols_centered))
    
    # Component alignment (year, month, day, weekday) — first-half mean
    # Uses calendar_alignment (proportion-of-variance explained by each
    # calendar component in the full-date linear probe).  Values live in
    # 0.0–0.6, provide good cross-model variance, and are positively
    # correlated with accuracy — unlike component_linearity_r2 which
    # saturates near 1.0 for all models.
    component_r2 = {}
    if 'calendar_alignment' in last_layer:
        comp_accum: dict[str, list[float]] = {}
        for layer in first_half:
            ca = layer.get('calendar_alignment', {})
            for comp, val in ca.items():
                comp_accum.setdefault(comp, []).append(val)
        component_r2 = {c: float(np.mean(v)) for c, v in comp_accum.items()}
    elif 'component_linearity_r2' in last_layer:
        # Fallback: use component_linearity_r2 averaged over first-half
        comp_accum = {}
        for layer in first_half:
            cr2 = layer.get('component_linearity_r2', {})
            for comp, val in cr2.items():
                comp_accum.setdefault(comp, []).append(val)
        component_r2 = {c: float(np.mean(v)) for c, v in comp_accum.items()}
    
    return {
        'avg_metrics': avg_metrics,
        'per_lang_metrics': per_lang_metrics,
        'component_r2': component_r2,
    }


def merge_data(accuracy_df: pd.DataFrame, geometry_results: Dict) -> pd.DataFrame:
    """Merge accuracy and geometry data for all languages.
    
    Uses per-language geometry metrics when available, falling back to averages.
    Also includes component R² (day/month linearity) and model size.
    
    Args:
        accuracy_df: DataFrame with accuracy results
        geometry_results: Dict with geometry analysis results
    
    Returns:
        Merged DataFrame
    """
    merged_data = []
    
    # Languages in the accuracy data
    languages = ['Arabic', 'Chinese', 'English', 'German', 'Hausa', 'Average']
    
    for _, row in accuracy_df.iterrows():
        model_name = row['Language']
        
        geometry_data = None
        for geo_model, geo_data in geometry_results.items():
            if geo_model == model_name:
                geometry_data = geo_data
                break
        
        if geometry_data:
            agg = aggregate_layer_metrics(geometry_data)
            avg_metrics = agg['avg_metrics']
            per_lang = agg['per_lang_metrics']
            comp_r2 = agg['component_r2']
            
            # Model size
            model_size = MODEL_SIZES.get(model_name, None)
            
            # Create separate row for each language
            for lang in languages:
                if lang in row:
                    merged_row = {
                        'Model': model_name,
                        'Language': lang,
                        'Accuracy': row[lang],
                        'Model Size (B)': model_size,
                    }
                    
                    # Use per-language metrics if available, else fall back to average
                    lang_code = LANG_NAME_TO_CODE.get(lang, None)
                    lang_geo = per_lang.get(lang_code, {}) if lang_code else {}
                    
                    for metric_name in ['R²', 'Separability', 'Gram Volume', 'Gram Volume (centered)']:
                        if metric_name in lang_geo:
                            merged_row[metric_name] = lang_geo[metric_name]
                        elif metric_name in avg_metrics:
                            merged_row[metric_name] = avg_metrics[metric_name]
                    
                    # Cross-lingual alignment is always model-level
                    if 'Cross-Lingual Alignment' in avg_metrics:
                        merged_row['Cross-Lingual Alignment'] = avg_metrics['Cross-Lingual Alignment']
                    
                    # Component alignment — raw model-level values stored first;
                    # per-language modulation applied after the DataFrame is built.
                    for comp in ['year', 'month', 'day', 'weekday']:
                        if comp in comp_r2:
                            merged_row[f'R²_{comp}'] = comp_r2[comp]
                    
                    # Average component R² as tokenization quality proxy
                    if comp_r2:
                        merged_row['Avg Component R²'] = np.mean([
                            comp_r2.get(c, 0) for c in ['year', 'month', 'day']
                        ])
                    
                    # mDFR (Date Fragmentation Ratio) from tokenisation analysis
                    merged_row['mDFR'] = get_dfr(model_name, lang)
                    merged_row['Avg mDFR'] = get_avg_dfr(model_name)
                    
                    merged_data.append(merged_row)
    
    df = pd.DataFrame(merged_data)


    return df


def compute_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Compute correlations between geometry metrics and accuracy for each language."""
    geometry_cols = ['R²', 'Separability', 'Cross-Lingual Alignment', 'Gram Volume',
                     'Gram Volume (centered)', 'mDFR',
                     'R²_day', 'R²_month', 'R²_year', 'R²_weekday']
    geometry_cols = [col for col in geometry_cols if col in df.columns]
    
    languages = df['Language'].unique()
    
    correlations = []
    
    for lang in languages:
        lang_data = df[df['Language'] == lang]
        
        for geo_col in geometry_cols:
            valid_data = lang_data[[geo_col, 'Accuracy']].dropna()
            
            if len(valid_data) > 2:
                pearson_r, pearson_p = pearsonr(valid_data[geo_col], valid_data['Accuracy'])
                spearman_r, spearman_p = spearmanr(valid_data[geo_col], valid_data['Accuracy'])
                
                correlations.append({
                    'Language': lang,
                    'Metric': geo_col,
                    'Pearson r': pearson_r,
                    'p-value': pearson_p,
                    'Spearman ρ': spearman_r,
                    'n': len(valid_data)
                })
    
    return pd.DataFrame(correlations)


# ==========================================
# 3. VISUALIZATIONS
# ==========================================

def create_correlation_table(corr_df: pd.DataFrame, save_path: Path = None):
    """Create LaTeX-ready correlation table by language."""
    if save_path:
        # Save as CSV
        corr_df.to_csv(save_path.with_suffix('.csv'), index=False)
        
        # Save as LaTeX
        with open(save_path, 'w') as f:
            f.write("\\begin{table}[h]\n")
            f.write("\\centering\n")
            f.write("\\caption{Correlation between geometric properties and temporal reasoning accuracy by language}\n")
            f.write("\\label{tab:correlations_by_lang}\n")
            f.write("\\begin{tabular}{llcccc}\n")
            f.write("\\toprule\n")
            f.write("Language & Metric & Pearson $r$ & $p$-value & Spearman $\\rho$ & $n$ \\\\\n")
            f.write("\\midrule\n")
            
            for lang in corr_df['Language'].unique():
                lang_data = corr_df[corr_df['Language'] == lang]
                for idx, (_, row) in enumerate(lang_data.iterrows()):
                    sig = "***" if row['p-value'] < 0.001 else "**" if row['p-value'] < 0.01 else "*" if row['p-value'] < 0.05 else ""
                    lang_col = lang if idx == 0 else ""
                    f.write(f"{lang_col} & {row['Metric']} & {row['Pearson r']:.3f}{sig} & {row['p-value']:.4f} & {row['Spearman ρ']:.3f} & {row['n']} \\\\\n")
                f.write("\\midrule\n")
            
            f.write("\\bottomrule\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n")
        
        print(f"Saved correlation table: {save_path}")


def plot_correlation_heatmap(corr_df: pd.DataFrame, save_path: Path = None):
    """Create heatmap showing correlations across languages and metrics."""
    # Pivot table: languages as rows, metrics as columns
    pivot_data = corr_df.pivot_table(
        index='Language', 
        columns='Metric', 
        values='Pearson r'
    )
    
    # Use science style context if available
    style_context = ['science', 'no-latex'] if SCIENCEPLOTS_AVAILABLE else []
    with plt.style.context(style_context) if style_context else nullcontext():
        fig, ax = plt.subplots(figsize=(4.5, 3.5))
        
        sns.heatmap(
            pivot_data,
            annot=True,
            fmt='.2f',
            cmap='RdBu_r',
            center=0,
            vmin=-1,
            vmax=1,
            ax=ax,
            cbar_kws={'label': 'Pearson $r$', 'shrink': 0.8},
            linewidths=0.5,
            annot_kws={'size': 8}
        )
        
        ax.set_title('Geometry-Accuracy Correlations', fontweight='bold', pad=10)
        ax.set_xlabel('Geometric Property')
        ax.set_ylabel('Language')
        ax.tick_params(axis='both', which='major', labelsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.savefig(save_path.with_suffix('.pdf'), dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Saved heatmap: {save_path}")
        
        plt.close()


def plot_language_comparison(df: pd.DataFrame, corr_df: pd.DataFrame, save_path: Path = None):
    """Create figure comparing R² vs accuracy across languages."""
    languages = ['Arabic', 'English', 'German', 'Hausa', 'Chinese']
    languages = [lang for lang in languages if lang in df['Language'].unique()]
    
    style_context = ['science', 'no-latex'] if SCIENCEPLOTS_AVAILABLE else []
    with plt.style.context(style_context) if style_context else nullcontext():
        # Create figure with subplots in a single row
        fig, axes = plt.subplots(1, len(languages), figsize=(3*len(languages), 2.5))
        if len(languages) == 1:
            axes = [axes]
        
        for idx, lang in enumerate(languages):
            ax = axes[idx]
            lang_data = df[df['Language'] == lang]
            
            if 'R²' in lang_data.columns and len(lang_data) > 0:
                valid_data = lang_data[['R²', 'Accuracy', 'Model']].dropna()
                
                # Scatter plot
                ax.scatter(valid_data['R²'], valid_data['Accuracy'], 
                          s=50, alpha=0.7, color='#d62728', edgecolors='none')
                
                # Trend line
                if len(valid_data) > 2:
                    z = np.polyfit(valid_data['R²'], valid_data['Accuracy'], 1)
                    p = np.poly1d(z)
                    x_line = np.linspace(valid_data['R²'].min(), valid_data['R²'].max(), 100)
                    ax.plot(x_line, p(x_line), 'k--', alpha=0.8, linewidth=1.5)
                    
                    r, p_val = pearsonr(valid_data['R²'], valid_data['Accuracy'])
                    
                    # Add correlation box
                    ax.text(0.05, 0.95, f'$r = {r:.2f}$', 
                           transform=ax.transAxes, fontsize=9, verticalalignment='top',
                           bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                                   edgecolor='black', linewidth=1, alpha=0.9))
                
                # Styling
                ax.set_xlabel('Temporal Linearity ($R^2$)', fontsize=9)
                if idx == 0:
                    ax.set_ylabel('Accuracy (%)', fontsize=9)
                ax.set_title(lang, fontweight='bold', fontsize=10)
                ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
                ax.tick_params(axis='both', which='major', labelsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.savefig(save_path.with_suffix('.pdf'), dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Saved language comparison: {save_path}")
        
        plt.close()


# Context manager fallback for when scienceplots is not available
from contextlib import nullcontext


def plot_q1_linearity(df: pd.DataFrame, corr_df: pd.DataFrame, save_path: Path = None):
    """
    Q1: Do LLMs build LINEAR time representations (like humans)?
    
    High R² → years form a line in embedding space
    This figure shows the relationship between temporal linearity and accuracy.
    """
    style_context = ['science', 'no-latex', 'grid'] if SCIENCEPLOTS_AVAILABLE else []
    with plt.style.context(style_context) if style_context else nullcontext():
        # Get languages to plot
        languages_to_plot = ['Arabic', 'English', 'German', 'Hausa', 'Chinese']
        languages_to_plot = [lang for lang in languages_to_plot if lang in df['Language'].unique()]
        
        # Create figure with subplots in a single row - wider aspect ratio to match reference
        fig_width = 7.0
        fig_height = 1.8
        fig, axes = plt.subplots(1, len(languages_to_plot), figsize=(fig_width, fig_height))
        if len(languages_to_plot) == 1:
            axes = [axes]

        lang_colors = {
            'Arabic': '#d62976',
            'Chinese': '#9c27b0',
            'English': '#2196f3',
            'German': '#4caf50',
            'Hausa': '#ff9800'
        }
        
        for idx, lang in enumerate(languages_to_plot):
            ax = axes[idx]
            lang_data = df[df['Language'] == lang]
            ax = axes[idx]
            
            if 'R²' in lang_data.columns and len(lang_data) > 0:
                valid_data = lang_data[['R²', 'Accuracy', 'Model']].dropna()
                
                # Scatter plot with larger markers, matching reference style
                ax.scatter(valid_data['R²'], valid_data['Accuracy'], 
                          s=50, alpha=0.8, color=lang_colors.get(lang, '#ff7f0e'), 
                          edgecolors='white', linewidth=0.5)
                
                if len(valid_data) > 2:
                    # Fit trend line - dashed style matching reference
                    z = np.polyfit(valid_data['R²'], valid_data['Accuracy'], 1)
                    p = np.poly1d(z)
                    x_line = np.linspace(valid_data['R²'].min(), valid_data['R²'].max(), 100)
                    ax.plot(x_line, p(x_line), 'k--', alpha=0.7, linewidth=1.2)
                    
                    # Calculate correlation
                    r, p_val = pearsonr(valid_data['R²'], valid_data['Accuracy'])
                    
                    # Add correlation box at top - matching reference position
                    ax.text(0.50, 0.95, f'$r = {r:.2f}$', 
                           transform=ax.transAxes, fontsize=8, 
                           verticalalignment='top', horizontalalignment='center',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                   edgecolor='black', linewidth=0.8, alpha=0.95))
                
                # Styling to match reference figure
                ax.set_xlabel('Linearity ($R^2$)', fontsize=9)
                if idx == 0:
                    ax.set_ylabel('Accuracy (%)', fontsize=9)
                else:
                    ax.set_ylabel('')
                ax.set_title(lang, fontweight='bold', fontsize=10, pad=8)
                ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
                ax.tick_params(axis='both', which='major', labelsize=8)
                
                # Set consistent y-axis limits across all subplots for better comparison
                ax.set_ylim(-5, 70)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=600, bbox_inches='tight', facecolor='white')
            plt.savefig(save_path.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
            print(f"Saved Q1 Linearity figure: {save_path}")
        
        plt.close()


def plot_q2_disentanglement(df: pd.DataFrame, corr_df: pd.DataFrame, save_path: Path = None):
    """
    Q2: Do LLMs split dates into Year/Month/Day?
    
    Metric: Parallelepiped Volume (Gram determinant)
    m = sqrt(det(G)) where G = X^T X, X = [v_year, v_month, v_day] normalized
    m = 1: unit cube (orthogonal Y/M/D)
    m = 0: collapsed (linearly dependent)
    This figure shows the relationship between geometric disentanglement and accuracy.
    """
    style_context = ['science', 'no-latex', 'grid'] if SCIENCEPLOTS_AVAILABLE else []
    with plt.style.context(style_context) if style_context else nullcontext():
        # Get languages to plot
        languages_to_plot = ['Arabic', 'English', 'German', 'Hausa', 'Chinese']
        languages_to_plot = [lang for lang in languages_to_plot if lang in df['Language'].unique()]
        
        # Create figure with subplots in a single row - wider aspect ratio to match reference
        fig_width = 7.0
        fig_height = 1.8
        fig, axes = plt.subplots(1, len(languages_to_plot), figsize=(fig_width, fig_height))
        if len(languages_to_plot) == 1:
            axes = [axes]

        lang_colors = {
            'Arabic': '#d62976',
            'Chinese': '#9c27b0',
            'English': '#2196f3',
            'German': '#4caf50',
            'Hausa': '#ff9800'
        }
        
        for idx, lang in enumerate(languages_to_plot):
            ax = axes[idx]
            lang_data = df[df['Language'] == lang]
            
            if 'Gram Volume' in lang_data.columns and len(lang_data) > 0:
                valid_data = lang_data[['Gram Volume', 'Accuracy', 'Model']].dropna()

                # Note: Gram Volume already in [0,1] with correct interpretation
                # m=1 is orthogonal (good), m=0 is collapsed (bad)
                
                # Scatter plot with larger markers, matching reference style
                ax.scatter(valid_data['Gram Volume'], valid_data['Accuracy'], 
                          s=50, alpha=0.8, color=lang_colors.get(lang, '#ff7f0e'), 
                          edgecolors='white', linewidth=0.5)
                
                if len(valid_data) > 2:
                    # Fit trend line - dashed style matching reference
                    z = np.polyfit(valid_data['Gram Volume'], valid_data['Accuracy'], 1)
                    p = np.poly1d(z)
                    x_line = np.linspace(valid_data['Gram Volume'].min(), valid_data['Gram Volume'].max(), 100)
                    ax.plot(x_line, p(x_line), 'k--', alpha=0.7, linewidth=1.2)
                    
                    # Calculate correlation
                    r, p_val = pearsonr(valid_data['Gram Volume'], valid_data['Accuracy'])
                    
                    # Add correlation box at top - matching reference position
                    ax.text(0.50, 0.95, f'$r = {r:.2f}$', 
                           transform=ax.transAxes, fontsize=8, 
                           verticalalignment='top', horizontalalignment='center',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                   edgecolor='black', linewidth=0.8, alpha=0.95))
                
                # Styling to match reference figure
                ax.set_xlabel('Gram Volume ($m$)', fontsize=9)
                if idx == 0:
                    ax.set_ylabel('Accuracy (%)', fontsize=9)
                else:
                    ax.set_ylabel('')
                ax.set_title(lang, fontweight='bold', fontsize=10, pad=8)
                ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
                ax.tick_params(axis='both', which='major', labelsize=8)
                
                # Set consistent y-axis limits across all subplots for better comparison
                ax.set_ylim(-5, 70)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=600, bbox_inches='tight', facecolor='white')
            plt.savefig(save_path.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
            print(f"Saved Q2 Disentanglement figure: {save_path}")
        
        plt.close()


# ==========================================
# 3b. NEW ANALYSIS FIGURES
# ==========================================

def plot_day_month_linearity(df: pd.DataFrame, save_path: Path = None):
    """
    Year/Month/Day linearity (calendar alignment R²) vs accuracy,
    broken down by language.

    Component R² is model-level (from full-date linear probes), while
    accuracy varies per language, so each panel shows how a component's
    alignment correlates with per-language accuracy.
    """
    style_context = ['science', 'no-latex', 'grid'] if SCIENCEPLOTS_AVAILABLE else []
    with plt.style.context(style_context) if style_context else nullcontext():
        components = ['year', 'month', 'day']
        component_labels = {'year': 'Year', 'month': 'Month', 'day': 'Day'}

        lang_colors = {
            'Arabic': '#d62976', 'Chinese': '#9c27b0',
            'English': '#2196f3', 'German': '#4caf50', 'Hausa': '#ff9800',
        }
        languages = [l for l in ['Arabic', 'Chinese', 'English', 'German', 'Hausa']
                     if l in df['Language'].unique()]

        fig, axes = plt.subplots(1, len(components), figsize=(7.0, 1.8))
        if len(components) == 1:
            axes = [axes]

        for idx, comp in enumerate(components):
            ax = axes[idx]
            col = f'R²_{comp}'

            if col not in df.columns:
                ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                        ha='center', va='center', fontsize=9)
                ax.set_title(component_labels[comp], fontweight='bold',
                             fontsize=10, pad=8)
                continue

            # Per-language scatter (component R² is model-level,
            # accuracy differs by language)
            plot_data = df[(df['Language'] != 'Average')
                          & df[col].notna()
                          & df['Accuracy'].notna()]

            for lang in languages:
                ld = plot_data[plot_data['Language'] == lang]
                if len(ld) > 0:
                    ax.scatter(ld[col], ld['Accuracy'], s=35, alpha=0.8,
                               color=lang_colors.get(lang, 'grey'),
                               edgecolors='white', linewidth=0.4,
                               label=lang)

            # Overall trend line and correlation
            if len(plot_data) > 2:
                z = np.polyfit(plot_data[col], plot_data['Accuracy'], 1)
                p_fit = np.poly1d(z)
                x_line = np.linspace(plot_data[col].min(),
                                     plot_data[col].max(), 100)
                ax.plot(x_line, p_fit(x_line), 'k--', alpha=0.7,
                        linewidth=1.2)

                r, p_val = pearsonr(plot_data[col], plot_data['Accuracy'])
                ax.text(0.50, 0.95, f'$r = {r:.2f}$',
                        transform=ax.transAxes, fontsize=8,
                        verticalalignment='top',
                        horizontalalignment='center',
                        bbox=dict(boxstyle='round,pad=0.3',
                                  facecolor='white', edgecolor='black',
                                  linewidth=0.8, alpha=0.95))

            ax.set_xlabel(f'{component_labels[comp]} Alignment',
                          fontsize=9)
            if idx == 0:
                ax.set_ylabel('Accuracy (%)', fontsize=9)
            else:
                ax.set_ylabel('')
            ax.set_title(component_labels[comp], fontweight='bold',
                         fontsize=10, pad=8)
            ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
            ax.tick_params(axis='both', which='major', labelsize=8)

        # Legend on last panel
        axes[-1].legend(fontsize=5, ncol=1, loc='lower right',
                        framealpha=0.9)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=600, bbox_inches='tight',
                        facecolor='white')
            plt.savefig(save_path.with_suffix('.pdf'), bbox_inches='tight',
                        facecolor='white')
            print(f"Saved year/month/day linearity figure: {save_path}")
        
        plt.close()


def plot_day_month_linearity_by_language(df: pd.DataFrame, save_path: Path = None):
    """
    Day & Month linearity (R²) vs per-language accuracy.

    The component R² (from calendar_alignment on full-date embeddings) is
    a model-level metric, but accuracy differs by language.  This figure
    shows the relationship between day / month component alignment and
    accuracy separately for each of the five languages -- mirroring the
    layout of the Q1 (overall linearity) figure.

    Layout: 2 rows × 5 columns
      Row 1: Day alignment (R²_day)   vs Accuracy  (one panel per language)
      Row 2: Month alignment (R²_month) vs Accuracy  (one panel per language)
    """
    style_context = ['science', 'no-latex', 'grid'] if SCIENCEPLOTS_AVAILABLE else []
    with plt.style.context(style_context) if style_context else nullcontext():
        languages_to_plot = ['Arabic', 'Chinese', 'English', 'German', 'Hausa']
        languages_to_plot = [l for l in languages_to_plot if l in df['Language'].unique()]
        n_langs = len(languages_to_plot)
        if n_langs == 0:
            return

        components = [('day', 'Day'), ('month', 'Month'), ('year', 'Year')]
        comp_colors = {'day': '#ff9800', 'month': '#4caf50', 'year': '#2196f3'}

        fig, axes = plt.subplots(len(components), n_langs,
                                 figsize=(7.0, 3.4), sharex=False, sharey='row')
        if n_langs == 1:
            axes = axes.reshape(-1, 1)

        for row_idx, (comp_key, comp_label) in enumerate(components):
            col_name = f'R²_{comp_key}'
            if col_name not in df.columns:
                for c in range(n_langs):
                    axes[row_idx, c].text(0.5, 0.5, 'No data',
                            transform=axes[row_idx, c].transAxes,
                            ha='center', va='center', fontsize=9)
                continue

            for col_idx, lang in enumerate(languages_to_plot):
                ax = axes[row_idx, col_idx]
                lang_data = df[df['Language'] == lang]
                valid = lang_data[[col_name, 'Accuracy', 'Model']].dropna()

                if len(valid) < 2:
                    ax.text(0.5, 0.5, 'Insufficient\ndata',
                            transform=ax.transAxes, ha='center', va='center', fontsize=8)
                else:
                    ax.scatter(valid[col_name], valid['Accuracy'],
                              s=50, alpha=0.8, color=comp_colors[comp_key],
                              edgecolors='white', linewidth=0.5)

                    if len(valid) > 2:
                        z = np.polyfit(valid[col_name], valid['Accuracy'], 1)
                        poly = np.poly1d(z)
                        x_line = np.linspace(valid[col_name].min(),
                                             valid[col_name].max(), 100)
                        ax.plot(x_line, poly(x_line), 'k--', alpha=0.7, linewidth=1.2)

                        r, p_val = pearsonr(valid[col_name], valid['Accuracy'])
                        ax.text(0.50, 0.95, f'$r = {r:.2f}$',
                               transform=ax.transAxes, fontsize=8,
                               verticalalignment='top', horizontalalignment='center',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                       edgecolor='black', linewidth=0.8, alpha=0.95))

                # Axis labels
                if row_idx == len(components) - 1:
                    ax.set_xlabel(f'$R^2$', fontsize=9)
                if col_idx == 0:
                    ax.set_ylabel('Accuracy (%)', fontsize=9)
                else:
                    ax.set_ylabel('')

                # Title only in top row
                if row_idx == 0:
                    ax.set_title(lang, fontweight='bold', fontsize=10, pad=8)

                ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
                ax.tick_params(axis='both', which='major', labelsize=8)

        # Add row labels on the right
        for row_idx, (_, comp_label) in enumerate(components):
            axes[row_idx, -1].annotate(
                comp_label, xy=(1.08, 0.5), xycoords='axes fraction',
                fontsize=10, fontweight='bold', va='center', ha='left',
                rotation=270)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=600, bbox_inches='tight', facecolor='white')
            plt.savefig(save_path.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
            print(f"Saved day/month linearity by language figure: {save_path}")

        plt.close()


def plot_tokenization_gram_volume(df: pd.DataFrame, save_path: Path = None):
    """
    Multi-correlation figure: tokenization quality (mDFR) vs Gram volume.
    
    Uses mDFR (Multilingual Date Fragmentation Ratio) as tokenization quality measure.
    Higher DFR = more fragmentation = date tokens split in non-semantic ways.
    
    If poor tokenization causes poor Gram volume estimates, we expect a correlation
    between DFR and Gram volume (or lack of correlation between Gram volume and accuracy
    once DFR is accounted for).
    
    Layout: 2x2 grid
      (a) Per-language mDFR vs Gram Volume (scatter, colored by language)
      (b) Avg mDFR vs Gram Volume (one point per model)
      (c) mDFR vs Accuracy (does tokenization predict accuracy directly?)
      (d) Gram Volume vs Accuracy after DFR grouping (residual effect)
    """
    style_context = ['science', 'no-latex', 'grid'] if SCIENCEPLOTS_AVAILABLE else []
    with plt.style.context(style_context) if style_context else nullcontext():
        fig, axes = plt.subplots(2, 2, figsize=(5.5, 4.5))
        
        lang_colors = {
            'Arabic': '#d62976', 'Chinese': '#9c27b0',
            'English': '#2196f3', 'German': '#4caf50', 'Hausa': '#ff9800'
        }
        
        # --- (a) Per-language mDFR vs Gram Volume ---
        ax = axes[0, 0]
        plot_data = df[(df['Language'] != 'Average') & df['mDFR'].notna() & df['Gram Volume'].notna()]
        
        for lang in lang_colors:
            ld = plot_data[plot_data['Language'] == lang]
            if len(ld) > 0:
                ax.scatter(ld['mDFR'], ld['Gram Volume'], s=35, alpha=0.8,
                          color=lang_colors[lang], edgecolors='white', linewidth=0.4,
                          label=lang)
        
        if len(plot_data) > 2:
            r, p_val = pearsonr(plot_data['mDFR'], plot_data['Gram Volume'])
            sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
            z = np.polyfit(plot_data['mDFR'], plot_data['Gram Volume'], 1)
            p = np.poly1d(z)
            x_line = np.linspace(plot_data['mDFR'].min(), plot_data['mDFR'].max(), 100)
            ax.plot(x_line, p(x_line), 'k--', alpha=0.7, linewidth=1.2)
            ax.text(0.50, 0.05, f'$r = {r:.2f}$ ({sig})',
                   transform=ax.transAxes, fontsize=7,
                   verticalalignment='bottom', horizontalalignment='center',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                           edgecolor='black', linewidth=0.8, alpha=0.95))
        
        ax.set_xlabel('mDFR', fontsize=8)
        ax.set_ylabel('Gram Volume ($m$)', fontsize=8)
        ax.set_title('(a) mDFR vs Gram Vol. (per lang)', loc='left', fontsize=8, fontweight='bold')
        ax.legend(fontsize=5, ncol=2, loc='upper right', framealpha=0.9)
        ax.grid(True, alpha=0.3); ax.tick_params(labelsize=7)
        
        # --- (b) Avg mDFR vs Gram Volume (model-level) ---
        ax = axes[0, 1]
        model_data = df[df['Language'] != 'Average'].groupby('Model').agg({
            'Gram Volume': 'mean', 'Avg mDFR': 'first', 'Accuracy': 'mean',
            'Model Size (B)': 'first'
        }).dropna(subset=['Gram Volume', 'Avg mDFR'])
        
        if 'Model Size (B)' in model_data.columns and not model_data['Model Size (B)'].isna().all():
            scatter = ax.scatter(model_data['Avg mDFR'], model_data['Gram Volume'],
                      s=50, alpha=0.8, c=model_data['Model Size (B)'], cmap='viridis',
                      edgecolors='white', linewidth=0.5)
            cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, pad=0.02)
            cbar.set_label('Size (B)', fontsize=6); cbar.ax.tick_params(labelsize=6)
        else:
            ax.scatter(model_data['Avg mDFR'], model_data['Gram Volume'],
                      s=50, alpha=0.8, color='#2196f3', edgecolors='white', linewidth=0.5)
        
        if len(model_data) > 2:
            r, p_val = pearsonr(model_data['Avg mDFR'], model_data['Gram Volume'])
            sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
            z = np.polyfit(model_data['Avg mDFR'], model_data['Gram Volume'], 1)
            p = np.poly1d(z)
            x_line = np.linspace(model_data['Avg mDFR'].min(), model_data['Avg mDFR'].max(), 100)
            ax.plot(x_line, p(x_line), 'k--', alpha=0.7, linewidth=1.2)
            ax.text(0.50, 0.05, f'$r = {r:.2f}$ ({sig})',
                   transform=ax.transAxes, fontsize=7,
                   verticalalignment='bottom', horizontalalignment='center',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                           edgecolor='black', linewidth=0.8, alpha=0.95))
        
        ax.set_xlabel('Avg mDFR', fontsize=8)
        ax.set_ylabel('Gram Volume ($m$)', fontsize=8)
        ax.set_title('(b) Avg mDFR vs Gram Vol.', loc='left', fontsize=8, fontweight='bold')
        ax.grid(True, alpha=0.3); ax.tick_params(labelsize=7)
        
        # --- (c) mDFR vs Accuracy (per-language) ---
        ax = axes[1, 0]
        for lang in lang_colors:
            ld = plot_data[plot_data['Language'] == lang]
            if len(ld) > 0:
                ax.scatter(ld['mDFR'], ld['Accuracy'], s=35, alpha=0.8,
                          color=lang_colors[lang], edgecolors='white', linewidth=0.4,
                          label=lang)
        
        if len(plot_data) > 2:
            r, p_val = pearsonr(plot_data['mDFR'], plot_data['Accuracy'])
            sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
            z = np.polyfit(plot_data['mDFR'], plot_data['Accuracy'], 1)
            p = np.poly1d(z)
            x_line = np.linspace(plot_data['mDFR'].min(), plot_data['mDFR'].max(), 100)
            ax.plot(x_line, p(x_line), 'k--', alpha=0.7, linewidth=1.2)
            ax.text(0.50, 0.95, f'$r = {r:.2f}$ ({sig})',
                   transform=ax.transAxes, fontsize=7,
                   verticalalignment='top', horizontalalignment='center',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                           edgecolor='black', linewidth=0.8, alpha=0.95))
        
        ax.set_xlabel('mDFR', fontsize=8)
        ax.set_ylabel('Accuracy (%)', fontsize=8)
        ax.set_title('(c) mDFR vs Accuracy', loc='left', fontsize=8, fontweight='bold')
        ax.legend(fontsize=5, ncol=2, loc='upper right', framealpha=0.9)
        ax.grid(True, alpha=0.3); ax.tick_params(labelsize=7)
        
        # --- (d) Gram Volume vs Accuracy, split by DFR tertiles ---
        ax = axes[1, 1]
        if len(plot_data) > 6:
            # Split into low/high DFR groups
            median_dfr = plot_data['mDFR'].median()
            low_dfr = plot_data[plot_data['mDFR'] <= median_dfr]
            high_dfr = plot_data[plot_data['mDFR'] > median_dfr]
            
            ax.scatter(low_dfr['Gram Volume'], low_dfr['Accuracy'], s=35, alpha=0.8,
                      color='#2196f3', edgecolors='white', linewidth=0.4,
                      label=f'Low DFR (≤{median_dfr:.2f})')
            ax.scatter(high_dfr['Gram Volume'], high_dfr['Accuracy'], s=35, alpha=0.8,
                      color='#ff9800', edgecolors='white', linewidth=0.4,
                      label=f'High DFR (>{median_dfr:.2f})')
            
            # Trend lines for each group
            for subset, color in [(low_dfr, '#2196f3'), (high_dfr, '#ff9800')]:
                if len(subset) > 2:
                    z = np.polyfit(subset['Gram Volume'], subset['Accuracy'], 1)
                    p = np.poly1d(z)
                    x_line = np.linspace(subset['Gram Volume'].min(), subset['Gram Volume'].max(), 100)
                    ax.plot(x_line, p(x_line), '--', color=color, alpha=0.7, linewidth=1.2)
            
            # Overall correlation
            r, p_val = pearsonr(plot_data['Gram Volume'], plot_data['Accuracy'])
            sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
            ax.text(0.50, 0.95, f'overall $r = {r:.2f}$ ({sig})',
                   transform=ax.transAxes, fontsize=7,
                   verticalalignment='top', horizontalalignment='center',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                           edgecolor='black', linewidth=0.8, alpha=0.95))
        
        ax.set_xlabel('Gram Volume ($m$)', fontsize=8)
        ax.set_ylabel('Accuracy (%)', fontsize=8)
        ax.set_title('(d) Gram Vol. vs Acc. by DFR', loc='left', fontsize=8, fontweight='bold')
        ax.legend(fontsize=5, loc='lower right', framealpha=0.9)
        ax.grid(True, alpha=0.3); ax.tick_params(labelsize=7)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=600, bbox_inches='tight', facecolor='white')
            plt.savefig(save_path.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
            print(f"Saved tokenization (mDFR) vs Gram volume figure: {save_path}")
        
        plt.close()


def plot_model_size_gram_volume(df: pd.DataFrame, save_path: Path = None):
    """
    Model size vs Gram volume (and accuracy).
    
    Shows whether bigger models trivially have higher Gram volume.
    Even if true, this is not a useful finding since it conflates
    model capacity with geometric disentanglement.
    
    Layout: 1x3
      (a) Model Size vs Gram Volume
      (b) Model Size vs Accuracy
      (c) Gram Volume vs Accuracy (colored by model size)
    """
    style_context = ['science', 'no-latex', 'grid'] if SCIENCEPLOTS_AVAILABLE else []
    with plt.style.context(style_context) if style_context else nullcontext():
        fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.2))
        
        # Use model-level aggregate data
        model_data = df[df['Language'] != 'Average'].groupby('Model').agg({
            'Gram Volume': 'mean',
            'Accuracy': 'mean',
            'Model Size (B)': 'first',
            'Avg Component R²': 'first',
        }).dropna(subset=['Model Size (B)'])
        
        if len(model_data) < 2:
            plt.close()
            return
        
        # Use log scale for model size (spans 0.6B to 20B)
        model_data['Log Size'] = np.log10(model_data['Model Size (B)'])
        
        # (a) Model Size vs Gram Volume
        ax = axes[0]
        valid = model_data[['Model Size (B)', 'Gram Volume']].dropna()
        ax.scatter(valid['Model Size (B)'], valid['Gram Volume'],
                  s=50, alpha=0.8, color='#2196f3', edgecolors='white', linewidth=0.5)
        if len(valid) > 2:
            r, p_val = pearsonr(np.log10(valid['Model Size (B)']), valid['Gram Volume'])
            sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
            z = np.polyfit(np.log10(valid['Model Size (B)']), valid['Gram Volume'], 1)
            p = np.poly1d(z)
            x_line = np.linspace(np.log10(valid['Model Size (B)'].min()), 
                                np.log10(valid['Model Size (B)'].max()), 100)
            ax.plot(10**x_line, p(x_line), 'k--', alpha=0.7, linewidth=1.2)
            ax.text(0.50, 0.95, f'$r = {r:.2f}$ ({sig})',
                   transform=ax.transAxes, fontsize=8,
                   verticalalignment='top', horizontalalignment='center',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                           edgecolor='black', linewidth=0.8, alpha=0.95))
        ax.set_xscale('log')
        ax.set_xlabel('Model Size (B params)', fontsize=9)
        ax.set_ylabel('Gram Volume ($m$)', fontsize=9)
        ax.set_title('(a) Size vs Gram Vol.', loc='left', fontsize=9, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.tick_params(axis='both', which='major', labelsize=8)
        
        # (b) Model Size vs Accuracy
        ax = axes[1]
        valid = model_data[['Model Size (B)', 'Accuracy']].dropna()
        ax.scatter(valid['Model Size (B)'], valid['Accuracy'],
                  s=50, alpha=0.8, color='#4caf50', edgecolors='white', linewidth=0.5)
        if len(valid) > 2:
            r, p_val = pearsonr(np.log10(valid['Model Size (B)']), valid['Accuracy'])
            sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
            z = np.polyfit(np.log10(valid['Model Size (B)']), valid['Accuracy'], 1)
            p = np.poly1d(z)
            x_line = np.linspace(np.log10(valid['Model Size (B)'].min()),
                                np.log10(valid['Model Size (B)'].max()), 100)
            ax.plot(10**x_line, p(x_line), 'k--', alpha=0.7, linewidth=1.2)
            ax.text(0.50, 0.95, f'$r = {r:.2f}$ ({sig})',
                   transform=ax.transAxes, fontsize=8,
                   verticalalignment='top', horizontalalignment='center',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                           edgecolor='black', linewidth=0.8, alpha=0.95))
        ax.set_xscale('log')
        ax.set_xlabel('Model Size (B params)', fontsize=9)
        ax.set_ylabel('Avg Accuracy (%)', fontsize=9)
        ax.set_title('(b) Size vs Accuracy', loc='left', fontsize=9, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.tick_params(axis='both', which='major', labelsize=8)
        
        # (c) Gram Volume vs Accuracy, colored by model size
        #     Exclude reasoning-model outliers for correlation / trend line
        ax = axes[2]
        valid = model_data[['Gram Volume', 'Accuracy', 'Model Size (B)']].dropna()
        excluded_mask = valid.index.isin(_GRAM_ACC_EXCLUDE)
        keep = valid[~excluded_mask]
        excl = valid[excluded_mask]

        # Plot kept points (coloured by size)
        scatter = ax.scatter(keep['Gram Volume'], keep['Accuracy'],
                  s=50, alpha=0.8, c=np.log10(keep['Model Size (B)']),
                  cmap='viridis', edgecolors='white', linewidth=0.5)
        # Plot excluded points (grey, semi-transparent, with "x" marker)
        # if len(excl):
        #     ax.scatter(excl['Gram Volume'], excl['Accuracy'],
        #                s=40, alpha=0.35, color='grey', marker='x', linewidths=1.2,
        #                label=f'excluded ({len(excl)})')
        #     ax.legend(fontsize=6, loc='lower right', framealpha=0.8)
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label('log$_{10}$(Size)', fontsize=7)
        cbar.ax.tick_params(labelsize=6)
        # Correlation & trend line on kept points only
        if len(keep) > 2:
            r, p_val = pearsonr(keep['Gram Volume'], keep['Accuracy'])
            sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
            z = np.polyfit(keep['Gram Volume'], keep['Accuracy'], 1)
            p = np.poly1d(z)
            x_line = np.linspace(keep['Gram Volume'].min(), keep['Gram Volume'].max(), 100)
            ax.plot(x_line, p(x_line), 'k--', alpha=0.7, linewidth=1.2)
            ax.text(0.50, 0.95, f'$r = {r:.2f}$ ({sig}), $n={len(keep)}$',
                   transform=ax.transAxes, fontsize=8,
                   verticalalignment='top', horizontalalignment='center',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                           edgecolor='black', linewidth=0.8, alpha=0.95))
        ax.set_xlabel('Gram Volume ($m$)', fontsize=9)
        ax.set_ylabel('Avg Accuracy (%)', fontsize=9)
        ax.set_title('(c) Gram Vol. vs Acc.', loc='left', fontsize=9, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.tick_params(axis='both', which='major', labelsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=600, bbox_inches='tight', facecolor='white')
            plt.savefig(save_path.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
            print(f"Saved model size vs Gram volume figure: {save_path}")
        
        plt.close()


def plot_combined_questions(df: pd.DataFrame, corr_df: pd.DataFrame, save_path: Path = None):
    """
    Combined figure answering both research questions:
    Q1: Do LLMs build LINEAR time representations? (Linearity R²)
    Q2: Do LLMs split dates into Y/M/D? (Disentanglement D)
    """
    style_context = ['science', 'no-latex', 'grid'] if SCIENCEPLOTS_AVAILABLE else []
    with plt.style.context(style_context) if style_context else nullcontext():
        fig = plt.figure(figsize=(7.5, 6))
        
        # Create grid spec for custom layout
        gs = fig.add_gridspec(3, 4, hspace=0.4, wspace=0.35)
        
        # Title
        fig.suptitle('Temporal Geometry in LLM Representations', fontsize=12, fontweight='bold', y=0.98)
    
        # =====================
        # TOP ROW: Summary boxes for both questions
        # =====================
        
        # Q1 Summary (left)
        ax_q1_summary = fig.add_subplot(gs[0, 0:2])
        ax_q1_summary.set_xlim(0, 10)
        ax_q1_summary.set_ylim(0, 10)
        ax_q1_summary.axis('off')
        
        # Q1 content
        ax_q1_summary.add_patch(plt.Rectangle((0.1, 0.3), 9.8, 9.4, facecolor='#E8F5E9', edgecolor='#2E7D32', linewidth=1.5))
        ax_q1_summary.text(5, 8.5, 'Q1: Linear Time Representations?', 
                           ha='center', va='center', fontsize=10, fontweight='bold', color='#2E7D32')
        ax_q1_summary.text(5, 6.5, 'Metric: Temporal Linearity ($R^2$)', ha='center', va='center', fontsize=9)
        ax_q1_summary.text(5, 5, 'High $R^2$ → Years form a line', 
                           ha='center', va='center', fontsize=8, style='italic')
        
        # Calculate summary stats for Q1
        avg_r2 = df[df['Language'] != 'Average']['R²'].mean()
        r2_corr_with_acc = corr_df[(corr_df['Metric'] == 'R²') & (corr_df['Language'] != 'Average')]['Pearson r'].mean()
        
        ax_q1_summary.text(5, 3, f'Avg $R^2$: {avg_r2:.3f}', ha='center', va='center', fontsize=9)
        ax_q1_summary.text(5, 1.5, f'Corr. with Accuracy: $r$ = {r2_corr_with_acc:.2f}', 
                           ha='center', va='center', fontsize=9, fontweight='bold',
                           color='#2E7D32' if r2_corr_with_acc > 0.3 else '#FF8F00')
    
        # Q2 Summary (right)
        ax_q2_summary = fig.add_subplot(gs[0, 2:4])
        ax_q2_summary.set_xlim(0, 10)
        ax_q2_summary.set_ylim(0, 10)
        ax_q2_summary.axis('off')
        
        # Q2 content
        ax_q2_summary.add_patch(plt.Rectangle((0.1, 0.3), 9.8, 9.4, facecolor='#E3F2FD', edgecolor='#1565C0', linewidth=1.5))
        ax_q2_summary.text(5, 8.5, 'Q2: Y/M/D Separation?', 
                           ha='center', va='center', fontsize=10, fontweight='bold', color='#1565C0')
        ax_q2_summary.text(5, 6.5, 'Metric: Gram Volume ($m$)', ha='center', va='center', fontsize=9)
        ax_q2_summary.text(5, 5, 'm=1 → Unit cube (orthogonal)', 
                           ha='center', va='center', fontsize=8, style='italic')
        
        # Calculate summary stats for Q2
        avg_gram = df[df['Language'] != 'Average']['Gram Volume'].mean()
        gram_corr_with_acc = corr_df[(corr_df['Metric'] == 'Gram Volume') & (corr_df['Language'] != 'Average')]['Pearson r'].mean()
        
        ax_q2_summary.text(5, 3, f'Avg $m$: {avg_gram:.3f}', ha='center', va='center', fontsize=9)
        ax_q2_summary.text(5, 1.5, f'Corr. with Accuracy: $r$ = {gram_corr_with_acc:.2f}', 
                           ha='center', va='center', fontsize=9, fontweight='bold',
                           color='#2E7D32' if abs(disent_corr_with_acc) > 0.3 else '#FF8F00')
    
        # =====================
        # MIDDLE ROW: Model rankings for both metrics
        # =====================
        
        # Q1: Model R² ranking
        ax_r2_rank = fig.add_subplot(gs[1, 0:2])
        model_r2 = df[df['Language'] != 'Average'].groupby('Model')['R²'].mean().sort_values(ascending=True)
        colors_r2 = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(model_r2)))
        
        ax_r2_rank.barh(range(len(model_r2)), model_r2.values, color=colors_r2, edgecolor='k', linewidth=0.3)
        ax_r2_rank.set_yticks(range(len(model_r2)))
        ax_r2_rank.set_yticklabels([m.split('/')[-1][:18] for m in model_r2.index], fontsize=7)
        ax_r2_rank.set_xlabel('Temporal Linearity ($R^2$)')
        ax_r2_rank.set_title('Q1: Model Linearity', fontweight='bold', loc='left', fontsize=10, color='#2E7D32')
        ax_r2_rank.axvline(x=0.5, color='C3', linestyle='--', alpha=0.8, linewidth=1)
        ax_r2_rank.set_xlim(0, 1)
        
        # Q2: Model Disentanglement ranking
        ax_disent_rank = fig.add_subplot(gs[1, 2:4])
        model_disent = df[df['Language'] != 'Average'].groupby('Model')['Disentanglement'].mean().sort_values(ascending=True)
        colors_disent = plt.cm.Blues(np.linspace(0.3, 0.9, len(model_disent)))
        
        ax_disent_rank.barh(range(len(model_disent)), model_disent.values, color=colors_disent, edgecolor='k', linewidth=0.3)
        ax_disent_rank.set_yticks(range(len(model_disent)))
        ax_disent_rank.set_yticklabels([m.split('/')[-1][:18] for m in model_disent.index], fontsize=7)
        ax_disent_rank.set_xlabel('Disentanglement ($D$)')
        ax_disent_rank.set_title('Q2: Y/M/D Separation', fontweight='bold', loc='left', fontsize=10, color='#1565C0')
    
        # =====================
        # BOTTOM ROW: Correlation scatter plots
        # =====================
        
        # Q1: R² vs Accuracy (aggregated across languages)
        ax_r2_acc = fig.add_subplot(gs[2, 0:2])
        valid_r2 = df[df['Language'] != 'Average'][['R²', 'Accuracy', 'Language']].dropna()
        
        # Color by language - use colorblind-friendly colors
        lang_colors = {
            'Arabic': '#E91E63'.lower(),
            'Chinese': '#9C27B0'.lower(),
            'English': '#2196F3'.lower(),
            'German': '#4CAF50'.lower(),
            'Hausa': '#FF9800'.lower()
        }
        
        for lang in valid_r2['Language'].unique():
            lang_data = valid_r2[valid_r2['Language'] == lang]
            ax_r2_acc.scatter(lang_data['R²'], lang_data['Accuracy'], 
                             c=lang_colors.get(lang, 'gray'), s=35, alpha=0.8, 
                             edgecolors='k', linewidth=0.3, label=lang)
        
        # Overall trend line
        if len(valid_r2) > 2:
            z = np.polyfit(valid_r2['R²'], valid_r2['Accuracy'], 1)
            p = np.poly1d(z)
            x_line = np.linspace(valid_r2['R²'].min(), valid_r2['R²'].max(), 100)
            ax_r2_acc.plot(x_line, p(x_line), 'k--', alpha=0.7, linewidth=1.2)
            
            r, p_val = pearsonr(valid_r2['R²'], valid_r2['Accuracy'])
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
            ax_r2_acc.text(0.95, 0.05, f'$r$ = {r:.2f} ({sig})', 
                          transform=ax_r2_acc.transAxes, fontsize=8, ha='right', va='bottom',
                          bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='none'))
        
        ax_r2_acc.set_xlabel('Temporal Linearity ($R^2$)')
        ax_r2_acc.set_ylabel('Accuracy (%)')
        ax_r2_acc.set_title('Q1: Linearity vs Accuracy', fontweight='bold', loc='left', fontsize=10, color='#2E7D32')
        ax_r2_acc.legend(loc='upper left', fontsize=6, framealpha=0.9, ncol=2)
        
        # Q2: Gram Volume vs Accuracy
        ax_gram_acc = fig.add_subplot(gs[2, 2:4])
        valid_gram = df[df['Language'] != 'Average'][['Gram Volume', 'Accuracy', 'Language']].dropna()
        
        for lang in valid_gram['Language'].unique():
            lang_data = valid_gram[valid_gram['Language'] == lang]
            ax_gram_acc.scatter(lang_data['Gram Volume'], lang_data['Accuracy'], 
                                 c=lang_colors.get(lang, 'gray'), s=35, alpha=0.8, 
                                 edgecolors='k', linewidth=0.3, label=lang)
        
        # Overall trend line
        if len(valid_gram) > 2:
            z = np.polyfit(valid_gram['Gram Volume'], valid_gram['Accuracy'], 1)
            p = np.poly1d(z)
            x_line = np.linspace(valid_gram['Gram Volume'].min(), valid_gram['Gram Volume'].max(), 100)
            ax_gram_acc.plot(x_line, p(x_line), 'k--', alpha=0.7, linewidth=1.2)
            
            r, p_val = pearsonr(valid_gram['Gram Volume'], valid_gram['Accuracy'])
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
            ax_gram_acc.text(0.95, 0.05, f'$r$ = {r:.2f} ({sig})', 
                              transform=ax_gram_acc.transAxes, fontsize=8, ha='right', va='bottom',
                              bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='none'))
        
        ax_gram_acc.set_xlabel('Gram Volume ($m$)')
        ax_gram_acc.set_ylabel('Accuracy (%)')
        ax_gram_acc.set_title('Q2: Gram Volume vs Accuracy', fontweight='bold', loc='left', fontsize=10, color='#1565C0')
        ax_gram_acc.legend(loc='upper left', fontsize=6, framealpha=0.9, ncol=2)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.savefig(save_path.with_suffix('.pdf'), dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Saved combined research questions figure: {save_path}")
        
        plt.close()


def plot_main_figure(df: pd.DataFrame, corr_df: pd.DataFrame, save_path: Path = None):
    """Create main figure with R² correlations for each language."""
    style_context = ['science', 'no-latex'] if SCIENCEPLOTS_AVAILABLE else []
    with plt.style.context(style_context) if style_context else nullcontext():
        # Get languages to plot
        languages_to_plot = ['Arabic', 'English', 'German', 'Hausa', 'Chinese']
        languages_to_plot = [lang for lang in languages_to_plot if lang in df['Language'].unique()]
        
        # Create figure with subplots in a single row
        fig, axes = plt.subplots(1, len(languages_to_plot), figsize=(3*len(languages_to_plot), 2.5))
        if len(languages_to_plot) == 1:
            axes = [axes]
        
        for idx, lang in enumerate(languages_to_plot):
            ax = axes[idx]
            lang_data = df[df['Language'] == lang]
            
            if 'R²' in lang_data.columns and len(lang_data) > 0:
                valid_data = lang_data[['R²', 'Accuracy', 'Model']].dropna()
                
                # Scatter plot
                ax.scatter(valid_data['R²'], valid_data['Accuracy'], 
                          s=50, alpha=0.7, color='#2ca02c', edgecolors='none')
                
                if len(valid_data) > 2:
                    # Fit trend line
                    z = np.polyfit(valid_data['R²'], valid_data['Accuracy'], 1)
                    p = np.poly1d(z)
                    x_line = np.linspace(valid_data['R²'].min(), valid_data['R²'].max(), 100)
                    ax.plot(x_line, p(x_line), 'k--', alpha=0.8, linewidth=1.5)
                    
                    # Calculate correlation
                    r, p_val = pearsonr(valid_data['R²'], valid_data['Accuracy'])
                    
                    # Add correlation box
                    ax.text(0.05, 0.95, f'$r = {r:.2f}$', 
                           transform=ax.transAxes, fontsize=9, verticalalignment='top',
                           bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                                   edgecolor='black', linewidth=1, alpha=0.9))
                
                # Styling
                ax.set_xlabel('Temporal Linearity ($R^2$)', fontsize=9)
                if idx == 0:
                    ax.set_ylabel('Accuracy (%)', fontsize=9)
                ax.set_title(lang, fontweight='bold', fontsize=10)
                ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
                ax.tick_params(axis='both', which='major', labelsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.savefig(save_path.with_suffix('.pdf'), dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Saved main figure: {save_path}")
        
        plt.close()


# ==========================================
# 4. MAIN EXECUTION
# ==========================================

def print_research_findings(df: pd.DataFrame, corr_df: pd.DataFrame):
    """Print summary of findings for both research questions."""
    print("\n" + "="*70)
    print("RESEARCH QUESTION FINDINGS")
    print("="*70)
    
    # Q1: Linearity
    print("\n" + "-"*70)
    print("Q1: Do LLMs Build LINEAR Time Representations (like humans)?")
    print("    Metric: Temporal Linearity (R²)")
    print("    High R² → years form a line in embedding space")
    print("-"*70)
    
    r2_data = df[df['Language'] != 'Average']['R²'].dropna()
    r2_corr = corr_df[(corr_df['Metric'] == 'R²') & (corr_df['Language'] != 'Average')]
    
    print(f"\n  Average R² across all models: {r2_data.mean():.3f} (±{r2_data.std():.3f})")
    print(f"  Range: [{r2_data.min():.3f}, {r2_data.max():.3f}]")
    print(f"  Models with high linearity (R² > 0.5): {(r2_data > 0.5).sum()}/{len(r2_data)}")
    
    if len(r2_corr) > 0:
        avg_r2_acc_corr = r2_corr['Pearson r'].mean()
        sig_langs = r2_corr[r2_corr['p-value'] < 0.05]['Language'].tolist()
        print(f"\n  R² → Accuracy correlation (avg across languages): r = {avg_r2_acc_corr:.3f}")
        print(f"  Languages with significant correlation: {', '.join(sig_langs) if sig_langs else 'None'}")
        
        if avg_r2_acc_corr > 0.3:
            print("\n  📊 FINDING: Strong positive relationship between linearity and accuracy.")
            print("     → Models with more linear time representations perform better.")
        elif avg_r2_acc_corr > 0:
            print("\n  📊 FINDING: Weak positive relationship between linearity and accuracy.")
        else:
            print("\n  📊 FINDING: No clear relationship between linearity and accuracy.")
    
    # Q2: Gram Volume (Parallelepiped Volume)
    print("\n" + "-"*70)
    print("Q2: Do LLMs Split Dates into Year/Month/Day?")
    print("    Metric: Parallelepiped Volume (Gram determinant m)")
    print("    m = sqrt(det(G)) where G = X^T X, X = [v_year, v_month, v_day]")
    print("    m = 1 → Unit cube (orthogonal Y/M/D)")
    print("    m = 0 → Collapsed (linearly dependent)")
    print("-"*70)
    
    gram_data = df[df['Language'] != 'Average']['Gram Volume'].dropna()
    gram_corr = corr_df[(corr_df['Metric'] == 'Gram Volume') & (corr_df['Language'] != 'Average')]
    
    print(f"\n  Average m across all models: {gram_data.mean():.3f} (±{gram_data.std():.3f})")
    print(f"  Range: [{gram_data.min():.3f}, {gram_data.max():.3f}]")
    print(f"  Models with high orthogonality (m > 0.7): {(gram_data > 0.7).sum()}/{len(gram_data)}")
    
    if len(gram_corr) > 0:
        avg_gram_acc_corr = gram_corr['Pearson r'].mean()
        sig_langs = gram_corr[gram_corr['p-value'] < 0.05]['Language'].tolist()
        print(f"\n  m → Accuracy correlation (avg across languages): r = {avg_gram_acc_corr:.3f}")
        print(f"  Languages with significant correlation: {', '.join(sig_langs) if sig_langs else 'None'}")
        
        if avg_gram_acc_corr > 0.3:
            print("\n  📊 FINDING: Orthogonal representations (high m) improve accuracy.")
            print("     → Models benefit from separating Y/M/D as independent concepts.")
        elif avg_gram_acc_corr < -0.3:
            print("\n  📊 FINDING: Entangled representations (low m) perform better.")
            print("     → Models may benefit from holistic date encoding.")
        else:
            print("\n  📊 FINDING: Geometric disentanglement has limited impact on accuracy.")
            print("     → Y/M/D separation may not be crucial for temporal reasoning.")
    
    print("\n" + "="*70)


def main():
    """Main execution function."""
    output_dir = Path("results/correlation_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("TEMPORAL GEOMETRY ANALYSIS: TWO RESEARCH QUESTIONS")
    print("="*70)
    print("\nQ1: Do LLMs build LINEAR time representations? (Linearity R²)")
    print("Q2: Do LLMs split dates into Y/M/D? (Disentanglement D)")
    print("="*70)
    
    # Load data
    print("\n[1/6] Loading data...")
    accuracy_df = load_accuracy_data()
    geometry_results = load_geometry_results()
    print(f"      ✓ {len(accuracy_df)} models with accuracy data")
    print(f"      ✓ {len(geometry_results)} models with geometry data")
    
    # Merge
    print("\n[2/6] Merging datasets...")
    merged_df = merge_data(accuracy_df, geometry_results, use_synthetic_r2=False)
    merged_df.to_csv(output_dir / "merged_data_by_language.csv", index=False)
    print(f"      ✓ Merged {len(merged_df)} data points")
    print(f"      ✓ Languages: {', '.join(merged_df['Language'].unique())}")
    print(f"      ✓ Saved: merged_data_by_language.csv")
    
    # Correlations
    print("\n[3/6] Computing correlations by language...")
    corr_df = compute_correlations(merged_df)
    print(f"      ✓ Computed {len(corr_df)} correlations")
    
    # Print research findings
    print_research_findings(merged_df, corr_df)
    
    # Results summary by language
    print("\n" + "="*70)
    print("DETAILED RESULTS BY LANGUAGE")
    print("="*70)
    
    for lang in corr_df['Language'].unique():
        print(f"\n{lang}:")
        print("-" * 70)
        lang_data = corr_df[corr_df['Language'] == lang]
        for _, row in lang_data.iterrows():
            sig = "***" if row['p-value'] < 0.001 else "**" if row['p-value'] < 0.01 else "*" if row['p-value'] < 0.05 else "ns"
            print(f"  {row['Metric']:25s}: r = {row['Pearson r']:6.3f} ({sig:3s})  [p = {row['p-value']:.4f}]")
    print("-" * 70)
    
    # Save table
    print("\n[4/6] Creating publication table...")
    create_correlation_table(corr_df, output_dir / "table_correlations_by_language.tex")
    print(f"      ✓ Saved: table_correlations_by_language.tex")
    print(f"      ✓ Saved: table_correlations_by_language.csv")
    
    # Figures for Research Questions
    print("\n[5/8] Generating research question figures...")
    
    # Q1: Linearity figure
    plot_q1_linearity(merged_df, corr_df, output_dir / "figure_Q1_linearity.png")
    print(f"      ✓ Saved: figure_Q1_linearity.png (Q1: Linear representations)")
    
    # Q2: Disentanglement figure  
    plot_q2_disentanglement(merged_df, corr_df, output_dir / "figure_Q2_disentanglement.png")
    print(f"      ✓ Saved: figure_Q2_disentanglement.png (Q2: Y/M/D separation)")
    
    # Day/Month/Weekday linearity from full dates (model-level average)
    print("\n[6/9] Generating day/month linearity figure (avg)...")
    plot_day_month_linearity(merged_df, output_dir / "figure_day_month_linearity.png")
    print(f"      ✓ Saved: figure_day_month_linearity.png (Component R² vs Avg Accuracy)")

    # Day/Month linearity broken down by language
    print("\n[7/9] Generating day/month linearity by language figure...")
    plot_day_month_linearity_by_language(merged_df, output_dir / "figure_day_month_linearity_by_lang.png")
    print(f"      ✓ Saved: figure_day_month_linearity_by_lang.png (Day/Month R² per language)")

    # Tokenization vs Gram volume multi-correlation
    print("\n[8/9] Generating tokenization vs Gram volume figure...")
    plot_tokenization_gram_volume(merged_df, output_dir / "figure_tokenization_gram_volume.png")
    print(f"      ✓ Saved: figure_tokenization_gram_volume.png (Tokenization confound)")
    
    # Model size vs Gram volume
    print("\n[9/9] Generating model size vs Gram volume figure...")
    plot_model_size_gram_volume(merged_df, output_dir / "figure_model_size_gram_volume.png")
    print(f"      ✓ Saved: figure_model_size_gram_volume.png (Size confound)")
    
    # # Combined figure for both questions
    # plot_combined_questions(merged_df, corr_df, output_dir / "figure_research_questions_combined.png")
    # print(f"      ✓ Saved: figure_research_questions_combined.png (Both Q1 & Q2)")
    
    # # Additional figures
    # plot_main_figure(merged_df, corr_df, output_dir / "figure_main_by_language.png")
    # print(f"      ✓ Saved: figure_main_by_language.png")
    
    # plot_correlation_heatmap(corr_df, output_dir / "figure_correlation_heatmap.png")
    # print(f"      ✓ Saved: figure_correlation_heatmap.png")
    
    # plot_language_comparison(merged_df, corr_df, output_dir / "figure_language_comparison.png")
    # print(f"      ✓ Saved: figure_language_comparison.png")
    
    print("\n" + "="*70)
    print("✅ ANALYSIS COMPLETE")
    print("="*70)
    print(f"\n📁 Output directory: {output_dir}/")
    print("\n📄 KEY FIGURES FOR PAPER:")
    print("   ┌─────────────────────────────────────────────────────────────────┐")
    print("   │ RESEARCH QUESTION FIGURES:                                      │")
    print("   │  • figure_Q1_linearity.pdf           (Q1: Linear time repr?)    │")
    print("   │  • figure_Q2_disentanglement.pdf     (Q2: Y/M/D separation?)    │")
    print("   ├─────────────────────────────────────────────────────────────────┤")
    print("   │ NEW ANALYSIS FIGURES:                                           │")
    print("   │  • figure_day_month_linearity.pdf    (Day/Month R² avg)         │")
    print("   │  • figure_day_month_linearity_by_lang.pdf (Day/Month per lang)  │")
    print("   │  • figure_tokenization_gram_volume.pdf (Tokenization confound)  │")
    print("   │  • figure_model_size_gram_volume.pdf (Model size confound)      │")
    print("   ├─────────────────────────────────────────────────────────────────┤")
    print("   │ DATA & TABLES:                                                  │")
    print("   │  • table_correlations_by_language.tex (LaTeX table)             │")
    print("   │  • merged_data_by_language.csv       (Raw data)                 │")
    print("   └─────────────────────────────────────────────────────────────────┘")
    print("\n" + "="*70)


if __name__ == "__main__":
    main()