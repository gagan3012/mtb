"""
Temporal Geometry Analysis for LLM Date Representations

This module implements the temporal geometry methodology to analyze:
1. Do LLMs split dates into day/month/year and understand them separately?
2. Do LLMs build an internal linear representation of time (like humans)?

Based on the paper's methodology for analyzing path directions, linear structure,
and calendar component disentanglement in transformer hidden states.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import cross_val_score
from typing import List, Dict, Tuple, Optional
import json
import os
from tqdm import tqdm
import seaborn as sns
from datetime import datetime, timedelta
import random
from itertools import combinations
import pandas as pd

# ==========================================
# 1. CONFIGURATION & SETUP
# ==========================================
MODEL_NAME = "Qwen/Qwen3-0.6B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Define the years to analyze [1990, 2024]
YEARS = list(range(1990, 1995))

# K = 5 date samples per year for robust year-invariant representations
K_SAMPLES_PER_YEAR = 1

# Primary language for analysis (can use single language since cross-lingual alignment not needed)
LANGUAGES = ["en", "de", "zh", "ar", "ha"]

# Templates for full date embedding
DATE_TEMPLATES = {
    "en": "The date is {}.",
    "de": "Das Datum ist {}.",
    "zh": "日期是{}。",
    "ar": "التاريخ هو {}.",
    "ha": "Ranar ita ce {}.",
}

# Templates for year-only analysis
YEAR_TEMPLATES = {
    "en": "The year is {}.",
    "de": "Das Jahr ist {}.",
    "zh": "现在是{}年。",
    "ar": "السنة هي {}.",
    "ha": "Shekarar ita ce {}.",
}

# ==========================================
# DATE ARITHMETIC EVALUATION TEMPLATES
# ==========================================
# Different types of date arithmetic tasks to evaluate model understanding

DATE_ARITHMETIC_TEMPLATES = {
    # Addition: "What date is N days after DATE?"
    "add_days": "What date is {delta} days after {date}? Answer with just the date.",
    "add_weeks": "What date is {delta} weeks after {date}? Answer with just the date.",
    "add_months": "What date is {delta} months after {date}? Answer with just the date.",

    # Subtraction: "What date is N days before DATE?"
    "sub_days": "What date is {delta} days before {date}? Answer with just the date.",
    "sub_weeks": "What date is {delta} weeks before {date}? Answer with just the date.",

    # Difference: "How many days between DATE1 and DATE2?"
    "diff_days": "How many days are between {date1} and {date2}? Answer with just a number.",

    # Day of week: "What day of the week is DATE?"
    "weekday": "What day of the week is {date}? Answer with just the day name.",

    # Month days: "How many days are in MONTH YEAR?"
    "days_in_month": "How many days are in {month} {year}? Answer with just a number.",

    # Leap year: "Is YEAR a leap year?"
    "leap_year": "Is {year} a leap year? Answer yes or no.",
}


# ==========================================
# 2. DATE FORMATTING AND DATA GENERATION
# ==========================================
def format_date_multilingual(date: datetime, language: str) -> str:
    """Format a date in the specified language with natural conventions."""
    year = date.year
    month = date.month
    day = date.day
    weekday = date.weekday()

    months = {
        "en": ["January", "February", "March", "April", "May", "June",
               "July", "August", "September", "October", "November", "December"],
        "de": ["Januar", "Februar", "März", "April", "Mai", "Juni",
               "Juli", "August", "September", "Oktober", "November", "Dezember"],
        "zh": ["一月", "二月", "三月", "四月", "五月", "六月",
               "七月", "八月", "九月", "十月", "十一月", "十二月"],
        "ar": ["يناير", "فبراير", "مارس", "أبريل", "مايو", "يونيو",
               "يوليو", "أغسطس", "سبتمبر", "أكتوبر", "نوفمبر", "ديسمبر"],
        "ha": ["Janairu", "Faburairu", "Maris", "Afirilu", "Mayu", "Yuni",
               "Yuli", "Agusta", "Satumba", "Oktoba", "Nuwamba", "Disamba"],
    }

    weekdays = {
        "en": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
        "de": ["Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag", "Samstag", "Sonntag"],
        "zh": ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"],
        "ar": ["الاثنين", "الثلاثاء", "الأربعاء", "الخميس", "الجمعة", "السبت", "الأحد"],
        "ha": ["Litinin", "Talata", "Laraba", "Alhamis", "Jumma'a", "Asabar", "Lahadi"],
    }

    month_name = months.get(language, months["en"])[month - 1]
    weekday_name = weekdays.get(language, weekdays["en"])[weekday]

    if language == "en":
        return f"{weekday_name}, {month_name} {day}, {year}"
    elif language == "de":
        return f"{weekday_name}, {day}. {month_name} {year}"
    elif language == "zh":
        return f"{year}年{month}月{day}日 {weekday_name}"
    elif language == "ar":
        return f"{weekday_name}، {day} {month_name} {year}"
    elif language == "ha":
        return f"{weekday_name}, {day} {month_name} {year}"
    else:
        return f"{year}-{month:02d}-{day:02d}"


def generate_date_samples(
    years: List[int],
    languages: List[str],
    k_samples: int = 5,
    seed: int = 42
) -> Dict[str, Dict[str, any]]:
    """
    Generate K distinct full dates for each year to create year-invariant representations.

    For each year y ∈ [1990, 2024], we sample K=5 distinct dates (e.g., "1995-03-12", "1995-11-05").

    Returns:
        Dict[language -> Dict with keys: 'years', 'months', 'days', 'weekdays',
             'date_strings', 'datetime_objects', 'year_indices']]
    """
    random.seed(seed)
    np.random.seed(seed)

    data = {lang: {
        'years': [],
        'months': [],
        'days': [],
        'weekdays': [],
        'date_strings': [],
        'datetime_objects': [],
        'year_indices': [],  # Maps each sample to its year index for averaging
    } for lang in languages}

    for year_idx, year in enumerate(years):
        # Generate K diverse dates spread throughout the year
        sample_dates = []
        for k in range(k_samples):
            # Distribute dates evenly across months
            target_month = int((k / k_samples) * 12) + 1
            target_month = min(target_month, 12)

            # Random day within month (handle month lengths)
            if target_month in [1, 3, 5, 7, 8, 10, 12]:
                max_day = 31
            elif target_month in [4, 6, 9, 11]:
                max_day = 30
            else:  # February
                max_day = 29 if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)) else 28

            day = random.randint(1, max_day)

            try:
                date = datetime(year, target_month, day)
            except ValueError:
                date = datetime(year, target_month, 1)

            sample_dates.append(date)

        # Store for each language
        for date in sample_dates:
            for lang in languages:
                data[lang]['years'].append(date.year)
                data[lang]['months'].append(date.month)
                data[lang]['days'].append(date.day)
                data[lang]['weekdays'].append(date.weekday())
                data[lang]['date_strings'].append(format_date_multilingual(date, lang))
                data[lang]['datetime_objects'].append(date)
                data[lang]['year_indices'].append(year_idx)

    # Convert to numpy arrays
    for lang in languages:
        for key in ['years', 'months', 'days', 'weekdays', 'year_indices']:
            data[lang][key] = np.array(data[lang][key])

    return data


# ==========================================
# DATE ARITHMETIC TASK GENERATION
# ==========================================

def generate_date_arithmetic_tasks(n_tasks: int = 100, seed: int = 42) -> List[Dict]:
    """
    Generate date arithmetic evaluation tasks with ground truth answers.

    Task types:
    - add_days: Add N days to a date
    - sub_days: Subtract N days from a date
    - diff_days: Difference between two dates
    - weekday: What day of the week is a date
    - days_in_month: How many days in a month
    - leap_year: Is a year a leap year

    Returns:
        List of task dicts with 'type', 'prompt', 'answer', 'date_info'
    """
    random.seed(seed)
    np.random.seed(seed)

    tasks = []
    months = ["January", "February", "March", "April", "May", "June",
              "July", "August", "September", "October", "November", "December"]
    weekday_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    for _ in range(n_tasks):
        task_type = random.choice(['add_days', 'sub_days', 'diff_days', 'weekday',
                                   'days_in_month', 'leap_year'])

        # Generate random base date
        year = random.randint(1990, 2024)
        month = random.randint(1, 12)
        max_day = 28 if month == 2 else (30 if month in [4, 6, 9, 11] else 31)
        day = random.randint(1, max_day)

        try:
            base_date = datetime(year, month, day)
        except ValueError:
            base_date = datetime(year, month, 1)

        date_str = f"{months[base_date.month-1]} {base_date.day}, {base_date.year}"

        if task_type == 'add_days':
            delta = random.randint(1, 30)
            result_date = base_date + timedelta(days=delta)
            prompt = f"What date is {delta} days after {date_str}? Answer with just the date."
            answer = f"{months[result_date.month-1]} {result_date.day}, {result_date.year}"
            task = {
                'type': task_type,
                'prompt': prompt,
                'answer': answer,
                'date_info': {
                    'base_year': year, 'base_month': month, 'base_day': day,
                    'delta': delta, 'operation': 'add'
                }
            }

        elif task_type == 'sub_days':
            delta = random.randint(1, 30)
            result_date = base_date - timedelta(days=delta)
            prompt = f"What date is {delta} days before {date_str}? Answer with just the date."
            answer = f"{months[result_date.month-1]} {result_date.day}, {result_date.year}"
            task = {
                'type': task_type,
                'prompt': prompt,
                'answer': answer,
                'date_info': {
                    'base_year': year, 'base_month': month, 'base_day': day,
                    'delta': delta, 'operation': 'subtract'
                }
            }

        elif task_type == 'diff_days':
            delta = random.randint(1, 60)
            date2 = base_date + timedelta(days=delta)
            date_str2 = f"{months[date2.month-1]} {date2.day}, {date2.year}"
            prompt = f"How many days are between {date_str} and {date_str2}? Answer with just a number."
            answer = str(delta)
            task = {
                'type': task_type,
                'prompt': prompt,
                'answer': answer,
                'date_info': {
                    'base_year': year, 'base_month': month, 'base_day': day,
                    'delta': delta, 'operation': 'difference'
                }
            }

        elif task_type == 'weekday':
            prompt = f"What day of the week is {date_str}? Answer with just the day name."
            answer = weekday_names[base_date.weekday()]
            task = {
                'type': task_type,
                'prompt': prompt,
                'answer': answer,
                'date_info': {
                    'base_year': year, 'base_month': month, 'base_day': day,
                    'weekday': base_date.weekday()
                }
            }

        elif task_type == 'days_in_month':
            m = random.randint(1, 12)
            y = random.randint(1990, 2024)
            if m == 2:
                days = 29 if (y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)) else 28
            elif m in [4, 6, 9, 11]:
                days = 30
            else:
                days = 31
            prompt = f"How many days are in {months[m-1]} {y}? Answer with just a number."
            answer = str(days)
            task = {
                'type': task_type,
                'prompt': prompt,
                'answer': answer,
                'date_info': {'month': m, 'year': y, 'days': days}
            }

        elif task_type == 'leap_year':
            y = random.randint(1990, 2024)
            is_leap = (y % 4 == 0 and (y % 100 != 0 or y % 400 == 0))
            prompt = f"Is {y} a leap year? Answer yes or no."
            answer = "yes" if is_leap else "no"
            task = {
                'type': task_type,
                'prompt': prompt,
                'answer': answer,
                'date_info': {'year': y, 'is_leap': is_leap}
            }

        tasks.append(task)

    return tasks


def evaluate_date_arithmetic(
    model,
    tokenizer,
    tasks: List[Dict],
    max_new_tokens: int = 50
) -> Dict[str, any]:
    """
    Evaluate model on date arithmetic tasks.

    Returns:
        Dict with accuracy by task type and overall metrics
    """
    results = {task_type: {'correct': 0, 'total': 0, 'predictions': []}
               for task_type in ['add_days', 'sub_days', 'diff_days', 'weekday',
                                'days_in_month', 'leap_year']}

    for task in tqdm(tasks, desc="Evaluating date arithmetic"):
        prompt = task['prompt']
        expected = task['answer'].lower().strip()
        task_type = task['type']

        # Generate response
        messages = [
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        response = tokenizer.decode(outputs[0][model_inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        response_clean = response.lower().strip().split('\n')[0]  # Take first line

        # Check if correct (flexible matching)
        correct = False
        if task_type in ['add_days', 'sub_days']:
            # Date matching - check if answer contains key parts
            correct = all(part.lower() in response_clean for part in expected.split()[:2])
        elif task_type == 'diff_days':
            # Number matching
            correct = expected in response_clean
        elif task_type == 'weekday':
            correct = expected in response_clean
        elif task_type == 'days_in_month':
            correct = expected in response_clean
        elif task_type == 'leap_year':
            correct = expected in response_clean

        results[task_type]['correct'] += int(correct)
        results[task_type]['total'] += 1
        results[task_type]['predictions'].append({
            'prompt': prompt,
            'expected': expected,
            'response': response_clean,
            'correct': correct,
            'date_info': task['date_info']
        })

    # Compute accuracy per task type
    for task_type in results:
        total = results[task_type]['total']
        if total > 0:
            results[task_type]['accuracy'] = results[task_type]['correct'] / total
        else:
            results[task_type]['accuracy'] = 0.0

    # Overall accuracy
    total_correct = sum(r['correct'] for r in results.values())
    total_tasks = sum(r['total'] for r in results.values())
    results['overall_accuracy'] = total_correct / total_tasks if total_tasks > 0 else 0.0

    return results


# ==========================================
# 3. MODEL LOADING AND EMBEDDING EXTRACTION
# ==========================================
def load_model(model_name: str):
    """Load model and tokenizer with hidden state output enabled."""
    print(f"Loading {model_name} on {DEVICE}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        output_hidden_states=True,
        device_map="auto",
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
    )
    model.eval()
    return model, tokenizer

def find_year_token_indices(tokenizer, text: str, year: int) -> list:
    """Find indices of tokens corresponding to the year."""
    year_str = str(year)
    tokens = tokenizer(text, return_tensors="pt")
    input_ids = tokens["input_ids"][0]
    
    year_indices = []
    for idx, tid in enumerate(input_ids):
        decoded = tokenizer.decode([tid])
        if any(digit in decoded for digit in year_str):
            year_indices.append(idx)
    
    return year_indices if year_indices else [len(input_ids) - 1]

    
def extract_hidden_state(
    model,
    tokenizer,
    text: str,
    target_layer: int,
    token_position: str = "last"
) -> np.ndarray:
    """
    Extract hidden state h^(ℓ)_{y,k,i} for a single text.

    Args:
        text: The input text (date sentence)
        target_layer: Layer index i to extract from
        token_position: Which token's hidden state ("last", "mean", "first")

    Returns:
        Hidden state vector h ∈ ℝ^d
    """
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)

    if token_position == "date_tokens" and year is not None:
        # Find year tokens (from geo_exp.py approach)
        date_indices = find_year_token_indices(tokenizer, text, year)
        if len(date_indices) > 1:
            embedding = hidden_states[date_indices].mean(dim=0)
        else:
            embedding = hidden_states[date_indices[0]]

    elif token_position == "last":
        embedding = hidden_states[-1]
    elif token_position == "mean":
        embedding = hidden_states.mean(dim=0)
    else:
        embedding = hidden_states[-1]

    return embedding.cpu().float().numpy()


def extract_all_embeddings(
    model,
    tokenizer,
    date_data: Dict[str, Dict],
    target_layer: int,
    use_full_dates: bool = True
) -> Dict[str, np.ndarray]:
    """
    Extract embeddings h^(ℓ)_{y,k,i} for all languages and date samples.

    Returns:
        Dict[language -> embeddings array of shape (N, d)]
        where N = len(years) * K_SAMPLES_PER_YEAR
    """
    embeddings = {lang: [] for lang in date_data.keys()}

    for lang in tqdm(date_data.keys(), desc=f"Extracting layer {target_layer}"):
        lang_data = date_data[lang]
        n_samples = len(lang_data['years'])

        for i in range(n_samples):
            if use_full_dates:
                # Use full date: "The date is Monday, March 12, 1995."
                date_str = lang_data['date_strings'][i]
                text = DATE_TEMPLATES[lang].format(date_str)
            else:
                # Use year only: "The year is 1995."
                year = lang_data['years'][i]
                text = YEAR_TEMPLATES[lang].format(year)

            vec = extract_hidden_state(model, tokenizer, text, target_layer)
            embeddings[lang].append(vec)

    return {lang: np.array(vecs) for lang, vecs in embeddings.items()}


def compute_year_averaged_embeddings(
    embeddings: Dict[str, np.ndarray],
    date_data: Dict[str, Dict],
    years: List[int]
) -> Dict[str, np.ndarray]:
    """
    Compute average embedding h̄^(ℓ)_{y,i} = (1/K) Σ_k h^(ℓ)_{y,k,i} for each year.

    This creates year-invariant representations by averaging over K date samples.

    Returns:
        Dict[language -> averaged embeddings of shape (|Y|, d)]
    """
    averaged = {}

    for lang, emb in embeddings.items():
        year_indices = date_data[lang]['year_indices']
        n_years = len(years)
        d = emb.shape[1]

        avg_emb = np.zeros((n_years, d))
        for year_idx in range(n_years):
            mask = year_indices == year_idx
            if mask.sum() > 0:
                avg_emb[year_idx] = emb[mask].mean(axis=0)

        averaged[lang] = avg_emb

    return averaged


# ==========================================
# 4. TEMPORAL GEOMETRY METRICS
# ==========================================

class TemporalGeometry:
    """
    Implements the temporal geometry analysis from the paper.

    Key concepts:
    - Line segment: s^(ℓ)_{y,i} = h̄^(ℓ)_{y+1,i} - h̄^(ℓ)_{y,i}
    - Path: P^(ℓ)_i = (s^(ℓ)_{y1,i}, s^(ℓ)_{y2,i}, ...)
    - Path direction: Δ^(ℓ)_i = mean of line segments
    """

    def __init__(
        self,
        embeddings: Dict[str, np.ndarray],
        years: List[int],
        date_data: Optional[Dict[str, Dict]] = None
    ):
        """
        Initialize with year-averaged embeddings.

        Args:
            embeddings: Dict[lang -> (|Y|, d) array] of year-averaged embeddings
            years: List of years
            date_data: Optional raw date data for component analysis
        """
        self.embeddings = embeddings
        self.years = years
        self.date_data = date_data
        self.languages = list(embeddings.keys())
        self.n_years = len(years)

        # Compute line segments and path directions
        self._compute_line_segments()
        self._compute_path_directions()

    def _compute_line_segments(self):
        """
        Compute line segments s^(ℓ)_{y,i} = h̄^(ℓ)_{y+1,i} - h̄^(ℓ)_{y,i}

        Equation: s^(ℓ)_{y,i} = h̄^(ℓ)_{y+1,i} - h̄^(ℓ)_{y,i}
        """
        self.line_segments = {}
        for lang, emb in self.embeddings.items():
            # Shape: (|Y|-1, d)
            segments = emb[1:] - emb[:-1]
            self.line_segments[lang] = segments

    def _compute_path_directions(self):
        """
        Compute path direction Δ^(ℓ)_i as average of line segments.

        Equation: Δ^(ℓ)_i = (1/(|Y|-1)) Σ_y s^(ℓ)_{y,i}
        """
        self.path_directions = {}
        for lang, segments in self.line_segments.items():
            # Shape: (d,)
            self.path_directions[lang] = segments.mean(axis=0)

    # ==========================================
    # 4.2 Linear Structure of Time (Linearity)
    # ==========================================

    def compute_temporal_linearity(self) -> Dict[str, float]:
        """
        Test whether years have an underlying linear structure.

        Projects h̄^(ℓ)_{y,i} onto a line using linear regression:
        ŷ = W · h̄^(ℓ)_{y,i} + b

        Equation: Linearity = R²(y, ŷ)

        Returns:
            Dict[lang -> R² score]
        """
        linearity = {}
        years_arr = np.array(self.years).reshape(-1, 1)

        for lang, emb in self.embeddings.items():
            reg = LinearRegression()
            reg.fit(emb, years_arr)
            r2 = reg.score(emb, years_arr)
            linearity[lang] = float(r2)

        return linearity

    def get_year_projection_weights(self) -> Dict[str, Tuple[np.ndarray, float]]:
        """
        Get the learned linear projection weights W and intercept b.

        These define the "year direction" in embedding space.

        Returns:
            Dict[lang -> (W vector, b scalar)]
        """
        projections = {}
        years_arr = np.array(self.years).reshape(-1, 1)

        for lang, emb in self.embeddings.items():
            reg = LinearRegression()
            reg.fit(emb, years_arr)
            # W shape is (1, d), flatten to (d,)
            W = reg.coef_.flatten()
            b = float(reg.intercept_[0])
            projections[lang] = (W, b)

        return projections

    # ==========================================
    # 4.3 Calendar Component Disentanglement
    # ==========================================

    def compute_component_path_directions(
        self,
        raw_embeddings: Dict[str, np.ndarray],
        date_data: Dict[str, Dict]
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Compute path direction Δ^(ℓ)_c for each calendar component c ∈ {Y, M, D}.

        For each component, we fit a linear probe and use its weight as the direction.

        Returns:
            Dict[lang -> Dict[component -> direction vector]]
        """
        component_directions = {}

        for lang in self.languages:
            emb = raw_embeddings[lang]
            data = date_data[lang]

            component_directions[lang] = {}

            for comp_name, comp_key in [('year', 'years'), ('month', 'months'), ('day', 'days')]:
                labels = data[comp_key].reshape(-1, 1)

                reg = LinearRegression()
                reg.fit(emb, labels)

                # The weight vector defines the "component direction"
                direction = reg.coef_.flatten()
                component_directions[lang][comp_name] = direction

        return component_directions

    def compute_component_disentanglement(
        self,
        component_directions: Dict[str, Dict[str, np.ndarray]]
    ) -> Dict[str, float]:
        """
        Test whether calendar components have orthogonal path directions.

        Equation: D_{ℓ,i} = 1 - avg_{c1≠c2}|cos(Δ^(ℓ)_{c1}, Δ^(ℓ)_{c2})|

        High D means year, month, day are encoded along independent axes.

        Returns:
            Dict[lang -> disentanglement score in [0, 1]]
        """
        disentanglement = {}

        for lang, comp_dirs in component_directions.items():
            components = list(comp_dirs.keys())

            pair_sims = []
            for c1, c2 in combinations(components, 2):
                v1 = comp_dirs[c1]
                v2 = comp_dirs[c2]

                # Normalize
                v1_norm = v1 / (np.linalg.norm(v1) + 1e-8)
                v2_norm = v2 / (np.linalg.norm(v2) + 1e-8)

                # Absolute cosine similarity
                sim = abs(np.dot(v1_norm, v2_norm))
                pair_sims.append(sim)

            # D = 1 - mean(|cos|)
            D = 1.0 - np.mean(pair_sims) if pair_sims else 0.0
            disentanglement[lang] = float(D)

        return disentanglement

    # ==========================================
    # 4.4 Component Prediction Accuracy
    # ==========================================

    def compute_component_prediction_accuracy(
        self,
        raw_embeddings: Dict[str, np.ndarray],
        date_data: Dict[str, Dict]
    ) -> Dict[str, Dict[str, float]]:
        """
        Measure how accurately each calendar component can be decoded.

        Uses cross-validated R² score.

        Returns:
            Dict[lang -> Dict[component -> R² score]]
        """
        accuracy = {}

        for lang in self.languages:
            emb = raw_embeddings[lang]
            data = date_data[lang]

            accuracy[lang] = {}

            for comp_name, comp_key in [('year', 'years'), ('month', 'months'),
                                         ('day', 'days'), ('weekday', 'weekdays')]:
                labels = data[comp_key]

                try:
                    reg = Ridge(alpha=1.0)
                    scores = cross_val_score(reg, emb, labels, cv=5, scoring='r2')
                    accuracy[lang][comp_name] = float(np.mean(scores))
                except Exception as e:
                    print(f"Warning: {lang}/{comp_name}: {e}")
                    accuracy[lang][comp_name] = 0.0

        return accuracy

    # ==========================================
    # 4.5 Path Stability Analysis
    # ==========================================

    def compute_path_stability(self) -> Dict[str, float]:
        """
        Measure how consistently line segments point in the same direction.

        If line segments are stable, Δ^(ℓ)_i represents a clear "forward-in-time" direction.

        Returns:
            Dict[lang -> stability score (mean cosine with path direction)]
        """
        stability = {}

        for lang, segments in self.line_segments.items():
            path_dir = self.path_directions[lang]
            path_dir_norm = path_dir / (np.linalg.norm(path_dir) + 1e-8)

            # Compute cosine of each segment with the mean direction
            cosines = []
            for seg in segments:
                seg_norm = seg / (np.linalg.norm(seg) + 1e-8)
                cos = np.dot(seg_norm, path_dir_norm)
                cosines.append(cos)

            stability[lang] = float(np.mean(cosines))

        return stability

    def compute_segment_magnitudes(self) -> Dict[str, np.ndarray]:
        """
        Compute the magnitude of each line segment (step size through time).

        Returns:
            Dict[lang -> array of magnitudes for each year transition]
        """
        magnitudes = {}

        for lang, segments in self.line_segments.items():
            mags = np.linalg.norm(segments, axis=1)
            magnitudes[lang] = mags

        return magnitudes

    def compute_pairwise_component_similarities(
        self,
        component_directions: Dict[str, Dict[str, np.ndarray]]
    ) -> Dict[str, Dict[Tuple[str, str], float]]:
        """
        Compute pairwise cosine similarities between all calendar component directions.

        Returns:
            Dict[lang -> Dict[(comp1, comp2) -> cosine similarity]]
        """
        pairwise_sims = {}

        for lang, comp_dirs in component_directions.items():
            components = list(comp_dirs.keys())
            pairwise_sims[lang] = {}

            for c1, c2 in combinations(components, 2):
                v1 = comp_dirs[c1]
                v2 = comp_dirs[c2]

                v1_norm = v1 / (np.linalg.norm(v1) + 1e-8)
                v2_norm = v2 / (np.linalg.norm(v2) + 1e-8)

                sim = np.dot(v1_norm, v2_norm)
                pairwise_sims[lang][(c1, c2)] = float(sim)

        return pairwise_sims


# ==========================================
# 5. GEOMETRY-ACCURACY CORRELATION ANALYSIS
# ==========================================

def correlate_geometry_with_accuracy(
    geometry_results: List[Dict],
    arithmetic_results: Dict,
    task_type_mapping: Dict[str, str] = None
) -> Dict:
    """
    Analyze correlation between temporal geometry metrics and date arithmetic accuracy.

    Key hypothesis: If Y/M/D directions are more orthogonal (higher disentanglement),
    the model should perform better on tasks requiring separate manipulation of these components.

    Args:
        geometry_results: Per-layer geometry analysis results
        arithmetic_results: Date arithmetic evaluation results
        task_type_mapping: Maps arithmetic task types to relevant geometry metrics

    Returns:
        Dict with correlation analysis
    """
    from scipy.stats import pearsonr, spearmanr

    # Default mapping: which geometry metric is most relevant for each task type
    if task_type_mapping is None:
        task_type_mapping = {
            'add_days': 'day',      # Day addition requires understanding day component
            'sub_days': 'day',      # Day subtraction requires understanding day component
            'diff_days': 'day',     # Day difference requires day understanding
            'weekday': 'weekday',   # Weekday prediction
            'days_in_month': 'month',  # Requires month understanding
            'leap_year': 'year',    # Requires year understanding
        }

    # Extract geometry metrics per layer
    layers = [r['layer'] for r in geometry_results]
    n_layers = len(layers)

    # Metrics to correlate
    linearity = [r['avg_linearity'] for r in geometry_results]
    disentanglement = [r['avg_disentanglement'] for r in geometry_results]
    stability = [r['avg_stability'] for r in geometry_results]

    # Component-specific decodability
    year_acc = []
    month_acc = []
    day_acc = []

    for r in geometry_results:
        lang = list(r['component_accuracy'].keys())[0]  # Use first language
        year_acc.append(r['component_accuracy'][lang].get('year', 0))
        month_acc.append(r['component_accuracy'][lang].get('month', 0))
        day_acc.append(r['component_accuracy'][lang].get('day', 0))

    # Get task accuracies
    task_accuracies = {
        task_type: arithmetic_results[task_type]['accuracy']
        for task_type in arithmetic_results
        if task_type != 'overall_accuracy' and isinstance(arithmetic_results[task_type], dict)
    }

    correlations = {
        'linearity_vs_overall': None,
        'disentanglement_vs_overall': None,
        'stability_vs_overall': None,
        'per_task_correlations': {},
        'insights': []
    }

    # Overall accuracy correlation (if we had per-layer accuracy, but we have single eval)
    overall_acc = arithmetic_results['overall_accuracy']

    # Find best layer based on different metrics
    best_linearity_layer = int(np.argmax(linearity))
    best_disent_layer = int(np.argmax(disentanglement))
    best_stability_layer = int(np.argmax(stability))

    correlations['best_layers'] = {
        'linearity': best_linearity_layer,
        'disentanglement': best_disent_layer,
        'stability': best_stability_layer,
    }

    correlations['metrics_at_best_layers'] = {
        'max_linearity': float(linearity[best_linearity_layer]),
        'max_disentanglement': float(disentanglement[best_disent_layer]),
        'max_stability': float(stability[best_stability_layer]),
    }

    # Component decodability correlations
    correlations['component_decodability'] = {
        'year': {'max': float(max(year_acc)), 'best_layer': int(np.argmax(year_acc))},
        'month': {'max': float(max(month_acc)), 'best_layer': int(np.argmax(month_acc))},
        'day': {'max': float(max(day_acc)), 'best_layer': int(np.argmax(day_acc))},
    }

    # Task-specific analysis
    for task_type, relevant_comp in task_type_mapping.items():
        if task_type in task_accuracies:
            task_acc = task_accuracies[task_type]

            # Get relevant component decodability
            if relevant_comp == 'year':
                comp_acc = max(year_acc)
            elif relevant_comp == 'month':
                comp_acc = max(month_acc)
            elif relevant_comp in ['day', 'weekday']:
                comp_acc = max(day_acc)
            else:
                comp_acc = 0

            correlations['per_task_correlations'][task_type] = {
                'task_accuracy': task_acc,
                'relevant_component': relevant_comp,
                'component_decodability': comp_acc,
            }

    # Generate insights
    if max(disentanglement) > 0.5:
        correlations['insights'].append(
            f"High disentanglement (D={max(disentanglement):.3f}) suggests Y/M/D are separable"
        )
    else:
        correlations['insights'].append(
            f"Low disentanglement (D={max(disentanglement):.3f}) suggests Y/M/D are entangled"
        )

    # Check if component decodability correlates with task performance
    if max(day_acc) > 0.5 and task_accuracies.get('add_days', 0) > 0.5:
        correlations['insights'].append(
            "Day decodability correlates with day arithmetic performance"
        )

    if max(year_acc) > 0.5 and task_accuracies.get('leap_year', 0) > 0.5:
        correlations['insights'].append(
            "Year decodability correlates with leap year task performance"
        )

    return correlations


def analyze_per_task_geometry_correlation(
    model,
    tokenizer,
    arithmetic_tasks: List[Dict],
    date_data: Dict[str, Dict],
    years: List[int],
    target_layer: int
) -> Dict:
    """
    Analyze how geometric properties of specific dates correlate with task accuracy.

    For each arithmetic task, we:
    1. Extract embedding of the input date
    2. Measure its position relative to component directions
    3. Correlate with task correctness
    """
    # Get embeddings and geometry for the target layer
    raw_embeddings = extract_all_embeddings(model, tokenizer, date_data, target_layer, True)
    avg_embeddings = compute_year_averaged_embeddings(raw_embeddings, date_data, years)
    geometry = TemporalGeometry(avg_embeddings, years, date_data)

    # Get component directions
    component_dirs = geometry.compute_component_path_directions(raw_embeddings, date_data)

    analysis = {
        'per_task_analysis': [],
        'correlation_summary': {}
    }

    lang = 'en'  # Focus on English
    if lang in component_dirs:
        year_dir = component_dirs[lang].get('year')
        month_dir = component_dirs[lang].get('month')
        day_dir = component_dirs[lang].get('day')

        if year_dir is not None:
            year_dir = year_dir / (np.linalg.norm(year_dir) + 1e-8)
        if month_dir is not None:
            month_dir = month_dir / (np.linalg.norm(month_dir) + 1e-8)
        if day_dir is not None:
            day_dir = day_dir / (np.linalg.norm(day_dir) + 1e-8)

        # For each task, measure alignment of input date with component directions
        for task in arithmetic_tasks[:50]:  # Sample for efficiency
            if 'date_info' in task:
                info = task['date_info']

                # Check if task requires specific component understanding
                task_type = task['type']
                correct = task.get('correct', None)

                analysis['per_task_analysis'].append({
                    'task_type': task_type,
                    'correct': correct,
                    'date_info': info,
                })

    return analysis

def analyze_layer(
    model,
    tokenizer,
    date_data: Dict[str, Dict],
    years: List[int],
    layer: int,
    use_full_dates: bool = True
) -> Dict:
    """
    Comprehensive temporal geometry analysis for a single layer.

    Returns dictionary with all metrics defined in the paper.
    """
    # 1. Extract raw embeddings (all K samples)
    raw_embeddings = extract_all_embeddings(
        model, tokenizer, date_data, layer, use_full_dates
    )

    # 2. Compute year-averaged embeddings h̄^(ℓ)_{y,i}
    avg_embeddings = compute_year_averaged_embeddings(raw_embeddings, date_data, years)

    # 3. Initialize geometry analyzer
    geometry = TemporalGeometry(avg_embeddings, years, date_data)

    # 4. Compute all metrics

    # 4.1 Temporal linearity R² (Do LLMs build linear time representations?)
    linearity = geometry.compute_temporal_linearity()

    # 4.2 Path stability (How consistent is the forward-in-time direction?)
    stability = geometry.compute_path_stability()

    # 4.3 Component directions (Year, Month, Day directions in embedding space)
    component_dirs = geometry.compute_component_path_directions(raw_embeddings, date_data)

    # 4.4 Component disentanglement D_{ℓ,i} (Are Y/M/D orthogonal?)
    disentanglement = geometry.compute_component_disentanglement(component_dirs)

    # 4.5 Component prediction accuracy (Can we decode Y/M/D from embeddings?)
    component_accuracy = geometry.compute_component_prediction_accuracy(raw_embeddings, date_data)

    # 4.6 Segment magnitudes (for trajectory analysis)
    segment_magnitudes = geometry.compute_segment_magnitudes()

    return {
        'layer': layer,
        'linearity': linearity,
        'avg_linearity': np.mean(list(linearity.values())),
        'stability': stability,
        'avg_stability': np.mean(list(stability.values())),
        'disentanglement': disentanglement,
        'avg_disentanglement': np.mean(list(disentanglement.values())),
        'component_accuracy': component_accuracy,
        'segment_magnitudes': segment_magnitudes,
        'embeddings': avg_embeddings,
        'raw_embeddings': raw_embeddings,
        'path_directions': geometry.path_directions,
        'component_directions': component_dirs,
    }


def analyze_all_layers(
    model,
    tokenizer,
    date_data: Dict[str, Dict],
    years: List[int],
    use_full_dates: bool = True
) -> List[Dict]:
    """Analyze all layers of the model."""
    # Get number of layers
    n_layers = model.config.num_hidden_layers + 1  # +1 for embedding layer

    results = []
    for layer in tqdm(range(n_layers), desc="Analyzing layers"):
        result = analyze_layer(model, tokenizer, date_data, years, layer, use_full_dates)
        results.append(result)

    return results


# ==========================================
# 6. VISUALIZATION FUNCTIONS
# ==========================================

def plot_metrics_across_layers(results: List[Dict], save_dir: str = "results"):
    """
    Plot all key metrics across layers to answer the research questions.
    """
    os.makedirs(save_dir, exist_ok=True)

    layers = [r['layer'] for r in results]

    # ==========================================
    # Figure 1: Linear Structure of Time
    # (Do LLMs build internal linear representations of time?)
    # ==========================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1a. Linearity (R²) across layers per language
    ax = axes[0]
    for lang in LANGUAGES:
        linearity_vals = [r['linearity'][lang] for r in results]
        ax.plot(layers, linearity_vals, marker='o', label=lang.upper(), linewidth=2, markersize=4)

    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Linearity (R²)', fontsize=12)
    ax.set_title('Linear Structure of Time\n(Higher = Years form a line in embedding space)', fontsize=13)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-0.05, 1.05])

    # 1b. Average linearity with path stability
    ax = axes[1]
    avg_linearity = [r['avg_linearity'] for r in results]
    avg_stability = [r['avg_stability'] for r in results]

    ax.plot(layers, avg_linearity, 'b-o', label='Linearity (R²)', linewidth=2, markersize=5)
    ax.plot(layers, avg_stability, 'g-s', label='Path Stability', linewidth=2, markersize=5)

    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Temporal Representation Quality\n(Do LLMs build linear time like humans?)', fontsize=13)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'linearity_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # ==========================================
    # Figure 2: Calendar Component Disentanglement
    # (Do LLMs split dates into day/month/year?)
    # ==========================================
    fig, ax = plt.subplots(figsize=(10, 6))

    disent_vals = [r['avg_disentanglement'] for r in results]
    ax.plot(layers, disent_vals, 'r-o', linewidth=2, markersize=6)

    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Disentanglement Score D', fontsize=12)
    ax.set_title('Calendar Component Disentanglement\n(D=1: Year/Month/Day are orthogonal; D=0: entangled)', fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-0.05, 1.05])
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Partial separation')
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'component_disentanglement.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # ==========================================
    # Figure 3: Component Prediction Accuracy
    # (How accurately can we decode Year/Month/Day?)
    # ==========================================
    fig, ax = plt.subplots(figsize=(12, 6))

    components = ['year', 'month', 'day', 'weekday']
    colors = {'year': '#1f77b4', 'month': '#ff7f0e', 'day': '#2ca02c', 'weekday': '#d62728'}

    for comp in components:
        # Average across languages
        avg_accuracy = []
        for r in results:
            accs = [r['component_accuracy'][lang].get(comp, 0) for lang in LANGUAGES]
            avg_accuracy.append(np.mean(accs))

        ax.plot(layers, avg_accuracy, marker='o', label=comp.capitalize(),
                color=colors[comp], linewidth=2, markersize=5)

    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Prediction Accuracy (R²)', fontsize=12)
    ax.set_title('Calendar Component Decodability\n(Can we extract Year/Month/Day from hidden states?)', fontsize=13)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-0.1, 1.1])
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'component_accuracy.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # ==========================================
    # Figure 4: Summary Dashboard
    # ==========================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 4a. Linearity - Do LLMs build linear time representations?
    ax = axes[0, 0]
    ax.plot(layers, [r['avg_linearity'] for r in results], 'b-o', linewidth=2)
    ax.set_title('Q1: Is time LINEAR in embedding space?\n(Linearity R²)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Layer')
    ax.set_ylabel('R²')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-0.05, 1.05])
    ax.axhline(y=0.7, color='green', linestyle='--', alpha=0.5, label='Strong linearity')
    ax.legend(fontsize=9)

    # 4b. Path Stability - Is temporal direction consistent?
    ax = axes[0, 1]
    ax.plot(layers, [r['avg_stability'] for r in results], 'g-o', linewidth=2)
    ax.set_title('Q2: Is forward-in-time direction STABLE?\n(Path Stability)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Avg Cosine with Mean Direction')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-0.1, 1.1])

    # 4c. Disentanglement - Are Y/M/D separated?
    ax = axes[1, 0]
    ax.plot(layers, [r['avg_disentanglement'] for r in results], 'r-o', linewidth=2)
    ax.set_title('Q3: Are Year/Month/Day SEPARATED?\n(Disentanglement D)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Layer')
    ax.set_ylabel('D score (1=orthogonal)')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-0.05, 1.05])
    ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Partial separation')
    ax.legend(fontsize=9)

    # 4d. Year prediction accuracy
    ax = axes[1, 1]
    year_acc = []
    for r in results:
        accs = [r['component_accuracy'][lang].get('year', 0) for lang in LANGUAGES]
        year_acc.append(np.mean(accs))
    ax.plot(layers, year_acc, 'm-o', linewidth=2)
    ax.set_title('Q4: Can we DECODE the year?\n(Year Prediction R²)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Layer')
    ax.set_ylabel('R²')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-0.1, 1.1])

    plt.suptitle('Temporal Geometry Analysis: Key Research Questions', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'summary_dashboard.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Plots saved to {save_dir}/")


def plot_embedding_trajectories(results: List[Dict], layers_to_plot: List[int], save_dir: str = "results"):
    """
    Visualize temporal trajectories in 2D/3D PCA space.
    """
    os.makedirs(save_dir, exist_ok=True)

    for layer in layers_to_plot:
        result = results[layer]
        embeddings = result['embeddings']

        # Concatenate all languages for joint PCA
        all_emb = np.vstack([embeddings[lang] for lang in LANGUAGES])
        pca = PCA(n_components=3)
        all_projected = pca.fit_transform(all_emb)

        # Split back by language
        n_years = len(results[0]['embeddings'][LANGUAGES[0]])
        idx = 0
        projected = {}
        for lang in LANGUAGES:
            projected[lang] = all_projected[idx:idx+n_years]
            idx += n_years

        # 2D plot
        fig, ax = plt.subplots(figsize=(12, 8))

        colors = plt.cm.tab10(np.linspace(0, 1, len(LANGUAGES)))
        years = list(range(1990, 2025))

        for i, lang in enumerate(LANGUAGES):
            proj = projected[lang]
            ax.plot(proj[:, 0], proj[:, 1], 'o-', color=colors[i],
                    label=lang.upper(), linewidth=1.5, markersize=4, alpha=0.8)

            # Mark start and end
            ax.scatter(proj[0, 0], proj[0, 1], s=100, c=[colors[i]], marker='s',
                       edgecolors='black', linewidths=2, zorder=5)
            ax.scatter(proj[-1, 0], proj[-1, 1], s=100, c=[colors[i]], marker='^',
                       edgecolors='black', linewidths=2, zorder=5)

        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
        ax.set_title(f'Temporal Trajectory in Embedding Space (Layer {layer})\n□=1990, △=2024', fontsize=13)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'trajectory_layer{layer}.png'), dpi=150, bbox_inches='tight')
        plt.close()


def plot_component_directions_heatmap(results: List[Dict], layer: int, save_dir: str = "results"):
    """
    Plot heatmap of pairwise cosine similarities between component directions (Y, M, D).
    Shows how orthogonal/aligned the calendar components are.
    """
    os.makedirs(save_dir, exist_ok=True)

    result = results[layer]
    comp_dirs = result['component_directions']

    # Use first language
    lang = LANGUAGES[0]
    directions = comp_dirs[lang]
    components = ['year', 'month', 'day']

    # Build similarity matrix
    n_comps = len(components)
    sim_matrix = np.zeros((n_comps, n_comps))

    for i, c1 in enumerate(components):
        for j, c2 in enumerate(components):
            v1 = directions[c1]
            v2 = directions[c2]
            v1_norm = v1 / (np.linalg.norm(v1) + 1e-8)
            v2_norm = v2 / (np.linalg.norm(v2) + 1e-8)
            sim_matrix[i, j] = np.dot(v1_norm, v2_norm)

    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(sim_matrix, cmap='RdYlBu', vmin=-1, vmax=1)

    ax.set_xticks(range(n_comps))
    ax.set_yticks(range(n_comps))
    ax.set_xticklabels([c.capitalize() for c in components])
    ax.set_yticklabels([c.capitalize() for c in components])

    # Add values
    for i in range(n_comps):
        for j in range(n_comps):
            ax.text(j, i, f'{sim_matrix[i, j]:.2f}', ha='center', va='center',
                    color='black' if -0.3 < sim_matrix[i, j] < 0.7 else 'white', fontsize=14)

    plt.colorbar(im, label='Cosine Similarity')
    ax.set_title(f'Calendar Component Direction Similarity (Layer {layer})\n(Values near 0 = orthogonal/disentangled)', fontsize=13)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'component_similarity_layer{layer}.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_geometry_accuracy_correlation(
    geometry_results: List[Dict],
    arithmetic_results: Dict,
    correlation_analysis: Dict,
    save_dir: str = "results"
):
    """
    Plot correlation between geometric properties and date arithmetic accuracy.
    """
    os.makedirs(save_dir, exist_ok=True)

    layers = [r['layer'] for r in geometry_results]

    # Extract metrics
    linearity = [r['avg_linearity'] for r in geometry_results]
    disentanglement = [r['avg_disentanglement'] for r in geometry_results]

    # Component decodability
    lang = list(geometry_results[0]['component_accuracy'].keys())[0]
    year_acc = [r['component_accuracy'][lang].get('year', 0) for r in geometry_results]
    month_acc = [r['component_accuracy'][lang].get('month', 0) for r in geometry_results]
    day_acc = [r['component_accuracy'][lang].get('day', 0) for r in geometry_results]

    # Get task accuracies
    task_types = ['add_days', 'sub_days', 'diff_days', 'weekday', 'days_in_month', 'leap_year']
    task_accs = {t: arithmetic_results[t]['accuracy'] for t in task_types
                 if t in arithmetic_results and isinstance(arithmetic_results[t], dict)}

    # ==========================================
    # Figure: Geometry vs Task Accuracy Dashboard
    # ==========================================
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # 1. Disentanglement across layers with task accuracy reference
    ax = axes[0, 0]
    ax.plot(layers, disentanglement, 'r-o', linewidth=2, markersize=5, label='Disentanglement D')
    ax.axhline(y=arithmetic_results['overall_accuracy'], color='blue', linestyle='--',
               linewidth=2, label=f'Overall Task Acc: {arithmetic_results["overall_accuracy"]:.2f}')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Score')
    ax.set_title('Disentanglement vs Overall Accuracy')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-0.05, 1.05])

    # 2. Component decodability across layers
    ax = axes[0, 1]
    ax.plot(layers, year_acc, 'b-o', label='Year R²', linewidth=2, markersize=4)
    ax.plot(layers, month_acc, 'orange', marker='s', label='Month R²', linewidth=2, markersize=4)
    ax.plot(layers, day_acc, 'g-^', label='Day R²', linewidth=2, markersize=4)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Decodability (R²)')
    ax.set_title('Component Decodability Across Layers')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-0.1, 1.1])

    # 3. Task accuracy by type
    ax = axes[0, 2]
    task_names = list(task_accs.keys())
    task_values = [task_accs[t] for t in task_names]
    colors = plt.cm.Set2(np.linspace(0, 1, len(task_names)))
    bars = ax.bar(range(len(task_names)), task_values, color=colors)
    ax.set_xticks(range(len(task_names)))
    ax.set_xticklabels([t.replace('_', '\n') for t in task_names], fontsize=9)
    ax.set_ylabel('Accuracy')
    ax.set_title('Date Arithmetic Task Accuracy')
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, val in zip(bars, task_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.2f}', ha='center', fontsize=9)

    # 4. Component decodability vs related task accuracy
    ax = axes[1, 0]
    # Map tasks to components
    task_comp_map = {
        'add_days': ('Day Tasks', max(day_acc), task_accs.get('add_days', 0)),
        'sub_days': ('Day Tasks', max(day_acc), task_accs.get('sub_days', 0)),
        'days_in_month': ('Month Tasks', max(month_acc), task_accs.get('days_in_month', 0)),
        'leap_year': ('Year Tasks', max(year_acc), task_accs.get('leap_year', 0)),
    }

    comp_x = []
    task_y = []
    labels = []
    for task, (label, comp_val, task_val) in task_comp_map.items():
        comp_x.append(comp_val)
        task_y.append(task_val)
        labels.append(task)

    ax.scatter(comp_x, task_y, s=100, c=['blue', 'green', 'orange', 'red'], alpha=0.7)
    for i, label in enumerate(labels):
        ax.annotate(label.replace('_', ' '), (comp_x[i], task_y[i]),
                   textcoords="offset points", xytext=(5, 5), fontsize=9)
    ax.set_xlabel('Max Component Decodability (R²)')
    ax.set_ylabel('Task Accuracy')
    ax.set_title('Component Decodability vs Task Accuracy')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-0.1, 1.1])
    ax.set_ylim([-0.1, 1.1])

    # Add diagonal reference line
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='y=x')

    # 5. Linearity vs Disentanglement scatter (per layer)
    ax = axes[1, 1]
    scatter = ax.scatter(linearity, disentanglement, c=layers, cmap='viridis', s=80, alpha=0.7)
    plt.colorbar(scatter, ax=ax, label='Layer')
    ax.set_xlabel('Linearity (R²)')
    ax.set_ylabel('Disentanglement D')
    ax.set_title('Linearity vs Disentanglement\n(Each point = one layer)')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])

    # 6. Summary insights
    ax = axes[1, 2]
    ax.axis('off')

    insights_text = "KEY INSIGHTS:\n\n"

    # Best layer analysis
    best_disent_layer = int(np.argmax(disentanglement))
    best_linearity_layer = int(np.argmax(linearity))

    insights_text += f"Best Disentanglement: Layer {best_disent_layer} (D={max(disentanglement):.3f})\n"
    insights_text += f"Best Linearity: Layer {best_linearity_layer} (R²={max(linearity):.3f})\n\n"

    insights_text += f"Overall Task Accuracy: {arithmetic_results['overall_accuracy']:.1%}\n\n"

    # Component analysis
    insights_text += "Component Decodability:\n"
    insights_text += f"  Year:  max R²={max(year_acc):.3f} (layer {np.argmax(year_acc)})\n"
    insights_text += f"  Month: max R²={max(month_acc):.3f} (layer {np.argmax(month_acc)})\n"
    insights_text += f"  Day:   max R²={max(day_acc):.3f} (layer {np.argmax(day_acc)})\n\n"

    # Correlation insights
    if correlation_analysis and 'insights' in correlation_analysis:
        insights_text += "Correlations:\n"
        for insight in correlation_analysis['insights'][:3]:
            insights_text += f"  • {insight}\n"

    ax.text(0.05, 0.95, insights_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('Geometry-Accuracy Correlation Analysis', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'geometry_accuracy_correlation.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Correlation plot saved to {save_dir}/geometry_accuracy_correlation.png")


# ==========================================
# 7. RESULTS SUMMARY AND INTERPRETATION
# ==========================================

def summarize_findings(results: List[Dict]) -> Dict:
    """
    Summarize findings to answer the key research questions:
    1. Do LLMs build an internal linear representation of time (like humans)?
    2. Do LLMs split dates into day/month/year and understand them separately?
    
    IMPORTANT: We focus on LATER layers (final 1/3 of model) because:
    - Early layers just do token embedding, not semantic processing
    - Later layers determine actual model behavior/outputs
    - Any linear structure in early layers is meaningless for generation
    """
    n_layers = len(results)
    
    # Define "later layers" as final 1/3 of the model
    later_layer_start = 2 * n_layers // 3
    later_layers = list(range(later_layer_start, n_layers))
    
    # Extract metrics
    linearity_by_layer = [r['avg_linearity'] for r in results]
    stability_by_layer = [r['avg_stability'] for r in results]
    disent_by_layer = [r['avg_disentanglement'] for r in results]

    # Metrics for ALL layers (for reference)
    best_linearity_layer = int(np.argmax(linearity_by_layer))
    best_stability_layer = int(np.argmax(stability_by_layer))
    best_disent_layer = int(np.argmax(disent_by_layer))
    
    # Metrics for LATER layers only (what actually matters)
    later_linearity = [linearity_by_layer[l] for l in later_layers]
    later_stability = [stability_by_layer[l] for l in later_layers]
    later_disent = [disent_by_layer[l] for l in later_layers]
    
    best_later_linearity_layer = later_layers[int(np.argmax(later_linearity))]
    best_later_stability_layer = later_layers[int(np.argmax(later_stability))]
    best_later_disent_layer = later_layers[int(np.argmax(later_disent))]
    
    max_later_linearity = float(np.max(later_linearity))
    max_later_disent = float(np.max(later_disent))
    avg_later_linearity = float(np.mean(later_linearity))
    avg_later_disent = float(np.mean(later_disent))

    # Get component prediction accuracy
    year_acc = []
    month_acc = []
    day_acc = []
    for r in results:
        year_acc.append(np.mean([r['component_accuracy'][lang].get('year', 0) for lang in LANGUAGES]))
        month_acc.append(np.mean([r['component_accuracy'][lang].get('month', 0) for lang in LANGUAGES]))
        day_acc.append(np.mean([r['component_accuracy'][lang].get('day', 0) for lang in LANGUAGES]))
    
    # Later layer component accuracy
    later_year_acc = [year_acc[l] for l in later_layers]
    later_month_acc = [month_acc[l] for l in later_layers]
    later_day_acc = [day_acc[l] for l in later_layers]

    best_year_acc_layer = int(np.argmax(year_acc))
    best_later_year_layer = later_layers[int(np.argmax(later_year_acc))]

    summary = {
        'n_layers': n_layers,
        'later_layer_start': later_layer_start,
        'later_layers': later_layers,

        # Q1: Do LLMs build linear representations of time? (FOCUS ON LATER LAYERS)
        'linearity': {
            # All layers (for reference)
            'max_value_all': float(np.max(linearity_by_layer)),
            'best_layer_all': best_linearity_layer,
            # LATER layers (what matters)
            'max_value': max_later_linearity,
            'avg_value': avg_later_linearity,
            'best_layer': best_later_linearity_layer,
            'final_layer_value': float(linearity_by_layer[-1]),
            'interpretation': 'YES - LLMs build LINEAR time representations in later layers'
                             if max_later_linearity > 0.7 else
                             'PARTIAL - Time has some linear structure in later layers'
                             if max_later_linearity > 0.4 else
                             'NO - Time is NOT linearly represented in later layers (behavior-relevant)'
        },

        # Path stability
        'stability': {
            'max_value_all': float(np.max(stability_by_layer)),
            'max_value': float(np.max(later_stability)),
            'best_layer': best_later_stability_layer,
            'final_layer_value': float(stability_by_layer[-1]),
        },

        # Q2: Do LLMs split dates into components? (FOCUS ON LATER LAYERS)
        'disentanglement': {
            # All layers
            'max_value_all': float(np.max(disent_by_layer)),
            'best_layer_all': best_disent_layer,
            # LATER layers (what matters)
            'max_value': max_later_disent,
            'avg_value': avg_later_disent,
            'best_layer': best_later_disent_layer,
            'final_layer_value': float(disent_by_layer[-1]),
            'interpretation': 'YES - Year/Month/Day ARE encoded along SEPARATE axes in later layers'
                             if max_later_disent > 0.6 else
                             'PARTIAL - Calendar components are partially disentangled in later layers'
                             if max_later_disent > 0.3 else
                             'NO - Calendar components are ENTANGLED in later layers'
        },

        # Component decodability (focus on later layers)
        'year_decodability': {
            'max_accuracy_all': float(np.max(year_acc)),
            'max_accuracy': float(np.max(later_year_acc)),
            'best_layer': best_later_year_layer,
        },
        'month_decodability': {
            'max_accuracy_all': float(np.max(month_acc)),
            'max_accuracy': float(np.max(later_month_acc)),
            'best_layer': later_layers[int(np.argmax(later_month_acc))],
        },
        'day_decodability': {
            'max_accuracy_all': float(np.max(day_acc)),
            'max_accuracy': float(np.max(later_day_acc)),
            'best_layer': later_layers[int(np.argmax(later_day_acc))],
        },

        # Overall conclusions
        'conclusions': []
    }

    # Generate conclusions (BASED ON LATER LAYERS)
    summary['conclusions'].append(
        f"NOTE: Conclusions based on LATER layers ({later_layer_start}-{n_layers-1}) which determine model behavior"
    )
    
    # Q1: Linear time representation in later layers
    if max_later_linearity > 0.7:
        summary['conclusions'].append(
            f"✓ Q1: LLMs DO build LINEAR time representations in later layers (R²={max_later_linearity:.3f} at layer {best_later_linearity_layer})"
        )
    elif max_later_linearity > 0.4:
        summary['conclusions'].append(
            f"~ Q1: Time has PARTIAL linear structure in later layers (R²={max_later_linearity:.3f}, avg={avg_later_linearity:.3f})"
        )
    else:
        summary['conclusions'].append(
            f"✗ Q1: Time is NOT linearly represented in later layers (R²={max_later_linearity:.3f}) - this matters for behavior!"
        )

    # Q2: Component disentanglement in later layers
    if max_later_disent > 0.6:
        summary['conclusions'].append(
            f"✓ Q2: LLMs DO separate Year/Month/Day in later layers (D={max_later_disent:.3f})"
        )
    elif max_later_disent > 0.3:
        summary['conclusions'].append(
            f"~ Q2: Calendar components are PARTIALLY separated in later layers (D={max_later_disent:.3f})"
        )
    else:
        summary['conclusions'].append(
            f"✗ Q2: Calendar components are ENTANGLED in later layers (D={max_later_disent:.3f})"
        )

    # Component decodability insights (later layers)
    summary['conclusions'].append(
        f"   Later layer decodability - Year: R²={summary['year_decodability']['max_accuracy']:.3f} | "
        f"Month: R²={summary['month_decodability']['max_accuracy']:.3f} | "
        f"Day: R²={summary['day_decodability']['max_accuracy']:.3f}"
    )
    
    # Compare early vs late (highlight if early layers are misleading)
    if float(np.max(linearity_by_layer[:later_layer_start])) > max_later_linearity + 0.2:
        summary['conclusions'].append(
            f"⚠ WARNING: Linearity is HIGHER in early layers ({np.max(linearity_by_layer[:later_layer_start]):.3f}) than later ({max_later_linearity:.3f}) - early layer results are misleading!"
        )

    return summary


def print_summary(summary: Dict):
    """Print a formatted summary of findings."""
    print("\n" + "="*70)
    print("TEMPORAL GEOMETRY ANALYSIS SUMMARY")
    print("="*70)

    later_start = summary.get('later_layer_start', 0)
    print(f"\nModel analyzed with {summary['n_layers']} layers")
    print(f"*** FOCUS: Later layers ({later_start}-{summary['n_layers']-1}) which determine model behavior ***")

    print("\n" + "-"*70)
    print("KEY RESEARCH QUESTIONS:")
    print("-"*70)
    print("Q1: Do LLMs build an internal LINEAR representation of time (like humans)?")
    print("Q2: Do LLMs split dates into Year/Month/Day and understand them SEPARATELY?")

    print("\n" + "-"*70)
    print("FINDINGS (BASED ON LATER LAYERS):")
    print("-"*70)

    for conclusion in summary['conclusions']:
        print(f"  {conclusion}")

    print("\n" + "-"*70)
    print("DETAILED METRICS (Later Layers vs All Layers):")
    print("-"*70)

    print(f"\n1. Linearity (Is time linear in embedding space?)")
    print(f"   Later Layers: R²={summary['linearity']['max_value']:.4f} (Layer {summary['linearity']['best_layer']}), avg={summary['linearity'].get('avg_value', 0):.4f}")
    if 'max_value_all' in summary['linearity']:
        print(f"   All Layers:   R²={summary['linearity']['max_value_all']:.4f} (Layer {summary['linearity'].get('best_layer_all', '?')})")
    print(f"   Final Layer:  R²={summary['linearity']['final_layer_value']:.4f}")
    print(f"   Interpretation: {summary['linearity']['interpretation']}")

    print(f"\n2. Disentanglement (Are Year/Month/Day orthogonal directions?)")
    print(f"   Later Layers: D={summary['disentanglement']['max_value']:.4f} (Layer {summary['disentanglement']['best_layer']}), avg={summary['disentanglement'].get('avg_value', 0):.4f}")
    if 'max_value_all' in summary['disentanglement']:
        print(f"   All Layers:   D={summary['disentanglement']['max_value_all']:.4f} (Layer {summary['disentanglement'].get('best_layer_all', '?')})")
    print(f"   Final Layer:  D={summary['disentanglement']['final_layer_value']:.4f}")
    print(f"   Interpretation: {summary['disentanglement']['interpretation']}")

    print(f"\n3. Component Decodability in Later Layers (Can we extract Y/M/D?)")
    print(f"   Year:  R²={summary['year_decodability']['max_accuracy']:.4f} (Layer {summary['year_decodability']['best_layer']})")
    print(f"   Month: R²={summary['month_decodability']['max_accuracy']:.4f} (Layer {summary['month_decodability']['best_layer']})")
    print(f"   Day:   R²={summary['day_decodability']['max_accuracy']:.4f} (Layer {summary['day_decodability']['best_layer']})")
    if 'max_accuracy_all' in summary['year_decodability']:
        print(f"\n   (For reference - All layers max: Year={summary['year_decodability']['max_accuracy_all']:.4f}, "
              f"Month={summary['month_decodability']['max_accuracy_all']:.4f}, "
              f"Day={summary['day_decodability']['max_accuracy_all']:.4f})")

    print("\n" + "="*70)


# ==========================================
# 8. MAIN EXECUTION
# ==========================================

def main(model_name: str = MODEL_NAME, save_dir: str = "results", run_arithmetic: bool = True, n_arithmetic_tasks: int = 100):
    """
    Main analysis pipeline.

    Answers:
    1. Do LLMs split dates into day/month/year and understand them separately?
    2. Do LLMs build an internal linear representation of time like humans do?
    3. How does geometric structure correlate with date arithmetic performance?
    """
    print("="*70)
    print("TEMPORAL GEOMETRY ANALYSIS")
    print("="*70)
    print(f"\nModel: {model_name}")
    print(f"Languages: {LANGUAGES}")
    print(f"Years: {YEARS[0]}-{YEARS[-1]}")
    print(f"Samples per year: {K_SAMPLES_PER_YEAR}")
    print(f"Date arithmetic evaluation: {run_arithmetic}")

    # 1. Load model
    print("\n[1/7] Loading model...")
    model, tokenizer = load_model(model_name)
    n_layers = model.config.num_hidden_layers + 1
    print(f"Model has {n_layers} layers")

    # 2. Generate date samples
    print("\n[2/7] Generating date samples...")
    date_data = generate_date_samples(YEARS, LANGUAGES, K_SAMPLES_PER_YEAR)
    n_samples = len(date_data['en']['years'])
    print(f"Generated {n_samples} total samples ({len(YEARS)} years × {K_SAMPLES_PER_YEAR} samples)")

    # 3. Analyze all layers
    print("\n[3/7] Analyzing layers...")
    results = analyze_all_layers(model, tokenizer, date_data, YEARS, use_full_dates=True)

    # 4. Date arithmetic evaluation
    arithmetic_results = None
    correlation_analysis = None

    if run_arithmetic:
        print(f"\n[4/7] Evaluating date arithmetic ({n_arithmetic_tasks} tasks)...")
        arithmetic_tasks = generate_date_arithmetic_tasks(n_tasks=n_arithmetic_tasks)
        arithmetic_results = evaluate_date_arithmetic(model, tokenizer, arithmetic_tasks)

        print(f"\n   Date Arithmetic Results:")
        print(f"   Overall Accuracy: {arithmetic_results['overall_accuracy']:.2%}")
        for task_type in ['add_days', 'sub_days', 'diff_days', 'weekday', 'days_in_month', 'leap_year']:
            if task_type in arithmetic_results and isinstance(arithmetic_results[task_type], dict):
                acc = arithmetic_results[task_type]['accuracy']
                n = arithmetic_results[task_type]['total']
                print(f"   {task_type}: {acc:.2%} ({int(acc*n)}/{n})")

        # 5. Correlate geometry with arithmetic accuracy
        print("\n[5/7] Correlating geometry with task performance...")
        correlation_analysis = correlate_geometry_with_accuracy(results, arithmetic_results)

        if correlation_analysis and 'insights' in correlation_analysis:
            print("\n   Correlation Insights:")
            for insight in correlation_analysis['insights']:
                print(f"   • {insight}")
    else:
        print("\n[4/7] Skipping date arithmetic evaluation (use --arithmetic to enable)")
        print("\n[5/7] Skipping correlation analysis")

    # 6. Generate visualizations
    print("\n[6/7] Generating visualizations...")
    os.makedirs(save_dir, exist_ok=True)

    plot_metrics_across_layers(results, save_dir)

    # Plot trajectories for early, middle, and late layers
    key_layers = [0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-1]
    key_layers = [l for l in key_layers if l < n_layers]
    plot_embedding_trajectories(results, key_layers, save_dir)

    # Plot component direction similarity at best disentanglement layer
    disent_vals = [r['avg_disentanglement'] for r in results]
    best_disent_layer = int(np.argmax(disent_vals))
    plot_component_directions_heatmap(results, best_disent_layer, save_dir)

    # Plot geometry-accuracy correlation if arithmetic was evaluated
    if arithmetic_results is not None:
        plot_geometry_accuracy_correlation(results, arithmetic_results, correlation_analysis, save_dir)

    # 7. Summarize findings
    print("\n[7/7] Summarizing findings...")
    summary = summarize_findings(results)
    print_summary(summary)

    # Print arithmetic summary if available
    if arithmetic_results is not None:
        print("\n" + "="*70)
        print("DATE ARITHMETIC PERFORMANCE")
        print("="*70)
        print(f"\nOverall Accuracy: {arithmetic_results['overall_accuracy']:.2%}")
        print(f"\nPer-Task Breakdown:")
        for task_type in ['add_days', 'sub_days', 'diff_days', 'weekday', 'days_in_month', 'leap_year']:
            if task_type in arithmetic_results and isinstance(arithmetic_results[task_type], dict):
                acc = arithmetic_results[task_type]['accuracy']
                print(f"  {task_type:15s}: {acc:.2%}")

        if correlation_analysis and 'insights' in correlation_analysis:
            print("\n" + "-"*70)
            print("GEOMETRY-ACCURACY CORRELATIONS:")
            print("-"*70)
            for insight in correlation_analysis['insights']:
                print(f"  • {insight}")

    # Save results
    results_to_save = {
        'model': model_name,
        'languages': LANGUAGES,
        'years': YEARS,
        'k_samples': K_SAMPLES_PER_YEAR,
        'summary': summary,
        'layer_metrics': [
            {
                'layer': r['layer'],
                'avg_linearity': r['avg_linearity'],
                'avg_stability': r['avg_stability'],
                'avg_disentanglement': r['avg_disentanglement'],
                'linearity': r['linearity'],
                'disentanglement': r['disentanglement'],
                'component_accuracy': r['component_accuracy'],
            }
            for r in results
        ]
    }

    if arithmetic_results is not None:
        results_to_save['arithmetic_evaluation'] = {
            'overall_accuracy': arithmetic_results['overall_accuracy'],
            'per_task': {
                task_type: arithmetic_results[task_type]
                for task_type in ['add_days', 'sub_days', 'diff_days', 'weekday', 'days_in_month', 'leap_year']
                if task_type in arithmetic_results and isinstance(arithmetic_results[task_type], dict)
            }
        }

        if correlation_analysis is not None:
            results_to_save['correlation_analysis'] = {
                'insights': correlation_analysis.get('insights', []),
            }

    with open(os.path.join(save_dir, 'temporal_geometry_results.json'), 'w') as f:
        json.dump(results_to_save, f, indent=2)

    print(f"\nResults saved to {save_dir}/")

    return results, summary, arithmetic_results


if __name__ == "__main__":

    import argparse

    results, summary, arithmetic_results = main(
        "Qwen/Qwen3-0.6B",
    )
