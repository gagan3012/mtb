import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import cross_val_score
from typing import List, Dict, Tuple
import json
import os
from tqdm import tqdm
import seaborn as sns
from datetime import datetime, timedelta
import random
from mpl_toolkits.mplot3d import Axes3D

# ==========================================
# 1. CONFIGURATION & SETUP
# ==========================================
MODEL_NAME = "Qwen/Qwen3-0.6B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Define the years to analyze
YEARS = list(range(1990, 2025))

# How many date samples per year (more = better disentanglement signal)
SAMPLES_PER_YEAR = 3

# Templates for year-only analysis (original approach)
TEMPLATES = {
    "en": "The year is {}.",
    "de": "Das Jahr ist {}.",
    "zh": "现在是{}年。",
    "ar": "السنة هي {}.",
    "ha": "Shekarar ita ce {}.",
}


def format_date_multilingual(date: datetime, language: str) -> str:
    """
    Format a date in the specified language.
    
    Provides natural date representations that match how dates are
    commonly written in each language.
    """
    year = date.year
    month = date.month
    day = date.day
    weekday = date.weekday()
    
    # Month names
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
        # "fr": ["janvier", "février", "mars", "avril", "mai", "juin",
        #        "juillet", "août", "septembre", "octobre", "novembre", "décembre"],
        # "es": ["enero", "febrero", "marzo", "abril", "mayo", "junio",
        #        "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"],
        # "hi": ["जनवरी", "फ़रवरी", "मार्च", "अप्रैल", "मई", "जून",
        #        "जुलाई", "अगस्त", "सितंबर", "अक्टूबर", "नवंबर", "दिसंबर"],
    }
    
    # Weekday names
    weekdays = {
        "en": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
        "de": ["Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag", "Samstag", "Sonntag"],
        "zh": ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"],
        "ar": ["الاثنين", "الثلاثاء", "الأربعاء", "الخميس", "الجمعة", "السبت", "الأحد"],
        "ha": ["Litinin", "Talata", "Laraba", "Alhamis", "Jumma'a", "Asabar", "Lahadi"],
        # "fr": ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"],
        # "es": ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"],
        # "hi": ["सोमवार", "मंगलवार", "बुधवार", "गुरुवार", "शुक्रवार", "शनिवार", "रविवार"],
    }
    
    # Get localized names
    month_name = months.get(language, months["en"])[month - 1]
    weekday_name = weekdays.get(language, weekdays["en"])[weekday]
    
    # Format according to language conventions
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
    # elif language == "fr":
    #     return f"{weekday_name} {day} {month_name} {year}"
    # elif language == "es":
    #     return f"{weekday_name}, {day} de {month_name} de {year}"
    # elif language == "hi":
    #     return f"{weekday_name}, {day} {month_name} {year}"
    else:
        return f"{year}-{month:02d}-{day:02d}"


def build_calendar_labels_from_years(
    years: List[int],
    languages: List[str],
    samples_per_year: int = 5,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Build calendar labels with real, diverse dates for each language.
    
    Creates multiple date samples per year with varying months, days, and weekdays
    to provide better signal for disentanglement learning.
    
    Args:
        years: List of years to generate dates for
        languages: List of language codes
        samples_per_year: Number of different dates to generate per year
        
    Returns:
        Dict mapping language -> dict of calendar components (year, month, day, weekday, date_string)
    """
    # Set random seed for reproducibility
    random.seed(42)
    
    # Generate diverse dates for each year
    all_dates = []
    date_strings_per_lang = {lang: [] for lang in languages}
    
    for year in years:
        # Generate samples_per_year diverse dates throughout the year
        year_dates = []
        
        # Ensure we get dates from different months and seasons
        for i in range(samples_per_year):
            # Distribute dates evenly across the year
            day_of_year = int((i / samples_per_year) * 365)
            try:
                date = datetime(year, 1, 1) + timedelta(days=day_of_year)
                # Add some randomness (±7 days) to avoid too-regular patterns
                random_offset = random.randint(-7, 7)
                date = date + timedelta(days=random_offset)
                # Clamp to valid year range
                if date.year != year:
                    date = datetime(year, 1, 1) + timedelta(days=day_of_year)
                year_dates.append(date)
            except ValueError:
                # Handle leap year edge cases
                date = datetime(year, 12, 31)
                year_dates.append(date)
        
        all_dates.extend(year_dates)
        
        # Generate multilingual date strings
        for lang in languages:
            for date in year_dates:
                date_strings_per_lang[lang].append(
                    format_date_multilingual(date, lang)
                )
    
    # Extract calendar components
    years_arr = np.array([d.year for d in all_dates], dtype=int)
    months_arr = np.array([d.month for d in all_dates], dtype=int)
    days_arr = np.array([d.day for d in all_dates], dtype=int)
    weekdays_arr = np.array([d.weekday() for d in all_dates], dtype=int)
    
    # Build labels per language
    labels_per_lang = {}
    for lang in languages:
        labels_per_lang[lang] = {
            "year": years_arr.copy(),
            "month": months_arr.copy(),
            "day": days_arr.copy(),
            "weekday": weekdays_arr.copy(),
            "date_strings": date_strings_per_lang[lang],
            "datetime_objects": all_dates.copy(),
        }
    
    return labels_per_lang


def build_multilingual_date_templates(
    years: List[int],
    languages: List[str],
    samples_per_year: int = 5,
) -> Dict[str, List[str]]:
    """
    Build complete date sentences in multiple languages for use with the model.
    
    Returns:
        Dict mapping language -> list of date sentences
    """
    labels = build_calendar_labels_from_years(years, languages, samples_per_year)
    
    templates = {
        "en": "The date is {}.",
        "de": "Das Datum ist {}.",
        "zh": "日期是{}。",
        "ar": "التاريخ هو {}.",
        "ha": "Ranar ita ce {}.",
        "fr": "La date est {}.",
        "es": "La fecha es {}.",
        "hi": "तारीख {} है।",
    }
    
    sentences = {}
    for lang in languages:
        template = templates.get(lang, "The date is {}.")
        date_strings = labels[lang]["date_strings"]
        sentences[lang] = [template.format(ds) for ds in date_strings]
    
    return sentences


# Build comprehensive calendar labels with real dates
languages = list(TEMPLATES.keys())
# This provides both the date_strings AND the year/month/day/weekday labels
# needed for R² computation and disentanglement analysis
CALENDAR_LABELS = build_calendar_labels_from_years(YEARS, languages, SAMPLES_PER_YEAR)

# Note: build_multilingual_date_templates only returns sentences, not the full structure
# If you need just sentences: DATE_SENTENCES = build_multilingual_date_templates(YEARS, languages, SAMPLES_PER_YEAR)


# ==========================================
# 2. MODEL LOADING
# ==========================================
def load_model(model_name: str):
    """Load model and tokenizer with proper configuration."""
    print(f"Loading {model_name} on {DEVICE}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, output_hidden_states=True, device_map="auto"
    )
    model.eval()
    return model, tokenizer


# ==========================================
# 3. TOKEN IDENTIFICATION
# ==========================================
def find_date_token_indices(tokenizer, text: str, year: int) -> List[int]:
    """
    Find the indices of tokens that correspond to the year in the text.
    This handles tokenization fragmentation (DFR).
    """
    year_str = str(year)

    # Tokenize the full text
    tokens = tokenizer(text, return_tensors="pt", return_offsets_mapping=True)
    input_ids = tokens["input_ids"][0]

    # Decode each token to find the year tokens
    year_token_indices = []
    decoded_tokens = [tokenizer.decode([tid]) for tid in input_ids]

    # Find where the year appears in the text
    year_start_pos = text.find(year_str)
    if year_start_pos == -1:
        # Fallback: return last token index
        return [len(input_ids) - 1]

    # Use offset mapping if available, otherwise use string matching
    if "offset_mapping" in tokens and tokens["offset_mapping"] is not None:
        offsets = tokens["offset_mapping"][0]
        for idx, (start, end) in enumerate(offsets):
            if start >= year_start_pos and end <= year_start_pos + len(year_str):
                year_token_indices.append(idx)
    else:
        # Fallback: look for tokens containing year digits
        for idx, decoded in enumerate(decoded_tokens):
            if any(digit in decoded for digit in year_str):
                year_token_indices.append(idx)

    return year_token_indices if year_token_indices else [len(input_ids) - 1]


# ==========================================
# 4. EXTRACTION ENGINE
# ==========================================
def get_date_embedding(
    model,
    tokenizer,
    year: int,
    text_or_template: str,
    target_layer: int,
    aggregation: str = "mean",
    is_template: bool = True,
) -> np.ndarray:
    """
    Extracts the hidden state for a date text.

    Args:
        year: The year value (used for finding year tokens if is_template=True)
        text_or_template: Either a template with {} (is_template=True) or a full date string
        target_layer: Which layer to extract from
        aggregation: How to combine multiple date tokens - "mean", "last", or "first"
        is_template: If True, text_or_template is a template with {} to format with year.
                    If False, text_or_template is already the complete text.
    
    For full date strings (is_template=False), we use the LAST token which captures
    the full context in autoregressive models. This is more reliable across languages.
    """
    if is_template:
        text = text_or_template.format(year)
    else:
        text = text_or_template
    
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)

    # Get hidden states at the target layer
    hidden_states = outputs.hidden_states[target_layer].squeeze(0)
    
    # For full date strings, use last token (captures full context, works across languages)
    # For templates, try to find year tokens
    if not is_template:
        # Use last token for full date strings - most reliable for autoregressive models
        embedding = hidden_states[-1]
    else:
        # Find date token indices for template-based extraction
        date_indices = find_date_token_indices(tokenizer, text, year)

        if aggregation == "mean" and len(date_indices) > 1:
            # Average all date tokens
            date_vectors = hidden_states[date_indices]
            embedding = date_vectors.mean(dim=0)
        elif aggregation == "last":
            embedding = hidden_states[date_indices[-1]]
        else:  # "first" or single token
            embedding = hidden_states[date_indices[0]]

    return embedding.cpu().float().numpy()


def extract_all_embeddings(
    model, tokenizer, years: List[int], templates: Dict[str, str], target_layer: int
) -> Dict[str, np.ndarray]:
    """Extract embeddings for all years across all languages."""
    embeddings = {lang: [] for lang in templates.keys()}

    for lang, tmpl in templates.items():
        for year in years:
            vec = get_date_embedding(
                model, tokenizer, year, tmpl, target_layer, is_template=True
            )
            embeddings[lang].append(vec)

    return {k: np.array(v) for k, v in embeddings.items()}


def extract_all_embeddings_batched(
    model,
    tokenizer,
    calendar_labels: Dict[str, Dict[str, np.ndarray]],
    target_layers: List[int],  # Get multiple layers at once
    batch_size: int = 32,
) -> Dict[int, Dict[str, np.ndarray]]:
    """Extract embeddings for multiple layers in one pass with batching."""
    results = {layer: {lang: [] for lang in calendar_labels} for layer in target_layers}
    
    for lang in calendar_labels:
        labels = calendar_labels[lang]
        texts = [TEMPLATES[lang].format(y) for y in labels["year"]]
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            inputs = tokenizer(batch_texts, return_tensors="pt", padding=True).to(DEVICE)
            
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Extract from all target layers at once
            for layer_idx in target_layers:
                hidden = outputs.hidden_states[layer_idx]
                # Take last token for each sequence
                batch_embeds = hidden[:, -1, :].cpu().float().numpy()
                results[layer_idx][lang].extend(batch_embeds)
    
    return {layer: {k: np.array(v) for k, v in langs.items()} 
            for layer, langs in results.items()}


def extract_all_embeddings_with_dates(
    model,
    tokenizer,
    calendar_labels: Dict[str, Dict[str, np.ndarray]],
    target_layer: int,
    use_full_dates: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Extract embeddings using either year-only or full date strings.
    
    Args:
        use_full_dates: If True, use full multilingual date strings (last token extraction).
                       If False, use simple year templates (year token extraction).
    """
    embeddings = {lang: [] for lang in calendar_labels.keys()}
    
    for lang in calendar_labels.keys():
        labels = calendar_labels[lang]
        n_samples = len(labels["year"])
        
        for i in range(n_samples):
            if use_full_dates:
                # Use full date string - extract last token for full context
                text = labels["date_strings"][i]
                year = labels["year"][i]
                vec = get_date_embedding(
                    model, tokenizer, year, text, target_layer, is_template=False
                )
            else:
                # Use simple year template (original behavior) - extract year tokens
                year = labels["year"][i]
                text = TEMPLATES[lang].format(year)
                vec = get_date_embedding(
                    model, tokenizer, year, text, target_layer, is_template=True
                )
            
            embeddings[lang].append(vec)
    
    return {k: np.array(v) for k, v in embeddings.items()}


# ==========================================
# 5. ANALYSIS FUNCTIONS
# ==========================================
def compute_time_vectors(embeddings: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Compute the average 'time direction' vector for each language."""
    time_vectors = {}
    for lang in embeddings:
        diffs = embeddings[lang][1:] - embeddings[lang][:-1]
        time_vectors[lang] = np.mean(diffs, axis=0)
    return time_vectors


def compute_cross_language_similarity(
    time_vectors: Dict[str, np.ndarray],
) -> Dict[Tuple[str, str], float]:
    """Compute pairwise cosine similarity between all language time vectors."""
    similarities = {}
    langs = list(time_vectors.keys())

    for i, lang1 in enumerate(langs):
        for lang2 in langs[i + 1 :]:
            sim = cosine_similarity(
                time_vectors[lang1].reshape(1, -1), time_vectors[lang2].reshape(1, -1)
            )[0][0]
            similarities[(lang1, lang2)] = sim

    return similarities


def compute_temporal_linearity(embeddings: np.ndarray, years: List[int]) -> float:
    """
    Measure how linear the temporal progression is.
    Uses R² score of linear regression from year to embedding.
    """
    years_normalized = np.array(years).reshape(-1, 1)

    # Fit linear regression and measure R²
    reg = LinearRegression()
    reg.fit(years_normalized, embeddings)
    r2 = reg.score(years_normalized, embeddings)

    return r2


def compute_year_separability(embeddings: np.ndarray) -> float:
    """
    Measure how well-separated consecutive years are.
    Returns average cosine distance between consecutive years.
    """
    distances = []
    for i in range(len(embeddings) - 1):
        sim = cosine_similarity(
            embeddings[i].reshape(1, -1), embeddings[i + 1].reshape(1, -1)
        )[0][0]
        distances.append(1 - sim)  # Convert similarity to distance
    return np.mean(distances)


# ===== CALENDAR DISENTANGLEMENT UTILITIES =====


def compute_calendar_component_vectors(
    embeddings: np.ndarray, component_labels: Dict[str, np.ndarray]
) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """
    For a single language, fit linear probes from embeddings -> each calendar component.

    component_labels: dict with keys like 'year', 'month', 'day', 'weekday',
                      each value shape (N,)
    Returns: 
        - dict component -> weight vector (dim,)
        - dict component -> R² score (how well linear structure is preserved)
    
    IMPORTANT: If R² is low for a component (e.g., month, day), the extracted
    direction vector is unreliable/distorted, and disentanglement metrics
    computed from it should be interpreted with caution.
    
    NOTE: We use Ridge regression with cross-validation to compute R², 
    avoiding overfitting in high-dimensional spaces (dim >> n_samples).
    We use ShuffleSplit to avoid the time-series extrapolation problem
    (standard KFold on sequential years causes negative R² for year).
    """
    from sklearn.model_selection import ShuffleSplit
    
    component_vectors = {}
    component_r2_scores = {}
    
    for comp, y in component_labels.items():
        if y is None or comp == "date_strings" or comp == "datetime_objects":
            continue
        y = np.asarray(y).ravel()  # Use 1D array for cross_val_score
        if y.shape[0] != embeddings.shape[0]:
            raise ValueError(
                f"Label length for component '{comp}' "
                f"({y.shape[0]}) does not match embeddings ({embeddings.shape[0]})"
            )
        
        # Use Ridge for direction vector (regularization prevents overfitting)
        reg = Ridge(alpha=1.0)
        reg.fit(embeddings, y)
        component_vectors[comp] = reg.coef_.copy()
        
        # Use cross-validated R² with SHUFFLED splits to get reliable score
        # ShuffleSplit avoids the time-series extrapolation problem:
        # - Standard KFold on sequential years causes negative R² because
        #   training on years 2000-2024 and testing on 1990-1999 requires extrapolation
        # - ShuffleSplit ensures each fold has a mix of all time periods
        try:
            # Use ShuffleSplit for random train/test splits
            cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
            cv_scores = cross_val_score(
                Ridge(alpha=1.0), embeddings, y,
                cv=cv,
                scoring='r2'
            )
            component_r2_scores[comp] = float(np.mean(cv_scores))
        except Exception:
            # Fallback to training score if CV fails
            component_r2_scores[comp] = reg.score(embeddings, y)
    
    return component_vectors, component_r2_scores


def compute_calendar_disentanglement(component_vectors: Dict[str, np.ndarray]) -> float:
    """
    Measure how orthogonal the calendar component directions are.

    Returns a score in [0,1], where 1 means perfectly orthogonal,
    0 means all directions identical.
    """
    comps = list(component_vectors.keys())
    if len(comps) < 2:
        return 0.0

    sims = []
    for i in range(len(comps)):
        v1 = component_vectors[comps[i]]
        v1 = v1 / (np.linalg.norm(v1) + 1e-8)
        for j in range(i + 1, len(comps)):
            v2 = component_vectors[comps[j]]
            v2 = v2 / (np.linalg.norm(v2) + 1e-8)
            sims.append(abs(np.dot(v1, v2)))
    if not sims:
        return 0.0
    return float(1.0 - np.mean(sims))


def compute_parallelepiped_volume(
    component_vectors: Dict[str, np.ndarray],
    components: List[str] = None
) -> float:
    """
    Compute the volume of a parallelepiped spanned by calendar component direction vectors.
    
    This is a geometric measure of calendar disentanglement using the Gram determinant:
    - X = [v_y, v_m, v_d], each vector is normalized to unit length
    - G = X^T X is the Gram matrix (off-diagonal = cosine similarities, diagonal = 1)
    - Volume m = sqrt(det(G))
    
    Interpretation:
    - m = 0: The parallelepiped collapses to a plane or line; directions are linearly dependent
             (at least two components are treated as the same concept by the LLM)
    - m = 1: Unit cube; all three directions are mutually orthogonal
             (components are viewed as fully independent concepts by the LLM)
    - Higher m means more independent/disentangled representations
    
    Reference: https://math.stackexchange.com/questions/981238/
    
    Args:
        component_vectors: Dict mapping component names to direction vectors
        components: List of component names to use (default: ['year', 'month', 'day'])
        
    Returns:
        Volume in [0, 1], where 1 = perfectly orthogonal (unit cube)
    """
    if components is None:
        components = ['year', 'month', 'day']
    
    # Filter to only available components
    available = [c for c in components if c in component_vectors]
    
    if len(available) < 2:
        return 0.0
    
    # Build matrix X where each column is a normalized direction vector
    vectors = []
    for comp in available:
        v = component_vectors[comp]
        v_norm = v / (np.linalg.norm(v) + 1e-8)
        vectors.append(v_norm)
    
    # Stack as columns: X has shape (dim, n_components)
    X = np.column_stack(vectors)
    
    # Compute Gram matrix G = X^T X
    # G[i,j] = v_i · v_j (cosine similarity for normalized vectors)
    # Diagonal entries are 1 (unit vectors)
    G = X.T @ X
    
    # Volume = sqrt(det(G))
    # Use absolute value to handle numerical issues
    det_G = np.linalg.det(G)
    volume = np.sqrt(max(0.0, det_G))  # Clamp to avoid sqrt of negative due to numerics
    
    return float(volume)


def compute_pairwise_cosine_from_gram(
    component_vectors: Dict[str, np.ndarray],
    components: List[str] = None
) -> Dict[Tuple[str, str], float]:
    """
    Compute pairwise cosine similarities between calendar component directions.
    These are exactly the off-diagonal elements of the Gram matrix.
    
    Args:
        component_vectors: Dict mapping component names to direction vectors
        components: List of component names to use (default: ['year', 'month', 'day'])
        
    Returns:
        Dict mapping (comp1, comp2) tuples to cosine similarity values
    """
    if components is None:
        components = ['year', 'month', 'day']
    
    available = [c for c in components if c in component_vectors]
    pairwise = {}
    
    for i, c1 in enumerate(available):
        v1 = component_vectors[c1]
        v1_norm = v1 / (np.linalg.norm(v1) + 1e-8)
        for j, c2 in enumerate(available):
            if i < j:
                v2 = component_vectors[c2]
                v2_norm = v2 / (np.linalg.norm(v2) + 1e-8)
                pairwise[(c1, c2)] = float(np.dot(v1_norm, v2_norm))
    
    return pairwise


def compute_cross_language_calendar_alignment(
    calendar_vectors_per_lang: Dict[str, Dict[str, np.ndarray]],
) -> Dict[str, float]:
    """
    For each calendar component (year, month, day, weekday), compute
    average cosine similarity of that component's direction across languages.
    """
    if not calendar_vectors_per_lang:
        return {}

    langs = list(calendar_vectors_per_lang.keys())
    components = list(next(iter(calendar_vectors_per_lang.values())).keys())
    alignment = {c: [] for c in components}

    for c in components:
        for i in range(len(langs)):
            for j in range(i + 1, len(langs)):
                v1 = calendar_vectors_per_lang[langs[i]].get(c)
                v2 = calendar_vectors_per_lang[langs[j]].get(c)
                if v1 is None or v2 is None:
                    continue
                sim = cosine_similarity(v1.reshape(1, -1), v2.reshape(1, -1))[0][0]
                alignment[c].append(sim)

    return {
        c: float(np.mean(vals)) if len(vals) > 0 else 0.0
        for c, vals in alignment.items()
    }


def compute_component_prediction_accuracy(
    embeddings: np.ndarray,
    component_labels: Dict[str, np.ndarray],
    n_splits: int = 5
) -> Dict[str, float]:
    """
    Measure prediction accuracy for each calendar component using cross-validation.
    """
    accuracies = {}
    for comp, labels in component_labels.items():
        if labels is None or len(labels) == 0 or comp in ["date_strings", "datetime_objects"]:
            continue
        reg = Ridge()
        try:
            scores = cross_val_score(
                reg, embeddings, labels, 
                cv=min(n_splits, len(labels)), 
                scoring='r2'
            )
            accuracies[comp] = float(scores.mean())
        except Exception as e:
            print(f"Warning: Could not compute accuracy for {comp}: {e}")
            accuracies[comp] = 0.0
    
    return accuracies


# ==========================================
# 5b. COMPREHENSIVE PER-LAYER ANALYSIS
# ==========================================
def analyze_layer(
    model,
    tokenizer,
    years: List[int],
    templates: Dict[str, str],
    layer: int,
    calendar_labels: Dict[str, Dict[str, np.ndarray]] = None,
    use_full_dates: bool = True,  # Must be True to get meaningful month/day R²
) -> Dict:
    """Comprehensive analysis for a single layer.
    
    Args:
        use_full_dates: If True, use full date strings (e.g., "Monday, January 15, 1990")
                       which enables meaningful R² for month/day/weekday.
                       If False, uses year-only templates (e.g., "The year is 1990.")
                       which means month/day/weekday R² will be meaningless noise!
    """
    # Extract embeddings - use the calendar_labels if available to handle multiple samples per year
    if calendar_labels is not None and SAMPLES_PER_YEAR > 1:
        # Use the enhanced extraction with real dates
        # NOTE: use_full_dates MUST be True for month/day/weekday R² to be meaningful!
        # If False, the embeddings only contain year info, so month/day R² is random noise.
        embeddings = extract_all_embeddings_with_dates(
            model, tokenizer, calendar_labels, layer, use_full_dates=use_full_dates
        )
        # For time vectors, we need to group by year
        years_expanded = []
        for y in years:
            years_expanded.extend([y] * SAMPLES_PER_YEAR)
    else:
        embeddings = extract_all_embeddings(model, tokenizer, years, templates, layer)
        years_expanded = years
    
    time_vectors = compute_time_vectors(embeddings)
    cross_lang_sim = compute_cross_language_similarity(time_vectors)

    # Compute per-language metrics
    linearity = {
        lang: compute_temporal_linearity(emb, years_expanded) for lang, emb in embeddings.items()
    }
    separability = {
        lang: compute_year_separability(emb) for lang, emb in embeddings.items()
    }

    # Calendar component vectors & disentanglement
    calendar_vectors = {}
    calendar_disent = {}
    parallelepiped_volume = {}  # New geometric metric
    parallelepiped_volume_centered = {}  # Gram volume with centered embeddings
    pairwise_cosines = {}  # Pairwise cosine similarities from Gram matrix
    component_accuracy = {}
    component_linearity_r2 = {}  # R² scores for each component's linear probe
    
    if calendar_labels is not None:
        for lang, emb in embeddings.items():
            if lang in calendar_labels:
                comp_vecs, comp_r2 = compute_calendar_component_vectors(
                    emb, calendar_labels[lang]
                )
                calendar_vectors[lang] = comp_vecs
                component_linearity_r2[lang] = comp_r2  # Store R² for each component
                calendar_disent[lang] = compute_calendar_disentanglement(comp_vecs)
                # Compute geometric parallelepiped volume (Gram determinant)
                parallelepiped_volume[lang] = compute_parallelepiped_volume(comp_vecs)
                pairwise_cosines[lang] = compute_pairwise_cosine_from_gram(comp_vecs)
                component_accuracy[lang] = compute_component_prediction_accuracy(
                    emb, calendar_labels[lang]
                )
                
                # Centered Gram volume: center embeddings before fitting probes
                # This removes the global mean offset, so direction vectors
                # reflect variation around the centroid rather than absolute position.
                # Only applied to Gram volume computation (not R² or other metrics).
                emb_centered = emb - emb.mean(axis=0)
                comp_vecs_centered, _ = compute_calendar_component_vectors(
                    emb_centered, calendar_labels[lang]
                )
                parallelepiped_volume_centered[lang] = compute_parallelepiped_volume(comp_vecs_centered)

    calendar_alignment = {}
    avg_calendar_disent = None
    avg_parallelepiped_volume = None
    avg_parallelepiped_volume_centered = None
    if calendar_vectors:
        calendar_alignment = compute_cross_language_calendar_alignment(calendar_vectors)
        if calendar_disent:
            avg_calendar_disent = float(np.mean(list(calendar_disent.values())))
        if parallelepiped_volume:
            avg_parallelepiped_volume = float(np.mean(list(parallelepiped_volume.values())))
        if parallelepiped_volume_centered:
            avg_parallelepiped_volume_centered = float(np.mean(list(parallelepiped_volume_centered.values())))

    # Average cross-language similarity
    avg_cross_sim = np.mean(list(cross_lang_sim.values()))

    # Compute average R² for each component across languages
    avg_component_r2 = {}
    if component_linearity_r2:
        all_comps = set()
        for lang_r2 in component_linearity_r2.values():
            all_comps.update(lang_r2.keys())
        for comp in all_comps:
            vals = [component_linearity_r2[lang].get(comp, 0.0) 
                    for lang in component_linearity_r2 if comp in component_linearity_r2[lang]]
            avg_component_r2[comp] = float(np.mean(vals)) if vals else 0.0

    return {
        "embeddings": embeddings,
        "time_vectors": time_vectors,
        "cross_language_similarity": cross_lang_sim,
        "avg_cross_language_similarity": avg_cross_sim,
        "temporal_linearity": linearity,
        "year_separability": separability,
        "calendar_component_vectors": calendar_vectors,
        "calendar_disentanglement": calendar_disent,
        "avg_calendar_disentanglement": avg_calendar_disent,
        "parallelepiped_volume": parallelepiped_volume,  # Gram determinant volume
        "avg_parallelepiped_volume": avg_parallelepiped_volume,
        "parallelepiped_volume_centered": parallelepiped_volume_centered,  # Centered Gram volume
        "avg_parallelepiped_volume_centered": avg_parallelepiped_volume_centered,
        "pairwise_cosines": pairwise_cosines,  # Off-diagonal Gram matrix elements
        "calendar_alignment": calendar_alignment,
        "component_accuracy": component_accuracy,
        "component_linearity_r2": component_linearity_r2,  # R² for each component's linear probe
        "avg_component_linearity_r2": avg_component_r2,  # Avg R² across languages per component
    }


# ==========================================
# 6. VISUALIZATION FUNCTIONS
# ==========================================

def plot_calendar_coordinate_system_3d(
    calendar_vectors: Dict[str, Dict[str, np.ndarray]],
    embeddings: Dict[str, np.ndarray],
    calendar_labels: Dict[str, Dict[str, np.ndarray]],
    layer: int,
    save_path: str = None
):
    """
    Visualize embeddings projected onto the learned calendar component axes.
    Shows how well the model separates year/month/day as orthogonal dimensions.
    """
    fig = plt.figure(figsize=(15, 5))
    
    for idx, (lang, comp_vecs) in enumerate(calendar_vectors.items()):
        ax = fig.add_subplot(1, len(calendar_vectors), idx + 1, projection='3d')
        
        # Get the component vectors and normalize
        year_vec = comp_vecs['year'] / (np.linalg.norm(comp_vecs['year']) + 1e-8)
        month_vec = comp_vecs['month'] / (np.linalg.norm(comp_vecs['month']) + 1e-8)
        day_vec = comp_vecs['day'] / (np.linalg.norm(comp_vecs['day']) + 1e-8)
        
        # Project embeddings onto these axes
        emb = embeddings[lang]
        coords_year = emb @ year_vec
        coords_month = emb @ month_vec
        coords_day = emb @ day_vec
        
        # Color by actual year
        years_data = calendar_labels[lang]["year"]
        colors = plt.cm.viridis((years_data - years_data.min()) / (years_data.max() - years_data.min()))
        
        ax.scatter(coords_year, coords_month, coords_day, c=colors, s=50, alpha=0.6)
        
        # Draw coordinate axes
        max_range = max(abs(coords_year).max(), abs(coords_month).max(), abs(coords_day).max())
        ax.quiver(0, 0, 0, max_range, 0, 0, color='r', alpha=0.3, linewidth=2, label='Year')
        ax.quiver(0, 0, 0, 0, max_range, 0, color='g', alpha=0.3, linewidth=2, label='Month')
        ax.quiver(0, 0, 0, 0, 0, max_range, color='b', alpha=0.3, linewidth=2, label='Day')
        
        ax.set_xlabel('Year Component')
        ax.set_ylabel('Month Component')
        ax.set_zlabel('Day Component')
        ax.set_title(f'{lang.upper()} - Layer {layer}')
        ax.legend()
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved 3D coordinate plot: {save_path}")
    plt.close()


def plot_component_dimension_contributions(
    calendar_vectors: Dict[str, Dict[str, np.ndarray]],
    layer: int,
    top_k: int = 20,
    save_path: str = None
):
    """
    Visualize which embedding dimensions are most important for each calendar component.
    """
    n_langs = len(calendar_vectors)
    fig, axes = plt.subplots(n_langs, 1, figsize=(14, 4 * n_langs))
    if n_langs == 1:
        axes = [axes]
    
    for idx, (lang, comp_vecs) in enumerate(calendar_vectors.items()):
        components = ['year', 'month', 'day', 'weekday']
        dim_size = comp_vecs['year'].shape[0]
        
        # Get top-k dimensions for each component
        contributions = np.zeros((len(components), dim_size))
        for i, comp in enumerate(components):
            contributions[i] = np.abs(comp_vecs[comp])

        actual_top_k = min(int(top_k), dim_size)
        
        # Plot heatmap of top dimensions
        top_dims = np.argsort(contributions.sum(axis=0))[-actual_top_k:]
        
        sns.heatmap(
            contributions[:, top_dims],
            xticklabels=top_dims,
            yticklabels=[c.capitalize() for c in components],
            cmap='YlOrRd',
            ax=axes[idx],
            cbar_kws={'label': 'Absolute Weight'}
        )
        axes[idx].set_title(f'{lang.upper()} - Top {top_k} Dimensions per Component (Layer {layer})')
        axes[idx].set_xlabel('Embedding Dimension')
        axes[idx].set_ylabel('Calendar Component')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved dimension contribution plot: {save_path}")
    plt.close()


def plot_temporal_trajectories_in_component_space(
    calendar_vectors: Dict[str, Dict[str, np.ndarray]],
    embeddings: Dict[str, np.ndarray],
    calendar_labels: Dict[str, Dict[str, np.ndarray]],
    layer: int,
    save_path: str = None
):
    """
    Plot 2D projections showing temporal progression in component pairs.
    """
    component_pairs = [('year', 'month'), ('year', 'day'), ('month', 'day')]
    n_langs = len(calendar_vectors)
    
    fig, axes = plt.subplots(n_langs, 3, figsize=(15, 5 * n_langs))
    if n_langs == 1:
        axes = axes.reshape(1, -1)
    
    for lang_idx, (lang, comp_vecs) in enumerate(calendar_vectors.items()):
        emb = embeddings[lang]
        labels = calendar_labels[lang]
        
        for pair_idx, (comp1, comp2) in enumerate(component_pairs):
            ax = axes[lang_idx, pair_idx]
            
            # Project onto component pair
            v1 = comp_vecs[comp1] / (np.linalg.norm(comp_vecs[comp1]) + 1e-8)
            v2 = comp_vecs[comp2] / (np.linalg.norm(comp_vecs[comp2]) + 1e-8)
            
            coords1 = emb @ v1
            coords2 = emb @ v2
            
            # Color by the first component's actual value
            scatter = ax.scatter(
                coords1, coords2,
                c=labels[comp1],
                cmap='viridis',
                s=50,
                alpha=0.6
            )
            
            # Draw trajectory (only for temporal ordering)
            if comp1 == 'year' or comp2 == 'year':
                ax.plot(coords1, coords2, 'k-', alpha=0.2, linewidth=1)
            
            ax.set_xlabel(f'{comp1.capitalize()} Component')
            ax.set_ylabel(f'{comp2.capitalize()} Component')
            ax.set_title(f'{lang.upper()}: {comp1.capitalize()} vs {comp2.capitalize()}')
            plt.colorbar(scatter, ax=ax, label=f'Actual {comp1.capitalize()}')
            ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Temporal Trajectories in Component Space (Layer {layer})', 
                 fontsize=16, y=1.00)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved trajectory plot: {save_path}")
    plt.close()


def plot_component_prediction_accuracy(
    layer_results: List[Dict],
    save_path: str = None
):
    """Plot prediction accuracy for each component across layers."""
    # Extract accuracy data
    component_data = {}
    layers = list(range(len(layer_results)))
    
    for layer_idx, result in enumerate(layer_results):
        # Average across languages
        if 'component_accuracy' in result and result['component_accuracy']:
            for lang, acc_dict in result['component_accuracy'].items():
                for comp, acc in acc_dict.items():
                    if comp not in component_data:
                        component_data[comp] = []
                    if len(component_data[comp]) <= layer_idx:
                        component_data[comp].append([])
                    component_data[comp][layer_idx].append(acc)
    
    if not component_data:
        print("No component accuracy data available to plot.")
        return
    
    # Average across languages
    avg_component_data = {
        comp: [np.mean(vals) if vals else 0.0 for vals in layer_vals]
        for comp, layer_vals in component_data.items()
    }
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 7))
    
    component_colors = {
        "year": "#1f77b4",
        "month": "#ff7f0e",
        "day": "#2ca02c",
        "weekday": "#d62728",
    }
    
    for component, scores in avg_component_data.items():
        color = component_colors.get(component, "gray")
        ax.plot(
            layers[:len(scores)],
            scores,
            "-o",
            label=component.capitalize(),
            linewidth=2.5,
            markersize=6,
            color=color,
            alpha=0.8,
        )
    
    ax.set_title(
        "Calendar Component Prediction Accuracy (R²)\n"
        "(Higher = Better Decodability)",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("R² Score", fontsize=12)
    ax.set_ylim([-0.05, 1.05])
    ax.axhline(y=0.0, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)
    ax.legend(loc="best", fontsize=11, frameon=True, shadow=True)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved component accuracy plot: {save_path}")
    plt.close()


def plot_component_linearity_r2(
    layer_results: List[Dict],
    save_path: str = None
):
    """
    Plot R² scores for each calendar component's linear probe across layers.
    
    This is CRITICAL for validating disentanglement metrics:
    - If R² for month/day is low, their direction vectors are unreliable
    - The parallelepiped volume becomes meaningless if component directions are distorted
    - A component with R² < 0.1 should not be trusted for disentanglement analysis
    """
    component_data = {}
    layers = list(range(len(layer_results)))
    
    for layer_idx, result in enumerate(layer_results):
        avg_r2 = result.get("avg_component_linearity_r2", {})
        for comp, r2 in avg_r2.items():
            if comp not in component_data:
                component_data[comp] = []
            component_data[comp].append(r2)
    
    if not component_data:
        print("No component linearity R² data available to plot.")
        return
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    component_colors = {
        "year": "#1f77b4",
        "month": "#ff7f0e",
        "day": "#2ca02c",
        "weekday": "#d62728",
    }
    
    for component, r2_scores in component_data.items():
        color = component_colors.get(component, "gray")
        ax.plot(
            layers[:len(r2_scores)],
            r2_scores,
            "-o",
            label=component.capitalize(),
            linewidth=2.5,
            markersize=6,
            color=color,
            alpha=0.8,
        )
    
    # Add danger zone for unreliable directions
    ax.axhspan(0, 0.1, alpha=0.15, color='red', label='Unreliable (R²<0.1)')
    ax.axhspan(0.1, 0.3, alpha=0.1, color='orange', label='Weak (0.1<R²<0.3)')
    
    ax.set_title(
        "Component Direction Linearity (R² of Linear Probe)\n"
        "⚠️ Low R² = Distorted Direction Vector = Unreliable Disentanglement Metrics",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("R² Score", fontsize=12)
    ax.set_ylim([-0.05, 1.05])
    ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)
    ax.legend(loc="best", fontsize=10, frameon=True, shadow=True)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved component linearity R² plot: {save_path}")
    plt.close()


def plot_calendar_component_alignment(layer_results: List[Dict], save_path: str = None):
    """
    Plot cross-language alignment for each calendar component across layers.
    """
    component_data = {}
    layers = list(range(len(layer_results)))

    for layer_idx, result in enumerate(layer_results):
        alignment = result.get("calendar_alignment", {})
        for component, score in alignment.items():
            if component not in component_data:
                component_data[component] = []
            component_data[component].append(score)

    if not component_data:
        print("No calendar alignment data available to plot.")
        return

    fig, ax = plt.subplots(figsize=(12, 7))

    component_colors = {
        "year": "#1f77b4",
        "month": "#ff7f0e",
        "day": "#2ca02c",
        "weekday": "#d62728",
    }

    for component, scores in component_data.items():
        color = component_colors.get(component, "gray")
        ax.plot(
            layers[: len(scores)],
            scores,
            "-o",
            label=component.capitalize(),
            linewidth=2.5,
            markersize=6,
            color=color,
            alpha=0.8,
        )

    ax.set_title(
        "Cross-Language Alignment of Calendar Components\n"
        "(Higher = More Consistent Across Languages)",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Average Cosine Similarity", fontsize=12)
    ax.set_ylim([0, 1.05])
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)
    ax.legend(loc="best", fontsize=11, frameon=True, shadow=True)

    for component, scores in component_data.items():
        if scores:
            last_layer = len(scores) - 1
            last_score = scores[-1]
            ax.annotate(
                f"{last_score:.3f}",
                xy=(last_layer, last_score),
                xytext=(5, 0),
                textcoords="offset points",
                fontsize=9,
                alpha=0.7,
            )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved calendar component alignment plot: {save_path}")
    plt.close()


def plot_calendar_component_heatmap(layer_results: List[Dict], save_path: str = None):
    """
    Create a heatmap showing calendar component alignment across all layers.
    """
    component_data = {}
    num_layers = len(layer_results)

    for layer_idx, result in enumerate(layer_results):
        alignment = result.get("calendar_alignment", {})
        for component, score in alignment.items():
            if component not in component_data:
                component_data[component] = [0.0] * num_layers
            component_data[component][layer_idx] = score

    if not component_data:
        print("No calendar alignment data available for heatmap.")
        return

    components = sorted(component_data.keys())
    matrix = np.array([component_data[comp] for comp in components])

    fig, ax = plt.subplots(figsize=(max(10, num_layers * 0.4), 6))

    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(np.arange(num_layers))
    ax.set_yticks(np.arange(len(components)))
    ax.set_xticklabels(np.arange(num_layers))
    ax.set_yticklabels([c.capitalize() for c in components])

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Cross-Language Alignment", rotation=270, labelpad=20)

    for i in range(len(components)):
        for j in range(num_layers):
            text = ax.text(
                j,
                i,
                f"{matrix[i, j]:.2f}",
                ha="center",
                va="center",
                color="black" if matrix[i, j] > 0.5 else "white",
                fontsize=8,
            )

    ax.set_title(
        "Calendar Component Cross-Language Alignment Heatmap",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Calendar Component", fontsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved calendar component heatmap: {save_path}")
    plt.close()


def plot_temporal_geometry(
    embeddings: Dict[str, np.ndarray],
    calendar_labels: Dict[str, Dict[str, np.ndarray]],
    layer: int,
    save_path: str = None,
):
    """Create PCA visualization of temporal embeddings."""
    color_map = {
        "en": "#1f77b4",
        "de": "#2ca02c",
        "zh": "#d62728",
        "ar": "#ff7f0e",
        "ha": "#9467bd",
        "fr": "#8c564b",
        "es": "#e377c2",
        "hi": "#7f7f7f",
    }

    all_vectors = []
    colors = []
    years_list = []

    for lang in embeddings:
        all_vectors.extend(embeddings[lang])
        years_data = calendar_labels[lang]["year"]
        years_list.extend(years_data)
        colors.extend([color_map.get(lang, "gray")] * len(years_data))

    if len(all_vectors) < 2:
        return

    pca = PCA(n_components=2)
    reduced = pca.fit_transform(all_vectors)

    fig, ax = plt.subplots(figsize=(12, 10))

    cursor = 0
    for lang in embeddings:
        n = len(embeddings[lang])
        lang_data = reduced[cursor : cursor + n]
        lang_years = years_list[cursor : cursor + n]

        ax.plot(
            lang_data[:, 0],
            lang_data[:, 1],
            color=color_map.get(lang, "gray"),
            alpha=0.6,
            linewidth=2,
        )

        ax.scatter(
            lang_data[:, 0],
            lang_data[:, 1],
            c=[color_map.get(lang, "gray")],
            alpha=0.8,
            s=50,
            label=lang.upper(),
        )

        # Annotate first and last unique years
        unique_years = sorted(set(lang_years))
        if unique_years:
            first_year_idx = lang_years.index(unique_years[0])
            last_year_idx = len(lang_years) - 1 - lang_years[::-1].index(unique_years[-1])
            ax.annotate(str(unique_years[0]), lang_data[first_year_idx], fontsize=8, alpha=0.7)
            ax.annotate(str(unique_years[-1]), lang_data[last_year_idx], fontsize=8, alpha=0.7)

        cursor += n

    ax.set_title(f"Geometry of Time Across Languages (Layer {layer})", fontsize=14)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
    ax.legend(loc="best", frameon=True)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot: {save_path}")
    plt.close()


def plot_layer_progression(
    layer_results: List[Dict], languages: List[str], save_path: str = None
):
    """Plot how metrics change across layers."""

    has_calendar = any(
        r.get("avg_calendar_disentanglement") is not None for r in layer_results
    )

    n_cols = 4 if has_calendar else 3
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))

    num_layers = len(layer_results)
    layers = list(range(num_layers))

    # Plot 1: Cross-language similarity
    avg_sims = [r["avg_cross_language_similarity"] for r in layer_results]
    axes[0].plot(layers, avg_sims, "b-o", linewidth=2)
    axes[0].set_title("Cross-Language Time Vector Similarity")
    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel("Average Cosine Similarity")
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Temporal linearity by language
    for lang in languages:
        linearity = [r["temporal_linearity"].get(lang, 0) for r in layer_results]
        axes[1].plot(layers, linearity, "-o", label=lang.upper(), linewidth=1.5)
    axes[1].set_title("Temporal Linearity (R²)")
    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("R² Score")
    axes[1].legend(loc="best", fontsize=8)
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Year separability by language
    for lang in languages:
        sep = [r["year_separability"].get(lang, 0) for r in layer_results]
        axes[2].plot(layers, sep, "-o", label=lang.upper(), linewidth=1.5)
    axes[2].set_title("Year Separability")
    axes[2].set_xlabel("Layer")
    axes[2].set_ylabel("Avg Cosine Distance")
    axes[2].legend(loc="best", fontsize=8)
    axes[2].grid(True, alpha=0.3)

    # Plot 4: Calendar disentanglement (if available)
    if has_calendar:
        # Plot both old metric and new parallelepiped volume
        disent = [
            r["avg_calendar_disentanglement"]
            if r["avg_calendar_disentanglement"] is not None
            else 0.0
            for r in layer_results
        ]
        volume = [
            r.get("avg_parallelepiped_volume", 0.0)
            if r.get("avg_parallelepiped_volume") is not None
            else 0.0
            for r in layer_results
        ]
        axes[3].plot(layers, disent, "-o", linewidth=2, label="1 - avg|cos|")
        axes[3].plot(layers, volume, "-s", linewidth=2, color="green", label="Gram Volume")
        axes[3].set_title("Calendar Disentanglement\n(Parallelepiped Volume)")
        axes[3].set_xlabel("Layer")
        axes[3].set_ylabel("Score (1 = orthogonal/unit cube)")
        axes[3].legend(loc="best", fontsize=8)
        axes[3].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot: {save_path}")
    plt.close()


def plot_similarity_heatmap(
    cross_lang_sim: Dict[Tuple[str, str], float],
    languages: List[str],
    layer: int,
    save_path: str = None,
):
    """Create heatmap of cross-language similarity."""
    n = len(languages)
    sim_matrix = np.eye(n)

    for i, lang1 in enumerate(languages):
        for j, lang2 in enumerate(languages):
            if i < j:
                key = (
                    (lang1, lang2)
                    if (lang1, lang2) in cross_lang_sim
                    else (lang2, lang1)
                )
                if key in cross_lang_sim:
                    sim_matrix[i, j] = cross_lang_sim[key]
                    sim_matrix[j, i] = cross_lang_sim[key]

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        sim_matrix,
        xticklabels=[l.upper() for l in languages],
        yticklabels=[l.upper() for l in languages],
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
    )
    plt.title(f"Cross-Language Time Vector Similarity (Layer {layer})")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot: {save_path}")
    plt.close()


# ==========================================
# 7. MAIN EXECUTION
# ==========================================
def main(model_name):
    """Main execution function."""
    safe_model_name = model_name.replace("/", "_").replace("-", "_")
    os.makedirs(f"results/temporal_geometry/{safe_model_name}", exist_ok=True)

    model, tokenizer = load_model(model_name)
    num_layers = model.config.num_hidden_layers if hasattr(model.config, 'num_hidden_layers') else model.config.text_config.num_hidden_layers
    languages = list(TEMPLATES.keys())

    print(f"Model has {num_layers} layers")
    print(f"Analyzing {len(languages)} languages: {languages}")
    print(f"Analyzing years: {YEARS[0]} to {YEARS[-1]}")
    print(f"Samples per year: {SAMPLES_PER_YEAR}")
    print(f"Total samples: {len(YEARS) * SAMPLES_PER_YEAR}")

    layer_results = []

    for layer in tqdm(range(num_layers), desc="Processing layers"):
        print(f"\n--- Layer {layer} ---")
        results = analyze_layer(
            model, tokenizer, YEARS, TEMPLATES, layer, calendar_labels=CALENDAR_LABELS
        )
        layer_results.append(results)

        # Print cross-language similarities
        print("Cross-language time vector similarities:")
        for (l1, l2), sim in results["cross_language_similarity"].items():
            print(f"  {l1.upper()}-{l2.upper()}: {sim:.4f}")

        # Print calendar alignment
        if results["calendar_alignment"]:
            print("Calendar component alignment:")
            for comp, val in results["calendar_alignment"].items():
                print(f"  {comp}: {val:.4f}")
        
        # Print parallelepiped volume (Gram determinant)
        if results.get("avg_parallelepiped_volume") is not None:
            print(f"Parallelepiped volume (Gram det): {results['avg_parallelepiped_volume']:.4f}")
            print("  (0 = collapsed, 1 = unit cube / fully orthogonal)")
            # Print per-language volumes
            for lang, vol in results.get("parallelepiped_volume", {}).items():
                print(f"    {lang.upper()}: {vol:.4f}")
        
        # Print component linearity R² scores (critical for validating directions)
        if results.get("avg_component_linearity_r2"):
            print("Component Linearity R² (higher = more reliable direction):")
            for comp, r2 in results["avg_component_linearity_r2"].items():
                warning = " ⚠️ LOW" if r2 < 0.3 else ""
                print(f"  {comp}: {r2:.4f}{warning}")
            # Warn if month/day directions might be unreliable
            month_r2 = results["avg_component_linearity_r2"].get("month", 0)
            day_r2 = results["avg_component_linearity_r2"].get("day", 0)
            if month_r2 < 0.1 or day_r2 < 0.1:
                print("  ⚠️  Warning: Low R² for month/day means their directions may be distorted!")

        # Create visualizations for select layers
        if layer in [
            0,
            num_layers // 4,
            num_layers // 2,
            3 * num_layers // 4,
            num_layers - 1,
        ]:
            plot_temporal_geometry(
                results["embeddings"],
                CALENDAR_LABELS,
                layer,
                f"results/temporal_geometry/{safe_model_name}/pca_layer_{layer}.png",
            )
            plot_similarity_heatmap(
                results["cross_language_similarity"],
                languages,
                layer,
                f"results/temporal_geometry/{safe_model_name}/similarity_layer_{layer}.png",
            )
            
            # New calendar visualizations
            if results["calendar_component_vectors"]:
                plot_calendar_coordinate_system_3d(
                    results["calendar_component_vectors"],
                    results["embeddings"],
                    CALENDAR_LABELS,
                    layer,
                    f"results/temporal_geometry/{safe_model_name}/calendar_3d_layer_{layer}.png"
                )
                
                plot_component_dimension_contributions(
                    results["calendar_component_vectors"],
                    layer,
                    top_k=20,
                    save_path=f"results/temporal_geometry/{safe_model_name}/component_dims_layer_{layer}.png"
                )
                
                plot_temporal_trajectories_in_component_space(
                    results["calendar_component_vectors"],
                    results["embeddings"],
                    CALENDAR_LABELS,
                    layer,
                    f"results/temporal_geometry/{safe_model_name}/trajectories_layer_{layer}.png"
                )

    # Create layer progression plots
    plot_layer_progression(
        layer_results, 
        languages, 
        f"results/temporal_geometry/{safe_model_name}/layer_progression.png"
    )

    plot_calendar_component_alignment(
        layer_results, 
        f"results/temporal_geometry/{safe_model_name}/calendar_component_alignment.png"
    )

    plot_calendar_component_heatmap(
        layer_results, 
        f"results/temporal_geometry/{safe_model_name}/calendar_component_heatmap.png"
    )
    
    plot_component_prediction_accuracy(
        layer_results,
        f"results/temporal_geometry/{safe_model_name}/component_prediction_accuracy.png"
    )
    
    plot_component_linearity_r2(
        layer_results,
        f"results/temporal_geometry/{safe_model_name}/component_linearity_r2.png"
    )

    # Save numerical results
    summary = {
        "model": model_name,
        "years": YEARS,
        "samples_per_year": SAMPLES_PER_YEAR,
        "languages": languages,
        "layers": [],
    }

    for layer, results in enumerate(layer_results):
        layer_entry = {
            "layer": layer,
            "avg_cross_language_similarity": float(
                results["avg_cross_language_similarity"]
            ),
            "cross_language_similarity": {
                f"{k[0]}-{k[1]}": float(v)
                for k, v in results["cross_language_similarity"].items()
            },
            "temporal_linearity": {
                k: float(v) for k, v in results["temporal_linearity"].items()
            },
            "year_separability": {
                k: float(v) for k, v in results["year_separability"].items()
            },
        }

        # Add calendar fields if present
        if results["avg_calendar_disentanglement"] is not None:
            layer_entry["avg_calendar_disentanglement"] = float(
                results["avg_calendar_disentanglement"]
            )
            layer_entry["calendar_alignment"] = {
                comp: float(val) for comp, val in results["calendar_alignment"].items()
            }
        
        # Add parallelepiped volume (Gram determinant metric)
        if results.get("avg_parallelepiped_volume") is not None:
            layer_entry["avg_parallelepiped_volume"] = float(
                results["avg_parallelepiped_volume"]
            )
            layer_entry["parallelepiped_volume_per_lang"] = {
                lang: float(vol) for lang, vol in results["parallelepiped_volume"].items()
            }
            # Include pairwise cosines (off-diagonal Gram matrix elements)
            layer_entry["pairwise_cosines"] = {
                lang: {f"{k[0]}-{k[1]}": float(v) for k, v in cosines.items()}
                for lang, cosines in results.get("pairwise_cosines", {}).items()
            }
        
        # Add centered parallelepiped volume (embeddings centered before fitting probes)
        if results.get("avg_parallelepiped_volume_centered") is not None:
            layer_entry["avg_parallelepiped_volume_centered"] = float(
                results["avg_parallelepiped_volume_centered"]
            )
            layer_entry["parallelepiped_volume_centered_per_lang"] = {
                lang: float(vol) for lang, vol in results["parallelepiped_volume_centered"].items()
            }
        
        if results["component_accuracy"]:
            # Average across languages
            avg_accuracy = {}
            for lang, acc_dict in results["component_accuracy"].items():
                for comp, acc in acc_dict.items():
                    if comp not in avg_accuracy:
                        avg_accuracy[comp] = []
                    avg_accuracy[comp].append(acc)
            layer_entry["component_prediction_accuracy"] = {
                comp: float(np.mean(vals)) for comp, vals in avg_accuracy.items()
            }
        
        # Add component linearity R² (critical for validating direction vectors)
        if results.get("avg_component_linearity_r2"):
            layer_entry["component_linearity_r2"] = {
                comp: float(r2) for comp, r2 in results["avg_component_linearity_r2"].items()
            }
            # Flag unreliable components
            unreliable_components = [
                comp for comp, r2 in results["avg_component_linearity_r2"].items() 
                if r2 < 0.1
            ]
            if unreliable_components:
                layer_entry["unreliable_direction_vectors"] = unreliable_components

        summary["layers"].append(layer_entry)

    with open(f"results/temporal_geometry/{safe_model_name}/analysis_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE!")
    print("=" * 60)
    print(f"Results saved in 'results/temporal_geometry/{safe_model_name}/'")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run temporal geometry analysis")
    parser.add_argument("model_name", type=str, help="Name of the model to analyze")
    args = parser.parse_args()

    main(args.model_name)