import os
import time
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from tqdm import tqdm
from openai import AzureOpenAI, OpenAI
from dotenv import load_dotenv
from datetime import datetime
import pandas as pd


# Load .env if present
load_dotenv()

# -----------------------------
# Config
# -----------------------------

OUT_DIR = Path("dataset_mtb")
OUT_DIR.mkdir(exist_ok=True)
PREDICTIONS_OUTPUT_JSONL = OUT_DIR / "dataset_all_predictions_final_v2.jsonl"

# System prompt for the model being tested
PREDICTION_SYSTEM_PROMPT = """Give me the answer to the following question only when you are sure of it. \
Otherwise, say 'I don't know'. Put your answer on its own line after 'Answer:'.\n"""

# API Configuration
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION", None)
AZURE_API_URL = os.getenv("AZURE_API_URL", None)
AZURE_API_KEY = os.getenv("AZURE_API_KEY", None)
AZURE_ENGINE_NAME = os.getenv("AZURE_ENGINE_NAME", None)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN", None)


def get_client() -> OpenAI:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set (env or Colab userdata).")
    return OpenAI(api_key=OPENAI_API_KEY)


def get_client_azure() -> AzureOpenAI:
    """Initialize Azure OpenAI client."""
    if not AZURE_API_KEY:
        raise RuntimeError("AZURE_API_KEY not set (env or .env file).")
    return AzureOpenAI(
        api_key=AZURE_API_KEY,
        api_version=AZURE_API_VERSION,
        azure_endpoint=AZURE_API_URL,
        azure_deployment=AZURE_ENGINE_NAME,
    )


def load_vllm_model(
    model_name: str,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    max_model_len: Optional[int] = None,
):
    """Load model using vLLM for efficient inference."""
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        raise ImportError(
            "vllm is required for efficient inference. "
            "Install with: pip install vllm"
        )

    print(f"\nLoading model with vLLM: {model_name}")
    print("This may take a few minutes for large models...")

    # Configure vLLM
    llm_kwargs = {
        "model": model_name,
        "tensor_parallel_size": tensor_parallel_size,
        "gpu_memory_utilization": gpu_memory_utilization,
        "trust_remote_code": True,
    }

    # Add token if available
    if HUGGINGFACE_TOKEN:
        llm_kwargs["download_dir"] = None  # Use default HF cache
        # Set HF token in environment for vLLM to use
        os.environ["HF_TOKEN"] = HUGGINGFACE_TOKEN

    # Add max_model_len if specified
    if max_model_len:
        llm_kwargs["max_model_len"] = max_model_len

    llm = LLM(**llm_kwargs)

    print(f"✓ Model loaded successfully with vLLM")
    return llm


def generate_vllm_predictions_batch(
    llm,
    questions: List[str],
    temperature: float = 0.7,
    max_tokens: int = 512,
    top_p: float = 0.95,
) -> List[str]:
    """Generate predictions using vLLM in batch mode."""
    try:
        from vllm import SamplingParams
    except ImportError:
        raise ImportError("vllm is required. Install with: pip install vllm")

    # Create sampling parameters
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        stop=None,  # Let model decide when to stop
    )

    # Format prompts
    prompts = []
    for question in questions:
        # Try to use chat template if available
        try:
            # Get tokenizer from llm
            tokenizer = llm.get_tokenizer()
            messages = [
                {"role": "user", "content": f"{question}"},
            ]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            # Fallback to simple format
            prompt = f"{PREDICTION_SYSTEM_PROMPT}\n\nQuestion: {question}\n\nAnswer:"
        
        prompts.append(prompt)

    print(f"Generating {len(prompts)} predictions with vLLM...")
    
    # Generate in batch
    outputs = llm.generate(prompts, sampling_params)

    # Extract generated text
    predictions = []
    for output in outputs:
        generated_text = output.outputs[0].text.strip()
        predictions.append(generated_text)

    return predictions


def custom_id_for_row(row: Dict[str, Any], index: int, model: str) -> str:
    """Generate custom ID for batch request tracking."""
    safe_model = model.replace("/", "-").replace(":", "-")
    return f"pred-{safe_model}-{index}"


def build_prediction_batch_line(
    row: Dict[str, Any], model: str, index: int
) -> Dict[str, Any]:
    """Build a single batch request line for prediction generation."""
    question = row.get("question", "")
    user_msg = (
        f"""Please provide an answer to the following question: Question: {question}"""
    )

    return {
        "custom_id": custom_id_for_row(row, index, model),
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "temperature": 0.7,
            "messages": [
                {"role": "system", "content": PREDICTION_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            "max_tokens": 150,
        },
    }


def load_dataset_csv(file_path: str) -> pd.DataFrame:
    """Load the MultiTempBench dataset from CSV."""
    # Try different encodings
    encodings = ["utf-8", "latin-1", "iso-8859-1", "cp1252", "utf-16"]

    for encoding in encodings:
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            print(
                f"Successfully loaded {len(df)} examples from {file_path} with encoding: {encoding}"
            )
            print(f"Columns: {df.columns.tolist()}")
            print(f"Date formats: {df['Date_format'].unique()}")
            print(f"Tasks: {df['Task'].unique()}")
            return df
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"Failed with encoding {encoding}: {e}")
            continue

    # If all encodings fail, try with error handling
    try:
        df = pd.read_csv(file_path, encoding="utf-8", encoding_errors="ignore")
        print(
            f"Loaded {len(df)} examples from {file_path} with UTF-8 (ignoring errors)"
        )
        print(f"Columns: {df.columns.tolist()}")
        return df
    except Exception as e:
        raise Exception(
            f"Could not load file {file_path} with any encoding. Error: {e}"
        )


def load_dataset_from_jsonl(file_path: Path) -> List[Dict[str, Any]]:
    """Load dataset from JSONL file."""
    dataset = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            dataset.append(json.loads(line))
    return dataset


def load_dataset_from_csv(file_path: Path) -> List[Dict[str, Any]]:
    """Load dataset from CSV file and convert to internal format."""
    # Use the robust CSV loader
    df = load_dataset_csv(str(file_path))
    
    # Convert DataFrame to list of dicts
    dataset = []
    for _, row in df.iterrows():
        dataset_row = {
            "question": row.get("Question", ""),
            "answer": row.get("Answer", ""),
            "date_format": row.get("Date_format", ""),
            "task": row.get("Task", ""),
            "source": row.get("Source", ""),
            "language": row.get("Language", ""),
        }
        dataset.append(dataset_row)
    
    return dataset


def load_dataset(file_path: Path) -> List[Dict[str, Any]]:
    """Load dataset from either CSV or JSONL file based on extension."""
    if file_path.suffix.lower() == '.csv':
        return load_dataset_from_csv(file_path)
    elif file_path.suffix.lower() == '.jsonl':
        return load_dataset_from_jsonl(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}. Use .csv or .jsonl")


def load_existing_predictions() -> Dict[str, Dict[str, Any]]:
    """Load existing predictions from output file if it exists."""
    if not PREDICTIONS_OUTPUT_JSONL.exists():
        return {}

    existing = {}
    with PREDICTIONS_OUTPUT_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            question = row.get("question", "")
            model = row.get("model_name", "")
            if question and model:
                key = (question, model)
                existing[key] = row

    return existing


def run_openai_batch(
    client: OpenAI, dataset: List[Dict[str, Any]], model: str
) -> Dict[str, str]:
    """Run batch prediction using OpenAI Batch API."""
    request_file = OUT_DIR / f"prediction_requests_{model.replace('/', '-')}.jsonl"
    results_file = OUT_DIR / f"prediction_results_{model.replace('/', '-')}.jsonl"

    # 1) Build JSONL
    cids = []
    with request_file.open("w", encoding="utf-8") as f:
        for index, row in enumerate(dataset):
            obj = build_prediction_batch_line(row, model, index)
            cids.append(obj["custom_id"])
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"Wrote {len(cids)} requests to {request_file.resolve()}")

    # 2) Upload + create batch job
    with open(request_file, "rb") as file:
        batch_input_file = client.files.create(file=file, purpose="batch")

    batch_job = client.batches.create(
        input_file_id=batch_input_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"task": "prediction-generation", "model": model},
    )
    print(f"Batch created: {batch_job.id}, status: {batch_job.status}")

    # 3) Poll
    TERMINAL = {"completed", "failed", "cancelled", "expired"}
    sleep_time = 5

    print("\nPolling batch job status...")
    while True:
        job = client.batches.retrieve(batch_job.id)
        counts = getattr(job, "request_counts", None)
        print(f"[{time.strftime('%X')}] status={job.status} counts={counts}")

        if job.status in TERMINAL:
            break

        time.sleep(sleep_time)
        sleep_time = min(sleep_time * 1.5, 60)

    if job.status != "completed":
        raise RuntimeError(f"Batch job ended with status={job.status}")

    # 4) Download results
    result_file_id = job.output_file_id
    content = client.files.content(result_file_id).content
    results_file.write_bytes(content)
    print(f"\nSaved results to {results_file.resolve()}")

    # 5) Parse results
    id_to_prediction: Dict[str, str] = {}
    with results_file.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            try:
                predicted_answer = rec["response"]["body"]["choices"][0]["message"][
                    "content"
                ]
                id_to_prediction[rec["custom_id"]] = predicted_answer
            except Exception as e:
                print(
                    f"Warning: Could not parse prediction for {rec.get('custom_id', 'unknown')}: {e}"
                )
                id_to_prediction[rec["custom_id"]] = f"[Error: {e}]"

    return id_to_prediction


def run_vllm_inference(
    llm,
    dataset: List[Dict[str, Any]],
    model_name: str,
    max_tokens: int = 512,
    batch_size: int = 32,
    temperature: float = 0.7,
) -> Dict[str, str]:
    """Run inference using vLLM in batched mode."""
    print(f"\nGenerating predictions with vLLM: {model_name}")
    print(f"Processing in batches of {batch_size}...")
    
    id_to_prediction: Dict[str, str] = {}
    
    # Process in batches
    total_batches = (len(dataset) + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(0, len(dataset), batch_size), 
                          total=total_batches,
                          desc="Batch processing"):
        batch_data = dataset[batch_idx:batch_idx + batch_size]
        batch_questions = [row.get("question", "") for row in batch_data]
        
        try:
            # Generate predictions for entire batch
            predictions = generate_vllm_predictions_batch(
                llm=llm,
                questions=batch_questions,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            
            # Map predictions back to custom IDs
            for i, (row, prediction) in enumerate(zip(batch_data, predictions)):
                global_idx = batch_idx + i
                cid = custom_id_for_row(row, global_idx, model_name)
                id_to_prediction[cid] = prediction
                
        except Exception as e:
            print(f"\nError in batch {batch_idx // batch_size}: {e}")
            # Mark all items in failed batch
            for i, row in enumerate(batch_data):
                global_idx = batch_idx + i
                cid = custom_id_for_row(row, global_idx, model_name)
                id_to_prediction[cid] = f"[Error: {e}]"
    
    return id_to_prediction


def merge_predictions_and_save(
    dataset: List[Dict[str, Any]],
    id_to_prediction: Dict[str, str],
    model: str,
    existing_predictions: Dict[str, Dict[str, Any]],
) -> None:
    """Merge predictions with original dataset and save."""
    print("\n=== Merging Predictions and Saving ===")

    # Collect all rows
    all_rows = []

    # Add existing predictions for other models
    for row in existing_predictions.values():
        if row.get("model_name") != model:
            all_rows.append(row)

    # Add new predictions
    for index, row in enumerate(dataset):
        cid = custom_id_for_row(row, index, model)
        predicted_answer = id_to_prediction.get(cid, "[Error: No prediction generated]")

        new_row = row.copy()
        new_row["predicted_answer"] = predicted_answer
        new_row["model_name"] = model
        all_rows.append(new_row)

    # Write all rows
    with PREDICTIONS_OUTPUT_JSONL.open("w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Saved dataset to {PREDICTIONS_OUTPUT_JSONL.resolve()}")

    # Statistics
    successful = sum(
        1 for v in id_to_prediction.values() if not v.startswith("[Error:")
    )
    print(f"\n=== Statistics for {model} ===")
    print(f"Total questions: {len(dataset)}")
    print(f"Successfully generated: {successful}")
    print(f"Failed: {len(dataset) - successful}")

    models_in_file = set(row.get("model_name") for row in all_rows)
    print(f"\nTotal models in file: {len(models_in_file)}")
    print(f"Models: {', '.join(sorted(models_in_file))}")


def detect_model_type(model_name: str) -> str:
    """Detect whether model is OpenAI, Azure, or vLLM."""
    openai_models = ["gpt-3.5", "gpt-4", "gpt-4o", "o1", "o3"]

    # Check if it's an OpenAI model
    if any(m in model_name.lower() for m in openai_models):
        # Check if Azure is configured
        if AZURE_API_KEY:
            return "azure"
        elif OPENAI_API_KEY:
            return "openai"
        else:
            raise RuntimeError("OpenAI or Azure API key required for OpenAI models")

    # Otherwise, assume vLLM
    return "vllm"


def process_single_model(
    model: str,
    dataset: List[Dict[str, Any]],
    existing_predictions: Dict[str, Dict[str, Any]],
    provider: Optional[str] = None,
    max_tokens: int = 512,
    batch_size: int = 32,
    temperature: float = 0.7,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    max_model_len: Optional[int] = None,
) -> None:
    """Process a single model and generate predictions."""
    print(f"\n{'=' * 80}")
    print(f"Processing Model: {model}")
    print(f"{'=' * 80}")

    # Check if predictions already exist
    existing_model_count = sum(
        1 for row in existing_predictions.values() if row.get("model_name") == model
    )
    if existing_model_count > 0:
        print(
            f"⚠️  Warning: Found {existing_model_count} existing predictions for {model}"
        )
        print("These will be replaced with new predictions.\n")

    # Detect provider if not specified
    if provider is None:
        provider = detect_model_type(model)

    print(f"Using provider: {provider}\n")

    # Run predictions based on provider
    try:
        if provider == "openai":
            client = get_client()
            id_to_prediction = run_openai_batch(client, dataset, model)
        elif provider == "azure":
            client = get_client_azure()
            id_to_prediction = run_openai_batch(client, dataset, model)
        elif provider == "vllm":
            llm = load_vllm_model(
                model,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
            )
            id_to_prediction = run_vllm_inference(
                llm, dataset, model, max_tokens, batch_size, temperature
            )
            # Clean up GPU memory
            del llm
            try:
                import torch
                import gc
                gc.collect()
                torch.cuda.empty_cache()
            except:
                pass
        else:
            raise ValueError(f"Unknown provider: {provider}")

        # Merge and save
        merge_predictions_and_save(
            dataset, id_to_prediction, model, existing_predictions
        )

        # Reload existing predictions for next model
        return load_existing_predictions()

    except Exception as e:
        print(f"\n❌ Error processing model {model}: {e}")
        import traceback
        traceback.print_exc()
        print("Continuing to next model...\n")
        return existing_predictions


def main(
    models: List[str],
    input_file: str,
    provider: Optional[str] = None,
    max_tokens: int = 512,
    batch_size: int = 32,
    temperature: float = 0.7,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    max_model_len: Optional[int] = None,
) -> None:
    """Main execution function."""
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print(f"Loading dataset from {input_path}")
    dataset = load_dataset(input_path)
    print(f"Loaded {len(dataset)} questions")
    print(f"Will process {len(models)} models: {', '.join(models)}\n")

    # Load existing predictions once at the start
    existing_predictions = load_existing_predictions()

    # Process each model
    for i, model in enumerate(models, 1):
        print(f"\n{'#' * 80}")
        print(f"# Model {i}/{len(models)}")
        print(f"{'#' * 80}")
        batch_size = len(dataset) if provider in ["vllm"] else batch_size

        if "Qwen3-30B-A3B" in model:
            max_model_len = 131072

        existing_predictions = process_single_model(
            model,
            dataset,
            existing_predictions,
            provider,
            max_tokens,
            batch_size,
            temperature,
            tensor_parallel_size,
            gpu_memory_utilization,
            max_model_len,
        )

    print(f"\n{'=' * 80}")
    print("✓ ALL MODELS PROCESSED SUCCESSFULLY!")
    print(f"{'=' * 80}")
    print(f"\nOutput file: {PREDICTIONS_OUTPUT_JSONL.resolve()}")
    print(
        f"Next step: Run evaluate.py with --input {PREDICTIONS_OUTPUT_JSONL.resolve()}"
    )


def run():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate predicted answers from AI models for evaluation"
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        help="Model names (e.g., gpt-4o-mini meta-llama/Llama-3.1-8B-Instruct)",
    )
    parser.add_argument(
        "--models-file",
        type=str,
        help="Path to a text file containing model names (one per line)",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="datasetv2/dataset_with_difficulty.jsonl",
        help="Input JSONL file path",
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=["openai", "azure", "vllm"],
        default=None,
        help="Model provider (auto-detected if not specified)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for vLLM inference",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism (vLLM)",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization for vLLM (0.0-1.0)",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Maximum model context length (vLLM)",
    )
    args = parser.parse_args()

    # Get models list
    if args.models:
        models = args.models
    elif args.models_file:
        with open(args.models_file, "r") as f:
            models = [
                line.strip() for line in f if line.strip() and not line.startswith("#")
            ]
    else:
        raise ValueError("Either --models or --models-file must be specified")

    main(
        models,
        args.input,
        args.provider,
        args.max_tokens,
        args.batch_size,
        args.temperature,
        args.tensor_parallel_size,
        args.gpu_memory_utilization,
        args.max_model_len,
    )


if __name__ == "__main__":
    run()