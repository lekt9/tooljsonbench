import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import threading
import random

import pandas as pd
import ray
from tqdm import tqdm
from transformers import LlamaTokenizerFast

from llmperf import common_metrics
from llmperf.common import SUPPORTED_APIS, construct_clients
from llmperf.models import RequestConfig, ResponseFormat
from llmperf.requests_launcher import RequestsLauncher
from llmperf.validation_utils import validate_json_response, calculate_accuracy_score
from llmperf.utils import LLMPerfResults
from llmperf.csv_results_manager import CSVResultsManager


# JSON test prompts and schemas
JSON_TEST_CASES = [
    {
        "prompt": "Generate a JSON object representing a person with name, age, and email fields.",
        "schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "email": {"type": "string", "format": "email"}
            },
            "required": ["name", "age", "email"]
        }
    },
    {
        "prompt": "Create a JSON array of 3 books, each with title, author, and year published.",
        "schema": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "author": {"type": "string"},
                    "year": {"type": "integer"}
                },
                "required": ["title", "author", "year"]
            },
            "minItems": 3,
            "maxItems": 3
        }
    },
    {
        "prompt": "Generate a JSON object for a product catalog item with nested categories.",
        "schema": {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "name": {"type": "string"},
                "price": {"type": "number"},
                "category": {
                    "type": "object",
                    "properties": {
                        "main": {"type": "string"},
                        "sub": {"type": "string"}
                    },
                    "required": ["main", "sub"]
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            },
            "required": ["id", "name", "price", "category"]
        }
    }
]


def get_json_accuracy_metrics(
    model: str,
    test_cases: List[Dict[str, Any]],
    num_concurrent_requests: int = 1,
    max_num_completed_requests: int = 100,
    test_timeout_s: int = 300,
    llm_api: str = "openai",
    additional_sampling_params: Optional[Dict[str, Any]] = None
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Get JSON accuracy metrics for the given model."""
    
    random.seed(42)
    
    tokenizer = LlamaTokenizerFast.from_pretrained(
        "hf-internal-testing/llama-tokenizer"
    )
    get_token_length = lambda text: len(tokenizer.encode(text))
    
    if not additional_sampling_params:
        additional_sampling_params = {}
    
    completed_requests_lock = threading.Lock()
    completed_requests = []
    num_completed_requests = 0
    
    # Prepare test cases
    test_prompts = []
    for i in range(max_num_completed_requests):
        test_case = random.choice(test_cases)
        prompt_text = test_case["prompt"]
        prompt_len = get_token_length(prompt_text)
        test_prompts.append((test_case, (prompt_text, prompt_len)))
    
    start_time = time.monotonic()
    pbar = tqdm(total=max_num_completed_requests, desc="JSON Accuracy Test")
    
    def launch_request(thread_index):
        nonlocal num_completed_requests
        clients = construct_clients(llm_api=llm_api, num_clients=1)
        req_launcher = RequestsLauncher(clients)
        request_index = thread_index % max_num_completed_requests
        
        while (
            time.monotonic() - start_time < test_timeout_s
            and num_completed_requests < max_num_completed_requests
        ):
            test_case, prompt = test_prompts[request_index]
            
            # Set up JSON response format
            response_format = ResponseFormat(
                type="json_schema",
                json_schema=test_case["schema"]
            )

            # For Ollama, pass the schema directly as 'format'
            # For OpenAI-compatible APIs, use 'response_format'
            default_sampling_params = {
                "max_tokens": 500,
                "temperature": 0.1,
            }

            if llm_api == "ollama":
                # Ollama expects the schema directly in the 'format' parameter
                default_sampling_params["format"] = test_case["schema"]
            else:
                # OpenAI and other APIs use response_format
                default_sampling_params["response_format"] = response_format.model_dump()
            default_sampling_params.update(additional_sampling_params)
            
            request_config = RequestConfig(
                model=model,
                prompt=prompt,
                sampling_params=default_sampling_params,
                llm_api=llm_api,
                test_type="json_accuracy",
                response_format=response_format
            )
            
            req_launcher.launch_requests(request_config)
            outs = req_launcher.get_next_ready()
            
            for out in outs:
                request_metrics, generated_text, _ = out
                
                # Validate JSON response
                is_valid, parsed_json, validation_errors = validate_json_response(
                    generated_text, test_case["schema"]
                )
                
                # Calculate accuracy score
                accuracy_score = 1.0 if is_valid else 0.0
                
                with completed_requests_lock:
                    if num_completed_requests < max_num_completed_requests:
                        # Add JSON-specific metrics
                        request_metrics[common_metrics.JSON_VALID] = is_valid
                        request_metrics[common_metrics.JSON_SCHEMA_VALID] = is_valid
                        request_metrics[common_metrics.ACCURACY_SCORE] = accuracy_score
                        request_metrics[common_metrics.VALIDATION_ERRORS] = validation_errors
                        request_metrics["test_case"] = test_case["prompt"]
                        request_metrics["generated_text"] = generated_text
                        request_metrics["parsed_json"] = parsed_json
                        
                        completed_requests.append(request_metrics)
                        pbar.update(1)
                        num_completed_requests += 1
                        request_index = (request_index + num_concurrent_requests) % max_num_completed_requests
    
    # Launch threads
    threads = []
    for i in range(num_concurrent_requests):
        thread = threading.Thread(target=launch_request, args=(i,))
        threads.append(thread)
        thread.start()
    
    for thread in threads:
        thread.join()
    
    pbar.close()
    end_time = time.monotonic()
    
    if end_time - start_time >= test_timeout_s:
        print("Test timed out before all requests could be completed.")
    
    print(f"\nResults for JSON accuracy benchmark for {model} queried with the {llm_api} api.\n")
    
    # Calculate summary metrics
    total_requests = len(completed_requests)
    valid_json_count = sum(1 for req in completed_requests if req.get(common_metrics.JSON_VALID, False))
    schema_valid_count = sum(1 for req in completed_requests if req.get(common_metrics.JSON_SCHEMA_VALID, False))
    avg_accuracy = sum(req.get(common_metrics.ACCURACY_SCORE, 0) for req in completed_requests) / total_requests if total_requests > 0 else 0
    
    summary = {
        "model": model,
        "test_type": "json_accuracy",
        "total_requests": total_requests,
        "valid_json_rate": valid_json_count / total_requests if total_requests > 0 else 0,
        "schema_valid_rate": schema_valid_count / total_requests if total_requests > 0 else 0,
        "average_accuracy": avg_accuracy,
        "test_duration_s": end_time - start_time,
        "requests_per_second": total_requests / (end_time - start_time) if end_time > start_time else 0
    }
    
    return summary, completed_requests


def main():
    parser = argparse.ArgumentParser(description="Run JSON structured output accuracy benchmark.")

    parser.add_argument("--model", type=str, required=True, help="The model to test")
    parser.add_argument("--llm-api", type=str, default="ollama", choices=SUPPORTED_APIS, help="LLM API to use")
    parser.add_argument("--max-num-completed-requests", type=int, default=50, help="Number of requests to complete")
    parser.add_argument("--num-concurrent-requests", type=int, default=1, help="Number of concurrent requests")
    parser.add_argument("--timeout", type=int, default=300, help="Test timeout in seconds")
    parser.add_argument("--results-dir", type=str, default="result_outputs", help="Directory to save results")
    parser.add_argument("--additional-sampling-params", type=str, default="{}", help="Additional sampling parameters as JSON")
    parser.add_argument("--skip-if-cached", action="store_true", help="Skip test if already completed with same config")

    args = parser.parse_args()

    # Parse additional sampling params
    try:
        additional_sampling_params = json.loads(args.additional_sampling_params)
    except json.JSONDecodeError:
        print("Error: additional-sampling-params must be valid JSON")
        return

    # Initialize CSV results manager
    csv_manager = CSVResultsManager(results_dir=args.results_dir)

    # Check if test already completed
    test_config = {
        "max_num_completed_requests": args.max_num_completed_requests,
        "num_concurrent_requests": args.num_concurrent_requests,
        "timeout": args.timeout,
        "llm_api": args.llm_api,
        "additional_sampling_params": additional_sampling_params
    }

    if args.skip_if_cached and csv_manager.is_test_completed(args.model, "json_accuracy", test_config):
        print(f"✅ Test already completed for {args.model} with same configuration. Skipping...")
        existing_results = csv_manager.load_existing_results(args.model, "json_accuracy")
        if existing_results is not None:
            print(f"📊 Found {len(existing_results)} existing results")
        return

    # Initialize Ray
    ray.init(ignore_reinit_error=True)

    # Run benchmark
    summary, individual_results = get_json_accuracy_metrics(
        model=args.model,
        test_cases=JSON_TEST_CASES,
        num_concurrent_requests=args.num_concurrent_requests,
        max_num_completed_requests=args.max_num_completed_requests,
        test_timeout_s=args.timeout,
        llm_api=args.llm_api,
        additional_sampling_params=additional_sampling_params
    )

    # Save results using CSV manager
    csv_filename = csv_manager.save_individual_results_csv(
        individual_results, args.model, "json_accuracy", test_config
    )

    # Also save traditional JSON files
    results_dir = Path(args.results_dir)
    results_dir.mkdir(exist_ok=True)

    timestamp = int(time.time())
    summary_filename = f"json_accuracy_summary_{args.model.replace('/', '_')}_{timestamp}"

    # Save summary
    with open(results_dir / f"{summary_filename}.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Results saved to {results_dir}")
    print(f"CSV file: {csv_filename}")
    print(f"Summary: {summary}")


if __name__ == "__main__":
    main()
