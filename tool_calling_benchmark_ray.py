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
from llmperf.models import RequestConfig, Tool, ToolFunction
from llmperf.requests_launcher import RequestsLauncher
from llmperf.validation_utils import validate_tool_calls, calculate_accuracy_score
from llmperf.utils import LLMPerfResults
from llmperf.csv_results_manager import CSVResultsManager


# Tool definitions for testing
AVAILABLE_TOOLS = [
    Tool(
        type="function",
        function=ToolFunction(
            name="get_weather",
            description="Get the current weather for a location",
            parameters={
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The temperature unit"
                    }
                },
                "required": ["location"]
            }
        )
    ),
    Tool(
        type="function",
        function=ToolFunction(
            name="calculate_tip",
            description="Calculate tip amount for a bill",
            parameters={
                "type": "object",
                "properties": {
                    "bill_amount": {
                        "type": "number",
                        "description": "The total bill amount"
                    },
                    "tip_percentage": {
                        "type": "number",
                        "description": "The tip percentage (e.g., 15 for 15%)"
                    }
                },
                "required": ["bill_amount", "tip_percentage"]
            }
        )
    ),
    Tool(
        type="function",
        function=ToolFunction(
            name="send_email",
            description="Send an email to a recipient",
            parameters={
                "type": "object",
                "properties": {
                    "to": {
                        "type": "string",
                        "description": "Email address of the recipient"
                    },
                    "subject": {
                        "type": "string",
                        "description": "Email subject"
                    },
                    "body": {
                        "type": "string",
                        "description": "Email body content"
                    }
                },
                "required": ["to", "subject", "body"]
            }
        )
    )
]

# Tool calling test cases
TOOL_TEST_CASES = [
    {
        "prompt": "What's the weather like in New York City?",
        "expected_tool": "get_weather",
        "expected_args": {"location": "New York City"}
    },
    {
        "prompt": "Calculate a 20% tip on a $85.50 bill",
        "expected_tool": "calculate_tip",
        "expected_args": {"bill_amount": 85.50, "tip_percentage": 20}
    },
    {
        "prompt": "Send an email to john@example.com with subject 'Meeting Tomorrow' and body 'Don't forget about our meeting at 2 PM tomorrow.'",
        "expected_tool": "send_email",
        "expected_args": {
            "to": "john@example.com",
            "subject": "Meeting Tomorrow",
            "body": "Don't forget about our meeting at 2 PM tomorrow."
        }
    },
    {
        "prompt": "What's the temperature in Los Angeles in Celsius?",
        "expected_tool": "get_weather",
        "expected_args": {"location": "Los Angeles", "unit": "celsius"}
    },
    {
        "prompt": "How much should I tip on a $42.75 restaurant bill if I want to tip 18%?",
        "expected_tool": "calculate_tip",
        "expected_args": {"bill_amount": 42.75, "tip_percentage": 18}
    }
]


def get_tool_calling_metrics(
    model: str,
    test_cases: List[Dict[str, Any]],
    available_tools: List[Tool],
    num_concurrent_requests: int = 1,
    max_num_completed_requests: int = 50,
    test_timeout_s: int = 300,
    llm_api: str = "openai",
    additional_sampling_params: Optional[Dict[str, Any]] = None
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Get tool calling accuracy metrics for the given model."""
    
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
    pbar = tqdm(total=max_num_completed_requests, desc="Tool Calling Test")
    
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
            
            default_sampling_params = {
                "max_tokens": 300,
                "temperature": 0.1,
                "tools": [tool.model_dump() for tool in available_tools],
                "tool_choice": "auto"
            }
            default_sampling_params.update(additional_sampling_params)
            
            request_config = RequestConfig(
                model=model,
                prompt=prompt,
                sampling_params=default_sampling_params,
                llm_api=llm_api,
                test_type="tool_calling",
                tools=available_tools
            )
            
            req_launcher.launch_requests(request_config)
            outs = req_launcher.get_next_ready()
            
            for out in outs:
                request_metrics, generated_text, _ = out
                
                # Validate tool calls
                has_valid_calls, extracted_calls, validation_errors = validate_tool_calls(
                    generated_text, [tool.model_dump() for tool in available_tools]
                )
                
                # Check if the correct tool was called
                expected_tool = test_case["expected_tool"]
                expected_args = test_case["expected_args"]
                
                tool_call_accuracy = 0.0
                if extracted_calls:
                    for call in extracted_calls:
                        if call.get("function") == expected_tool:
                            # Check argument accuracy
                            actual_args = call.get("arguments", {})
                            if all(actual_args.get(k) == v for k, v in expected_args.items()):
                                tool_call_accuracy = 1.0
                                break
                
                with completed_requests_lock:
                    if num_completed_requests < max_num_completed_requests:
                        # Add tool calling specific metrics
                        request_metrics[common_metrics.TOOL_CALL_SUCCESS] = has_valid_calls
                        request_metrics[common_metrics.NUM_TOOL_CALLS] = len(extracted_calls)
                        request_metrics[common_metrics.TOOL_CALL_ACCURACY] = tool_call_accuracy
                        request_metrics[common_metrics.VALIDATION_ERRORS] = validation_errors
                        request_metrics["test_case"] = test_case["prompt"]
                        request_metrics["expected_tool"] = expected_tool
                        request_metrics["expected_args"] = expected_args
                        request_metrics["extracted_calls"] = extracted_calls
                        request_metrics["generated_text"] = generated_text
                        
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
    
    print(f"\nResults for tool calling benchmark for {model} queried with the {llm_api} api.\n")
    
    # Calculate summary metrics
    total_requests = len(completed_requests)
    successful_calls = sum(1 for req in completed_requests if req.get(common_metrics.TOOL_CALL_SUCCESS, False))
    avg_accuracy = sum(req.get(common_metrics.TOOL_CALL_ACCURACY, 0) for req in completed_requests) / total_requests if total_requests > 0 else 0
    total_tool_calls = sum(req.get(common_metrics.NUM_TOOL_CALLS, 0) for req in completed_requests)
    
    summary = {
        "model": model,
        "test_type": "tool_calling",
        "total_requests": total_requests,
        "successful_call_rate": successful_calls / total_requests if total_requests > 0 else 0,
        "average_accuracy": avg_accuracy,
        "total_tool_calls": total_tool_calls,
        "avg_calls_per_request": total_tool_calls / total_requests if total_requests > 0 else 0,
        "test_duration_s": end_time - start_time,
        "requests_per_second": total_requests / (end_time - start_time) if end_time > start_time else 0
    }
    
    return summary, completed_requests


def main():
    parser = argparse.ArgumentParser(description="Run tool calling accuracy benchmark.")
    
    parser.add_argument("--model", type=str, required=True, help="The model to test")
    parser.add_argument("--llm-api", type=str, default="ollama", choices=SUPPORTED_APIS, help="LLM API to use")
    parser.add_argument("--max-num-completed-requests", type=int, default=25, help="Number of requests to complete")
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

    if args.skip_if_cached and csv_manager.is_test_completed(args.model, "tool_calling", test_config):
        print(f"✅ Test already completed for {args.model} with same configuration. Skipping...")
        existing_results = csv_manager.load_existing_results(args.model, "tool_calling")
        if existing_results is not None:
            print(f"📊 Found {len(existing_results)} existing results")
        return

    # Initialize Ray
    ray.init(ignore_reinit_error=True)
    
    # Run benchmark
    summary, individual_results = get_tool_calling_metrics(
        model=args.model,
        test_cases=TOOL_TEST_CASES,
        available_tools=AVAILABLE_TOOLS,
        num_concurrent_requests=args.num_concurrent_requests,
        max_num_completed_requests=args.max_num_completed_requests,
        test_timeout_s=args.timeout,
        llm_api=args.llm_api,
        additional_sampling_params=additional_sampling_params
    )
    
    # Save results using CSV manager
    csv_filename = csv_manager.save_individual_results_csv(
        individual_results, args.model, "tool_calling", test_config
    )

    # Also save traditional JSON files
    results_dir = Path(args.results_dir)
    results_dir.mkdir(exist_ok=True)

    timestamp = int(time.time())
    summary_filename = f"tool_calling_summary_{args.model.replace('/', '_')}_{timestamp}"

    # Save summary
    with open(results_dir / f"{summary_filename}.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Results saved to {results_dir}")
    print(f"CSV file: {csv_filename}")
    print(f"Summary: {summary}")


if __name__ == "__main__":
    main()
