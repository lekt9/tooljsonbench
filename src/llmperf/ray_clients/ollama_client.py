import json
import os
import time
from typing import Any, Dict, Optional
import ray
import requests

from llmperf.ray_llm_client import LLMClient
from llmperf.models import RequestConfig
from llmperf import common_metrics


@ray.remote
class OllamaClient(LLMClient):
    """Client for Ollama API with support for structured outputs and tool calls."""

    def __init__(self):
        self.base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        if not self.base_url.endswith("/"):
            self.base_url += "/"

    def llm_request(self, request_config: RequestConfig) -> Dict[str, Any]:
        prompt = request_config.prompt
        prompt_text, prompt_len = prompt

        model = request_config.model
        sampling_params = request_config.sampling_params or {}

        # Check if tools are being used - if so, use chat completion API
        if "tools" in sampling_params or request_config.tools:
            return self.chat_completion_request(request_config)

        # Prepare the request body for generate API
        body = {
            "model": model,
            "prompt": prompt_text,
            "stream": True,
            "options": {}
        }
        
        # Handle structured output parameters
        if "format" in sampling_params:
            # Direct format parameter (preferred for Ollama)
            body["format"] = sampling_params["format"]
        elif "response_format" in sampling_params:
            # Legacy response_format parameter - extract the schema
            response_format = sampling_params["response_format"]
            if isinstance(response_format, dict):
                if "json_schema" in response_format:
                    body["format"] = response_format["json_schema"]
                elif response_format.get("type") == "json_object":
                    body["format"] = "json"
                else:
                    body["format"] = response_format
            else:
                body["format"] = response_format
        
        if "tools" in sampling_params:
            body["tools"] = sampling_params["tools"]
            
        if "tool_choice" in sampling_params:
            body["tool_choice"] = sampling_params["tool_choice"]
        
        # Handle standard sampling parameters
        if "max_tokens" in sampling_params:
            body["options"]["num_predict"] = sampling_params["max_tokens"]
        if "temperature" in sampling_params:
            body["options"]["temperature"] = sampling_params["temperature"]
        if "top_p" in sampling_params:
            body["options"]["top_p"] = sampling_params["top_p"]
        if "top_k" in sampling_params:
            body["options"]["top_k"] = sampling_params["top_k"]
        if "stop" in sampling_params:
            body["options"]["stop"] = sampling_params["stop"]

        # Initialize metrics tracking
        time_to_next_token = []
        tokens_received = 0
        ttft = 0
        error_response_code = -1
        generated_text = ""
        error_msg = ""
        output_throughput = 0
        total_request_time = 0

        metrics = {}
        metrics[common_metrics.ERROR_CODE] = None
        metrics[common_metrics.ERROR_MSG] = ""

        start_time = time.monotonic()
        most_recent_received_token_time = time.monotonic()
        
        try:
            url = f"{self.base_url}api/generate"
            with requests.post(
                url,
                json=body,
                stream=True,
                timeout=180,
                headers={"Content-Type": "application/json"}
            ) as response:
                if response.status_code != 200:
                    error_msg = response.text
                    error_response_code = response.status_code
                    response.raise_for_status()
                
                for line in response.iter_lines():
                    if not line:
                        continue
                    
                    try:
                        data = json.loads(line.decode('utf-8'))
                    except json.JSONDecodeError:
                        continue
                    
                    if "error" in data:
                        error_msg = data["error"]
                        error_response_code = -1
                        raise RuntimeError(data["error"])
                    
                    # Check if this is a token response
                    if "response" in data and data["response"]:
                        tokens_received += 1
                        
                        if not ttft:
                            ttft = time.monotonic() - start_time
                            time_to_next_token.append(ttft)
                        else:
                            time_to_next_token.append(
                                time.monotonic() - most_recent_received_token_time
                            )
                        most_recent_received_token_time = time.monotonic()
                        generated_text += data["response"]
                    
                    # Check if generation is done
                    if data.get("done", False):
                        break

            total_request_time = time.monotonic() - start_time
            output_throughput = tokens_received / total_request_time if total_request_time > 0 else 0

        except Exception as e:
            metrics[common_metrics.ERROR_MSG] = error_msg or str(e)
            metrics[common_metrics.ERROR_CODE] = error_response_code
            print(f"Warning Or Error: {e}")
            print(error_response_code)

        # Calculate metrics
        metrics[common_metrics.INTER_TOKEN_LAT] = sum(time_to_next_token)
        metrics[common_metrics.TTFT] = ttft
        metrics[common_metrics.E2E_LAT] = total_request_time
        metrics[common_metrics.REQ_OUTPUT_THROUGHPUT] = output_throughput
        metrics[common_metrics.NUM_TOTAL_TOKENS] = tokens_received + prompt_len
        metrics[common_metrics.NUM_OUTPUT_TOKENS] = tokens_received
        metrics[common_metrics.NUM_INPUT_TOKENS] = prompt_len

        return metrics, generated_text, request_config

    def chat_completion_request(self, request_config: RequestConfig) -> Dict[str, Any]:
        """Handle chat completion requests for Ollama."""
        prompt = request_config.prompt
        prompt_text, prompt_len = prompt

        model = request_config.model
        sampling_params = request_config.sampling_params or {}
        
        # For chat completions, we need to format the prompt as messages
        # If prompt_text is already a list of messages, use it directly
        if isinstance(prompt_text, list):
            messages = prompt_text
        else:
            messages = [{"role": "user", "content": prompt_text}]
        
        # Prepare the request body for chat completion
        body = {
            "model": model,
            "messages": messages,
            "stream": True,
            "options": {}
        }
        
        # Handle structured output parameters
        if "format" in sampling_params:
            # Direct format parameter (preferred for Ollama)
            body["format"] = sampling_params["format"]
        elif "response_format" in sampling_params:
            # Legacy response_format parameter - extract the schema
            response_format = sampling_params["response_format"]
            if isinstance(response_format, dict):
                if "json_schema" in response_format:
                    body["format"] = response_format["json_schema"]
                elif response_format.get("type") == "json_object":
                    body["format"] = "json"
                else:
                    body["format"] = response_format
            else:
                body["format"] = response_format
        
        if "tools" in sampling_params:
            body["tools"] = sampling_params["tools"]
            
        if "tool_choice" in sampling_params:
            body["tool_choice"] = sampling_params["tool_choice"]
        
        # Handle standard sampling parameters
        if "max_tokens" in sampling_params:
            body["options"]["num_predict"] = sampling_params["max_tokens"]
        if "temperature" in sampling_params:
            body["options"]["temperature"] = sampling_params["temperature"]
        if "top_p" in sampling_params:
            body["options"]["top_p"] = sampling_params["top_p"]
        if "top_k" in sampling_params:
            body["options"]["top_k"] = sampling_params["top_k"]
        if "stop" in sampling_params:
            body["options"]["stop"] = sampling_params["stop"]

        # Initialize metrics tracking
        time_to_next_token = []
        tokens_received = 0
        ttft = 0
        error_response_code = -1
        generated_text = ""
        error_msg = ""
        output_throughput = 0
        total_request_time = 0
        tool_calls = []  # Track tool calls

        metrics = {}
        metrics[common_metrics.ERROR_CODE] = None
        metrics[common_metrics.ERROR_MSG] = ""

        start_time = time.monotonic()
        most_recent_received_token_time = time.monotonic()
        
        try:
            url = f"{self.base_url}api/chat"
            with requests.post(
                url,
                json=body,
                stream=True,
                timeout=180,
                headers={"Content-Type": "application/json"}
            ) as response:
                if response.status_code != 200:
                    error_msg = response.text
                    error_response_code = response.status_code
                    response.raise_for_status()
                
                for line in response.iter_lines():
                    if not line:
                        continue
                    
                    try:
                        data = json.loads(line.decode('utf-8'))
                    except json.JSONDecodeError:
                        continue
                    
                    if "error" in data:
                        error_msg = data["error"]
                        error_response_code = -1
                        raise RuntimeError(data["error"])
                    
                    # Check if this is a message response
                    if "message" in data:
                        message = data["message"]

                        # Handle content
                        if "content" in message and message["content"]:
                            content = message["content"]
                            tokens_received += 1

                            if not ttft:
                                ttft = time.monotonic() - start_time
                                time_to_next_token.append(ttft)
                            else:
                                time_to_next_token.append(
                                    time.monotonic() - most_recent_received_token_time
                                )
                            most_recent_received_token_time = time.monotonic()
                            generated_text += content

                        # Handle tool calls
                        if "tool_calls" in message and message["tool_calls"]:
                            tool_calls.extend(message["tool_calls"])
                    
                    # Check if generation is done
                    if data.get("done", False):
                        break

            total_request_time = time.monotonic() - start_time
            output_throughput = tokens_received / total_request_time if total_request_time > 0 else 0

        except Exception as e:
            metrics[common_metrics.ERROR_MSG] = error_msg or str(e)
            metrics[common_metrics.ERROR_CODE] = error_response_code
            print(f"Warning Or Error: {e}")
            print(error_response_code)

        # If tool calls were made, append them to generated_text in a parseable format
        if tool_calls:
            for tool_call in tool_calls:
                if "function" in tool_call:
                    func_name = tool_call["function"]["name"]
                    func_args = tool_call["function"]["arguments"]
                    # Format as JSON that the validation logic can parse
                    tool_call_json = json.dumps({
                        "function": func_name,
                        "arguments": func_args
                    })
                    generated_text += f"\n{tool_call_json}"

        # Calculate metrics
        metrics[common_metrics.INTER_TOKEN_LAT] = sum(time_to_next_token)
        metrics[common_metrics.TTFT] = ttft
        metrics[common_metrics.E2E_LAT] = total_request_time
        metrics[common_metrics.REQ_OUTPUT_THROUGHPUT] = output_throughput
        metrics[common_metrics.NUM_TOTAL_TOKENS] = tokens_received + prompt_len
        metrics[common_metrics.NUM_OUTPUT_TOKENS] = tokens_received
        metrics[common_metrics.NUM_INPUT_TOKENS] = prompt_len

        return metrics, generated_text, request_config
