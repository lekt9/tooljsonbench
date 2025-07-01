# LLMPerf Structured Output & Tool Calling Benchmarking

This enhanced version of LLMPerf adds support for benchmarking JSON structured outputs and tool calling capabilities, with comprehensive performance metrics including TTFT (Time to First Token) and TPS (Tokens Per Second).

## New Features

- **Ollama API Support**: Native support for Ollama API with streaming responses
- **JSON Structured Output Testing**: Validate JSON schema compliance and accuracy
- **Tool Calling Benchmarks**: Test function calling capabilities with accuracy metrics
- **CSV Results with Caching**: Individual record tracking with smart caching to skip completed tests
- **Master Benchmark Script**: Run comprehensive benchmarks across multiple models and test types

## Installation

```bash
# Install additional dependencies for structured output validation
pip install jsonschema pandas
```

## Quick Start

### 1. Test a Single Model with Ollama

```bash
# Start Ollama and pull models
ollama pull gemma3n
ollama pull smollm2:360m
ollama pull llama3.2:1b

# Run JSON accuracy benchmark
python json_benchmark_ray.py --model gemma3n --llm-api ollama --max-num-completed-requests 30

# Run tool calling benchmark
python tool_calling_benchmark_ray.py --model smollm2:360m --llm-api ollama --max-num-completed-requests 20

# Run performance benchmark (original functionality)
python token_benchmark_ray.py --model llama3.2:1b --llm-api ollama --max-num-completed-requests 50
```

### 2. Run Master Benchmark (All Models, All Tests)

```bash
# Run comprehensive benchmark suite
python master_benchmark.py --models gemma3n smollm2:360m llama3.2:1b --llm-api ollama

# Run with caching (skip already completed tests)
python master_benchmark.py --models gemma3n smollm2:360m llama3.2:1b --llm-api ollama --skip-cached

# Run specific test types only
python master_benchmark.py --test-types json_accuracy tool_calling --models gemma3n
```

## Test Types

### 1. Performance Benchmarking (Enhanced)
- **Metrics**: TTFT, TPS, end-to-end latency, throughput
- **Features**: Streaming support, concurrent requests
- **Usage**: Same as original LLMPerf but with Ollama support

### 2. JSON Structured Output Testing
- **Validates**: JSON parsing, schema compliance, structure accuracy
- **Test Cases**: Person objects, book arrays, nested product catalogs
- **Metrics**: JSON validity rate, schema compliance rate, accuracy scores

### 3. Tool Calling Benchmarking
- **Tests**: Function calling accuracy, parameter validation
- **Available Tools**: Weather lookup, tip calculator, email sender
- **Metrics**: Tool call success rate, parameter accuracy, function selection accuracy

## Configuration

### Environment Variables

```bash
# Ollama API (default: http://localhost:11434)
export OLLAMA_BASE_URL="http://localhost:11434"

# For other APIs (existing functionality)
export OPENAI_API_KEY="your-key"
export OPENAI_API_BASE="your-endpoint"
```

### Test Configuration

The master benchmark script uses these default configurations:

```python
TEST_CONFIGS = {
    "performance": {
        "max-num-completed-requests": 50,
        "mean-input-tokens": 550,
        "mean-output-tokens": 150
    },
    "json_accuracy": {
        "max-num-completed-requests": 30,
        "timeout": 300
    },
    "tool_calling": {
        "max-num-completed-requests": 20,
        "timeout": 300
    }
}
```

## Results Management

### CSV Results with Caching

All benchmarks now save results in CSV format with individual record tracking:

```bash
# View cache statistics
python manage_results.py stats

# List completed tests
python manage_results.py list

# Consolidate all results into single CSV
python manage_results.py consolidate

# Clean up old results (keep last 7 days)
python manage_results.py clean --days 7
```

### Result Files Structure

```
benchmark_results/
├── benchmark_cache.json              # Cache metadata
├── json_accuracy_gemma3n_20241201_143022.csv
├── tool_calling_smollm2_360m_20241201_143045.csv
├── performance_llama3_2_1b_20241201_143108.csv
├── consolidated_all_20241201_150000.csv
└── benchmark_summary_20241201_150000.csv
```

## Example Usage Scenarios

### Scenario 1: Quick Model Comparison

```bash
# Compare JSON accuracy across models
python master_benchmark.py \
  --models gemma3n smollm2:360m llama3.2:1b \
  --test-types json_accuracy \
  --json-requests 50 \
  --llm-api ollama
```

### Scenario 2: Comprehensive Evaluation with Caching

```bash
# First run - complete benchmark
python master_benchmark.py --models gemma3n smollm2:360m --llm-api ollama

# Second run - only new tests (cached results skipped)
python master_benchmark.py --models gemma3n smollm2:360m llama3.2:1b --llm-api ollama --skip-cached
```

### Scenario 3: Performance-Focused Testing

```bash
# High-volume performance testing
python token_benchmark_ray.py \
  --model gemma3n \
  --llm-api ollama \
  --max-num-completed-requests 200 \
  --num-concurrent-requests 5 \
  --mean-input-tokens 1000 \
  --mean-output-tokens 500
```

## Metrics Explained

### Performance Metrics
- **TTFT (Time to First Token)**: Latency until first token is received
- **TPS (Tokens Per Second)**: Generation speed during streaming
- **End-to-End Latency**: Total request completion time
- **Inter-token Latency**: Average time between tokens

### Accuracy Metrics
- **JSON Valid Rate**: Percentage of responses with valid JSON
- **Schema Valid Rate**: Percentage meeting JSON schema requirements
- **Tool Call Success Rate**: Percentage of successful function calls
- **Parameter Accuracy**: Correctness of function call parameters

## Advanced Features

### Custom JSON Schemas

Add your own test cases to `json_benchmark_ray.py`:

```python
CUSTOM_TEST_CASE = {
    "prompt": "Generate a JSON object for a user profile",
    "schema": {
        "type": "object",
        "properties": {
            "username": {"type": "string"},
            "age": {"type": "integer", "minimum": 0},
            "preferences": {
                "type": "array",
                "items": {"type": "string"}
            }
        },
        "required": ["username", "age"]
    }
}
```

### Custom Tool Definitions

Add new tools to `tool_calling_benchmark_ray.py`:

```python
CUSTOM_TOOL = Tool(
    type="function",
    function=ToolFunction(
        name="custom_function",
        description="Your custom function description",
        parameters={
            "type": "object",
            "properties": {
                "param1": {"type": "string"},
                "param2": {"type": "number"}
            },
            "required": ["param1"]
        }
    )
)
```

## Troubleshooting

### Common Issues

1. **Ollama Connection Error**
   ```bash
   # Check if Ollama is running
   curl http://localhost:11434/api/tags
   
   # Start Ollama if needed
   ollama serve
   ```

2. **Model Not Found**
   ```bash
   # Pull the model first
   ollama pull gemma3n
   ```

3. **Import Errors**
   ```bash
   # Install missing dependencies
   pip install jsonschema pandas tqdm transformers
   ```

4. **Cache Issues**
   ```bash
   # Clear cache if needed
   python manage_results.py clear-cache
   ```

## Performance Tips

1. **Use Caching**: Always use `--skip-cached` for iterative testing
2. **Concurrent Requests**: Increase `--num-concurrent-requests` for throughput testing
3. **Batch Processing**: Use larger `--max-num-completed-requests` for statistical significance
4. **Resource Monitoring**: Monitor GPU/CPU usage during benchmarks

## Contributing

To add support for new APIs or test types:

1. Create a new client in `src/llmperf/ray_clients/`
2. Add the API to `SUPPORTED_APIS` in `common.py`
3. Update `construct_clients()` function
4. Add test configurations to master benchmark script

## Results Analysis

The CSV results can be analyzed using pandas:

```python
import pandas as pd

# Load consolidated results
df = pd.read_csv('benchmark_results/consolidated_all_20241201_150000.csv')

# Analyze by model
model_performance = df.groupby('model').agg({
    'ttft_s': 'mean',
    'accuracy_score': 'mean',
    'json_valid': 'mean'
})

print(model_performance)
```
