#!/usr/bin/env python3
"""
Example usage script for the enhanced LLMPerf with structured output benchmarking.
This script demonstrates how to run benchmarks for the specified models.
"""

import subprocess
import sys
import time
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Success!")
        if result.stdout:
            print("Output:", result.stdout[-500:])  # Show last 500 chars
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed with return code {e.returncode}")
        if e.stderr:
            print("Error:", e.stderr)
        return False
    except Exception as e:
        print(f"💥 Exception: {e}")
        return False


def check_ollama():
    """Check if Ollama is running and models are available."""
    print("🔍 Checking Ollama setup...")
    
    try:
        # Check if Ollama is running
        result = subprocess.run(["curl", "-s", "http://localhost:11434/api/tags"], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode != 0:
            print("❌ Ollama is not running. Please start it with: ollama serve")
            return False
        
        print("✅ Ollama is running")
        
        # Check for required models
        models_to_check = ["gemma3n", "smollm2:360m", "llama3.2:1b"]
        available_models = []
        
        for model in models_to_check:
            result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
            if model in result.stdout:
                available_models.append(model)
                print(f"✅ Model {model} is available")
            else:
                print(f"⚠️  Model {model} not found. You can pull it with: ollama pull {model}")
        
        return len(available_models) > 0, available_models
        
    except Exception as e:
        print(f"❌ Error checking Ollama: {e}")
        return False, []


def main():
    print("🎯 LLMPerf Enhanced Benchmarking Example")
    print("This script demonstrates benchmarking JSON structured outputs and tool calls")
    
    # Check Ollama setup
    ollama_ok, available_models = check_ollama()
    if not ollama_ok:
        print("\n❌ Ollama setup issues detected. Please fix them before continuing.")
        return
    
    if not available_models:
        print("\n❌ No models available. Please pull at least one model:")
        print("  ollama pull gemma3n")
        print("  ollama pull smollm2:360m") 
        print("  ollama pull llama3.2:1b")
        return
    
    print(f"\n✅ Found {len(available_models)} available models: {', '.join(available_models)}")
    
    # Use the first available model for examples
    test_model = available_models[0]
    print(f"\n🎯 Using {test_model} for demonstration")
    
    # Example 1: JSON Accuracy Benchmark
    cmd1 = [
        "python", "json_benchmark_ray.py",
        "--model", test_model,
        "--llm-api", "ollama",
        "--max-num-completed-requests", "10",
        "--results-dir", "example_results"
    ]
    
    success1 = run_command(cmd1, f"JSON Accuracy Benchmark for {test_model}")
    
    # Example 2: Tool Calling Benchmark
    cmd2 = [
        "python", "tool_calling_benchmark_ray.py", 
        "--model", test_model,
        "--llm-api", "ollama",
        "--max-num-completed-requests", "5",
        "--results-dir", "example_results"
    ]
    
    success2 = run_command(cmd2, f"Tool Calling Benchmark for {test_model}")
    
    # Example 3: Performance Benchmark
    cmd3 = [
        "python", "token_benchmark_ray.py",
        "--model", test_model,
        "--llm-api", "ollama", 
        "--max-num-completed-requests", "10",
        "--mean-input-tokens", "100",
        "--mean-output-tokens", "50",
        "--results-dir", "example_results"
    ]
    
    success3 = run_command(cmd3, f"Performance Benchmark for {test_model}")
    
    # Example 4: Master Benchmark (if we have multiple models)
    if len(available_models) > 1:
        models_arg = available_models[:2]  # Use first 2 models
        cmd4 = [
            "python", "master_benchmark.py",
            "--models"] + models_arg + [
            "--llm-api", "ollama",
            "--performance-requests", "5",
            "--json-requests", "5", 
            "--tool-requests", "3",
            "--results-dir", "example_results"
        ]
        
        success4 = run_command(cmd4, f"Master Benchmark for {', '.join(models_arg)}")
    else:
        success4 = True
        print(f"\n⏭️  Skipping master benchmark (only one model available)")
    
    # Results summary
    print(f"\n{'='*60}")
    print("📊 EXAMPLE RUN SUMMARY")
    print(f"{'='*60}")
    
    results = [
        ("JSON Accuracy", success1),
        ("Tool Calling", success2), 
        ("Performance", success3),
        ("Master Benchmark", success4 if len(available_models) > 1 else None)
    ]
    
    for test_name, success in results:
        if success is None:
            print(f"  {test_name}: ⏭️  Skipped")
        elif success:
            print(f"  {test_name}: ✅ Success")
        else:
            print(f"  {test_name}: ❌ Failed")
    
    # Show results management
    print(f"\n📁 Results saved to: example_results/")
    print("You can manage results with:")
    print("  python manage_results.py --results-dir example_results stats")
    print("  python manage_results.py --results-dir example_results list")
    print("  python manage_results.py --results-dir example_results consolidate")
    
    # Show next steps
    print(f"\n🎯 Next Steps:")
    print("1. Check the results in the example_results/ directory")
    print("2. Try running with --skip-cached to see caching in action")
    print("3. Experiment with different models and parameters")
    print("4. Use the master benchmark for comprehensive testing")
    
    print(f"\n✨ Example completed! Check STRUCTURED_OUTPUT_BENCHMARKING.md for more details.")


if __name__ == "__main__":
    main()
