#!/usr/bin/env python3
"""
Master benchmark script for testing multiple models with various test types.
Supports performance, JSON accuracy, and tool calling benchmarks.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd
import ray

# Add the src directory to the path so we can import llmperf modules
sys.path.insert(0, str(Path(__file__).parent / "src"))
from llmperf.csv_results_manager import CSVResultsManager


# Default models to test
DEFAULT_MODELS = [
    "gemma3n",
    "smollm2:360m", 
    "llama3.2:1b"
]

# Test configurations
TEST_CONFIGS = {
    "performance": {
        "script": "token_benchmark_ray.py",
        "params": {
            "mean-input-tokens": 550,
            "stddev-input-tokens": 150,
            "mean-output-tokens": 150,
            "stddev-output-tokens": 10,
            "max-num-completed-requests": 50,
            "num-concurrent-requests": 1,
            "timeout": 300
        }
    },
    "json_accuracy": {
        "script": "json_benchmark_ray.py",
        "params": {
            "max-num-completed-requests": 30,
            "num-concurrent-requests": 1,
            "timeout": 300
        }
    },
    "tool_calling": {
        "script": "tool_calling_benchmark_ray.py",
        "params": {
            "max-num-completed-requests": 20,
            "num-concurrent-requests": 1,
            "timeout": 300
        }
    }
}


class MasterBenchmark:
    def __init__(self, models: List[str], llm_api: str = "ollama", results_dir: str = "benchmark_results",
                 skip_cached: bool = False):
        self.models = models
        self.llm_api = llm_api
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.results = {}
        self.skip_cached = skip_cached
        self.csv_manager = CSVResultsManager(results_dir=results_dir)
        
    def run_single_benchmark(self, model: str, test_type: str, additional_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Run a single benchmark test."""
        print(f"\n{'='*60}")
        print(f"Running {test_type} benchmark for {model}")
        print(f"{'='*60}")

        config = TEST_CONFIGS[test_type]

        # Check if test is already completed and should be skipped
        test_config = config["params"].copy()
        if additional_params:
            test_config.update(additional_params)
        test_config["llm_api"] = self.llm_api

        if self.skip_cached and self.csv_manager.is_test_completed(model, test_type, test_config):
            print(f"✅ Test already completed for {model} - {test_type}. Skipping...")
            existing_results = self.csv_manager.load_existing_results(model, test_type)
            if existing_results is not None:
                print(f"📊 Found {len(existing_results)} existing results")
            return {
                "status": "skipped_cached",
                "duration": 0,
                "message": "Test already completed with same configuration"
            }

        script_path = Path(__file__).parent / config["script"]
        
        # Build command
        cmd = [
            "python", str(script_path),
            "--model", model,
            "--llm-api", self.llm_api,
            "--results-dir", str(self.results_dir)
        ]

        # Add skip-if-cached flag if enabled
        if self.skip_cached:
            cmd.append("--skip-if-cached")

        # Add test-specific parameters
        params = config["params"].copy()
        if additional_params:
            params.update(additional_params)

        for key, value in params.items():
            cmd.extend([f"--{key}", str(value)])
        
        # Run the benchmark
        start_time = time.time()
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            end_time = time.time()
            
            if result.returncode == 0:
                print(f"✅ {test_type} benchmark completed successfully")
                return {
                    "status": "success",
                    "duration": end_time - start_time,
                    "stdout": result.stdout,
                    "stderr": result.stderr
                }
            else:
                print(f"❌ {test_type} benchmark failed")
                print(f"Error: {result.stderr}")
                return {
                    "status": "failed",
                    "duration": end_time - start_time,
                    "error": result.stderr,
                    "stdout": result.stdout
                }
        except subprocess.TimeoutExpired:
            print(f"⏰ {test_type} benchmark timed out")
            return {
                "status": "timeout",
                "duration": 600,
                "error": "Benchmark timed out after 10 minutes"
            }
        except Exception as e:
            print(f"💥 {test_type} benchmark crashed: {str(e)}")
            return {
                "status": "crashed",
                "duration": time.time() - start_time,
                "error": str(e)
            }
    
    def run_all_benchmarks(self, test_types: List[str] = None) -> Dict[str, Any]:
        """Run all benchmarks for all models."""
        if test_types is None:
            test_types = list(TEST_CONFIGS.keys())
        
        print(f"🚀 Starting master benchmark run")
        print(f"Models: {', '.join(self.models)}")
        print(f"Test types: {', '.join(test_types)}")
        print(f"API: {self.llm_api}")
        print(f"Results directory: {self.results_dir}")
        
        overall_start = time.time()
        
        for model in self.models:
            self.results[model] = {}
            
            for test_type in test_types:
                result = self.run_single_benchmark(model, test_type)
                self.results[model][test_type] = result
        
        overall_end = time.time()
        
        # Generate summary report
        self.generate_summary_report(overall_end - overall_start)
        
        return self.results
    
    def generate_summary_report(self, total_duration: float):
        """Generate a comprehensive summary report."""
        print(f"\n{'='*80}")
        print("📊 BENCHMARK SUMMARY REPORT")
        print(f"{'='*80}")
        
        # Create summary data
        summary_data = []
        
        for model in self.models:
            for test_type, result in self.results[model].items():
                summary_data.append({
                    "model": model,
                    "test_type": test_type,
                    "status": result["status"],
                    "duration_s": round(result["duration"], 2),
                    "success": result["status"] == "success"
                })
        
        # Create DataFrame for easy analysis
        df = pd.DataFrame(summary_data)
        
        # Print summary table
        print("\n📋 Test Results Overview:")
        print(df.to_string(index=False))
        
        # Calculate success rates
        print(f"\n📈 Success Rates:")
        success_by_model = df.groupby("model")["success"].mean()
        for model, rate in success_by_model.items():
            print(f"  {model}: {rate:.1%}")
        
        success_by_test = df.groupby("test_type")["success"].mean()
        for test_type, rate in success_by_test.items():
            print(f"  {test_type}: {rate:.1%}")
        
        overall_success = df["success"].mean()
        print(f"  Overall: {overall_success:.1%}")
        
        # Print timing info
        print(f"\n⏱️  Timing:")
        print(f"  Total duration: {total_duration:.1f}s ({total_duration/60:.1f}m)")
        print(f"  Average per test: {df['duration_s'].mean():.1f}s")
        
        # Save detailed results
        timestamp = int(time.time())
        
        # Save summary CSV
        csv_path = self.results_dir / f"benchmark_summary_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        
        # Save detailed JSON
        json_path = self.results_dir / f"benchmark_detailed_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump({
                "summary": {
                    "total_duration": total_duration,
                    "models": self.models,
                    "test_types": list(TEST_CONFIGS.keys()),
                    "overall_success_rate": overall_success,
                    "timestamp": timestamp
                },
                "results": self.results
            }, f, indent=2)
        
        print(f"\n💾 Results saved:")
        print(f"  Summary CSV: {csv_path}")
        print(f"  Detailed JSON: {json_path}")
        
        # Print failed tests
        failed_tests = df[df["status"] != "success"]
        if not failed_tests.empty:
            print(f"\n❌ Failed Tests:")
            for _, row in failed_tests.iterrows():
                print(f"  {row['model']} - {row['test_type']}: {row['status']}")


def main():
    parser = argparse.ArgumentParser(description="Master benchmark script for LLM performance and accuracy testing")
    
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, 
                       help="Models to test (default: gemma3n smollm2:360m llama3.2:1b)")
    parser.add_argument("--llm-api", type=str, default="ollama", 
                       choices=["ollama", "openai", "anthropic", "litellm"],
                       help="LLM API to use (default: ollama)")
    parser.add_argument("--test-types", nargs="+", default=list(TEST_CONFIGS.keys()),
                       choices=list(TEST_CONFIGS.keys()),
                       help="Test types to run (default: all)")
    parser.add_argument("--results-dir", type=str, default="benchmark_results",
                       help="Directory to save results (default: benchmark_results)")
    parser.add_argument("--performance-requests", type=int, default=50,
                       help="Number of requests for performance test")
    parser.add_argument("--json-requests", type=int, default=30,
                       help="Number of requests for JSON accuracy test")
    parser.add_argument("--tool-requests", type=int, default=20,
                       help="Number of requests for tool calling test")
    parser.add_argument("--skip-cached", action="store_true",
                       help="Skip tests that have already been completed with same configuration")

    args = parser.parse_args()
    
    # Update test configs with custom request counts
    TEST_CONFIGS["performance"]["params"]["max-num-completed-requests"] = args.performance_requests
    TEST_CONFIGS["json_accuracy"]["params"]["max-num-completed-requests"] = args.json_requests
    TEST_CONFIGS["tool_calling"]["params"]["max-num-completed-requests"] = args.tool_requests
    
    # Initialize Ray
    ray.init(ignore_reinit_error=True)
    
    # Create and run benchmark
    benchmark = MasterBenchmark(
        models=args.models,
        llm_api=args.llm_api,
        results_dir=args.results_dir,
        skip_cached=args.skip_cached
    )
    
    try:
        benchmark.run_all_benchmarks(args.test_types)
        print(f"\n🎉 Master benchmark completed!")
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Benchmark interrupted by user")
    except Exception as e:
        print(f"\n💥 Benchmark failed with error: {str(e)}")
        raise
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
