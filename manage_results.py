#!/usr/bin/env python3
"""
Utility script for managing benchmark results and cache.
"""

import argparse
import sys
from pathlib import Path

# Add the src directory to the path so we can import llmperf modules
sys.path.insert(0, str(Path(__file__).parent / "src"))

from llmperf.csv_results_manager import CSVResultsManager


def show_cache_stats(csv_manager: CSVResultsManager):
    """Show cache statistics."""
    stats = csv_manager.get_cache_stats()
    
    print("📊 Cache Statistics:")
    print(f"  Total completed tests: {stats['total_completed_tests']}")
    print(f"  Models tested: {stats['models_tested']}")
    print(f"  Test types: {stats['test_types']}")
    print(f"  Total records: {stats['total_records']}")
    print(f"  Cache file size: {stats['cache_file_size']} bytes")


def show_completed_tests(csv_manager: CSVResultsManager):
    """Show completed tests summary."""
    df = csv_manager.get_completed_tests_summary()
    
    if df.empty:
        print("No completed tests found.")
        return
    
    print("✅ Completed Tests:")
    print(df.to_string(index=False))


def consolidate_results(csv_manager: CSVResultsManager, test_type: str = None):
    """Consolidate results into a single CSV."""
    print(f"🔄 Consolidating results...")
    filename = csv_manager.consolidate_results(test_type)
    print(f"✅ Consolidated results saved to: {filename}")


def clean_old_results(csv_manager: CSVResultsManager, days: int):
    """Clean up old result files."""
    print(f"🧹 Cleaning up results older than {days} days...")
    csv_manager.clean_old_results(keep_days=days)
    print("✅ Cleanup completed")


def main():
    parser = argparse.ArgumentParser(description="Manage benchmark results and cache")
    
    parser.add_argument("--results-dir", type=str, default="benchmark_results",
                       help="Results directory (default: benchmark_results)")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Stats command
    subparsers.add_parser("stats", help="Show cache statistics")
    
    # List command
    subparsers.add_parser("list", help="List completed tests")
    
    # Consolidate command
    consolidate_parser = subparsers.add_parser("consolidate", help="Consolidate results into single CSV")
    consolidate_parser.add_argument("--test-type", type=str, 
                                  choices=["performance", "json_accuracy", "tool_calling"],
                                  help="Consolidate only specific test type")
    
    # Clean command
    clean_parser = subparsers.add_parser("clean", help="Clean up old result files")
    clean_parser.add_argument("--days", type=int, default=7,
                            help="Keep files newer than this many days (default: 7)")
    
    # Clear cache command
    subparsers.add_parser("clear-cache", help="Clear the entire cache")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize CSV manager
    csv_manager = CSVResultsManager(results_dir=args.results_dir)
    
    if args.command == "stats":
        show_cache_stats(csv_manager)
    
    elif args.command == "list":
        show_completed_tests(csv_manager)
    
    elif args.command == "consolidate":
        consolidate_results(csv_manager, args.test_type)
    
    elif args.command == "clean":
        clean_old_results(csv_manager, args.days)
    
    elif args.command == "clear-cache":
        cache_file = csv_manager.cache_file
        if cache_file.exists():
            cache_file.unlink()
            print("✅ Cache cleared")
        else:
            print("ℹ️  No cache file found")


if __name__ == "__main__":
    main()
