import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import pandas as pd
from datetime import datetime


class CSVResultsManager:
    """
    Manages CSV results with caching capabilities to skip already generated data on reruns.
    Supports individual record tracking as preferred by users.
    """
    
    def __init__(self, results_dir: str = "benchmark_results", cache_file: str = "benchmark_cache.json"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.cache_file = self.results_dir / cache_file
        self.cache = self._load_cache()
        
    def _load_cache(self) -> Dict[str, Any]:
        """Load existing cache from file."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                return {"completed_tests": {}, "file_hashes": {}}
        return {"completed_tests": {}, "file_hashes": {}}
    
    def _save_cache(self):
        """Save cache to file."""
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f, indent=2)
    
    def _generate_test_hash(self, model: str, test_type: str, config: Dict[str, Any]) -> str:
        """Generate a unique hash for a test configuration."""
        test_key = {
            "model": model,
            "test_type": test_type,
            "config": config
        }
        test_str = json.dumps(test_key, sort_keys=True)
        return hashlib.md5(test_str.encode()).hexdigest()
    
    def is_test_completed(self, model: str, test_type: str, config: Dict[str, Any]) -> bool:
        """Check if a test has already been completed with the same configuration."""
        test_hash = self._generate_test_hash(model, test_type, config)
        return test_hash in self.cache["completed_tests"]
    
    def mark_test_completed(self, model: str, test_type: str, config: Dict[str, Any], 
                          csv_file: str, num_records: int):
        """Mark a test as completed in the cache."""
        test_hash = self._generate_test_hash(model, test_type, config)
        self.cache["completed_tests"][test_hash] = {
            "model": model,
            "test_type": test_type,
            "config": config,
            "csv_file": csv_file,
            "num_records": num_records,
            "timestamp": datetime.now().isoformat(),
            "completed": True
        }
        self._save_cache()
    
    def get_csv_filename(self, model: str, test_type: str, timestamp: Optional[str] = None) -> str:
        """Generate standardized CSV filename."""
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        safe_model = model.replace("/", "_").replace(":", "_")
        return f"{test_type}_{safe_model}_{timestamp}.csv"
    
    def save_individual_results_csv(self, results: List[Dict[str, Any]], model: str, 
                                  test_type: str, config: Dict[str, Any]) -> str:
        """
        Save individual benchmark results to CSV with proper structure.
        Returns the filename of the saved CSV.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = self.get_csv_filename(model, test_type, timestamp)
        csv_path = self.results_dir / csv_filename
        
        if not results:
            print(f"Warning: No results to save for {model} - {test_type}")
            return csv_filename
        
        # Flatten nested dictionaries and prepare data for CSV
        flattened_results = []
        for result in results:
            flattened = self._flatten_dict(result)
            # Add metadata
            flattened.update({
                "model": model,
                "test_type": test_type,
                "timestamp": timestamp,
                "config_hash": self._generate_test_hash(model, test_type, config)
            })
            flattened_results.append(flattened)
        
        # Write to CSV
        if flattened_results:
            df = pd.DataFrame(flattened_results)
            df.to_csv(csv_path, index=False)
            
            print(f"✅ Saved {len(flattened_results)} individual results to {csv_filename}")
            
            # Mark test as completed
            self.mark_test_completed(model, test_type, config, csv_filename, len(flattened_results))
        
        return csv_filename
    
    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
        """Flatten nested dictionary for CSV export."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, list):
                # Convert lists to JSON strings for CSV
                items.append((new_key, json.dumps(v) if v else ""))
            else:
                items.append((new_key, v))
        return dict(items)
    
    def load_existing_results(self, model: str, test_type: str) -> Optional[pd.DataFrame]:
        """Load existing results for a model and test type."""
        # Find the most recent CSV file for this model and test type
        pattern = f"{test_type}_{model.replace('/', '_').replace(':', '_')}_*.csv"
        csv_files = list(self.results_dir.glob(pattern))
        
        if not csv_files:
            return None
        
        # Get the most recent file
        latest_file = max(csv_files, key=lambda x: x.stat().st_mtime)
        
        try:
            df = pd.read_csv(latest_file)
            print(f"📂 Loaded {len(df)} existing results from {latest_file.name}")
            return df
        except Exception as e:
            print(f"⚠️  Error loading existing results from {latest_file}: {e}")
            return None
    
    def get_completed_tests_summary(self) -> pd.DataFrame:
        """Get a summary of all completed tests."""
        completed = []
        for test_hash, test_info in self.cache["completed_tests"].items():
            completed.append({
                "model": test_info["model"],
                "test_type": test_info["test_type"],
                "num_records": test_info["num_records"],
                "csv_file": test_info["csv_file"],
                "timestamp": test_info["timestamp"],
                "test_hash": test_hash
            })
        
        return pd.DataFrame(completed) if completed else pd.DataFrame()
    
    def consolidate_results(self, test_type: Optional[str] = None) -> str:
        """
        Consolidate all individual CSV files into a single comprehensive CSV.
        Returns the filename of the consolidated CSV.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if test_type:
            pattern = f"{test_type}_*.csv"
            consolidated_filename = f"consolidated_{test_type}_{timestamp}.csv"
        else:
            pattern = "*.csv"
            consolidated_filename = f"consolidated_all_{timestamp}.csv"
        
        csv_files = [f for f in self.results_dir.glob(pattern) 
                    if not f.name.startswith("consolidated_") and not f.name.startswith("benchmark_summary")]
        
        if not csv_files:
            print(f"No CSV files found to consolidate")
            return consolidated_filename
        
        # Load and combine all CSV files
        all_data = []
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                df['source_file'] = csv_file.name
                all_data.append(df)
            except Exception as e:
                print(f"⚠️  Error reading {csv_file}: {e}")
        
        if all_data:
            consolidated_df = pd.concat(all_data, ignore_index=True, sort=False)
            consolidated_path = self.results_dir / consolidated_filename
            consolidated_df.to_csv(consolidated_path, index=False)
            
            print(f"✅ Consolidated {len(all_data)} files into {consolidated_filename}")
            print(f"   Total records: {len(consolidated_df)}")
            
            return consolidated_filename
        
        return consolidated_filename
    
    def clean_old_results(self, keep_days: int = 7):
        """Clean up old result files, keeping only recent ones."""
        import time
        cutoff_time = time.time() - (keep_days * 24 * 60 * 60)
        
        cleaned_files = []
        for csv_file in self.results_dir.glob("*.csv"):
            if csv_file.stat().st_mtime < cutoff_time:
                csv_file.unlink()
                cleaned_files.append(csv_file.name)
        
        if cleaned_files:
            print(f"🧹 Cleaned up {len(cleaned_files)} old result files")
        
        # Also clean up cache entries for deleted files
        to_remove = []
        for test_hash, test_info in self.cache["completed_tests"].items():
            csv_file = self.results_dir / test_info["csv_file"]
            if not csv_file.exists():
                to_remove.append(test_hash)
        
        for test_hash in to_remove:
            del self.cache["completed_tests"][test_hash]
        
        if to_remove:
            self._save_cache()
            print(f"🧹 Cleaned up {len(to_remove)} stale cache entries")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about the cache."""
        completed_tests = self.cache["completed_tests"]
        
        stats = {
            "total_completed_tests": len(completed_tests),
            "models_tested": len(set(test["model"] for test in completed_tests.values())),
            "test_types": len(set(test["test_type"] for test in completed_tests.values())),
            "total_records": sum(test["num_records"] for test in completed_tests.values()),
            "cache_file_size": self.cache_file.stat().st_size if self.cache_file.exists() else 0
        }
        
        return stats
