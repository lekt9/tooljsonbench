#!/usr/bin/env python3
"""
Quick test script to verify Ollama client functionality.
"""

import sys
import os
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import ray
from llmperf.models import RequestConfig
from llmperf.ray_clients.ollama_client import OllamaClient


def test_ollama_client():
    """Test basic Ollama client functionality."""
    print("🧪 Testing Ollama Client")
    
    # Initialize Ray
    ray.init(ignore_reinit_error=True)
    
    try:
        # Create client
        client = OllamaClient.remote()
        
        # Test prompt
        prompt_text = "Hello, how are you?"
        prompt_len = len(prompt_text.split())  # Simple word count
        
        # Create request config
        request_config = RequestConfig(
            model="gemma3n",
            prompt=(prompt_text, prompt_len),
            sampling_params={
                "max_tokens": 50,
                "temperature": 0.7
            },
            llm_api="ollama"
        )
        
        print(f"📤 Sending request to model: {request_config.model}")
        print(f"📝 Prompt: {prompt_text}")
        
        # Make request
        result = ray.get(client.llm_request.remote(request_config))
        metrics, generated_text, config = result
        
        print(f"✅ Request completed!")
        print(f"📊 Metrics: {metrics}")
        print(f"📄 Generated text: {generated_text}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
    
    finally:
        ray.shutdown()


if __name__ == "__main__":
    success = test_ollama_client()
    if success:
        print("🎉 Ollama client test passed!")
    else:
        print("💥 Ollama client test failed!")
        sys.exit(1)
