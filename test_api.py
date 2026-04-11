#!/usr/bin/env python3
"""
Quick test script for the LexGuard API.

Usage:
  python test_api.py --url http://localhost:8000

Make sure the API is running first:
  docker-compose up
  # or
  python -m src.api.app
"""

import sys
import argparse
import requests
import json

def test_health(base_url):
    """Test the health endpoint."""
    print("Testing /health endpoint...")
    try:
        resp = requests.get(f"{base_url}/health", timeout=5)
        resp.raise_for_status()
        data = resp.json()
        print(f"✓ Health check passed: {data['status']}")
        print(f"  Models: {json.dumps(data['loaded_models'], indent=2)}")
        return True
    except Exception as e:
        print(f"✗ Health check failed: {e}")
        return False


def test_compliance_check(base_url):
    """Test the compliance check endpoint."""
    print("\nTesting /compliance-check endpoint...")
    
    sample_text = """
    Our company collects personal data from users to improve our services.
    We store this data in secure servers and do not share it with third parties
    without explicit consent. Users can request access or deletion of their data
    in compliance with GDPR regulations.
    """
    
    payload = {
        "text": sample_text.strip(),
        "lambda_thresh": 0.75,
        "hop_k": 1,
        "max_triples": 60,
        "prefer_local": True,  # Use local model (faster for testing)
        "openai_model": "gpt-3.5-turbo"
    }
    
    try:
        resp = requests.post(
            f"{base_url}/compliance-check",
            json=payload,
            timeout=60
        )
        resp.raise_for_status()
        data = resp.json()
        print(f"✓ Compliance check completed")
        print(f"  Verdict: {data['verdict']}")
        print(f"  Evidence: {len(data['evidence'])} items")
        print(f"  Hits: {len(data['hits'])} nodes")
        print(f"  Triples:\n{data['triples_text'][:200]}...")
        return True
    except Exception as e:
        print(f"✗ Compliance check failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Test LexGuard API")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    args = parser.parse_args()
    
    print(f"Testing API at {args.url}\n")
    
    health_ok = test_health(args.url)
    if not health_ok:
        print("\n✗ API is not responding. Make sure it's running:")
        print("  docker-compose up")
        print("  # or")
        print("  python -m src.api.app")
        sys.exit(1)
    
    compliance_ok = test_compliance_check(args.url)
    
    print("\n" + "="*50)
    if health_ok and compliance_ok:
        print("✓ All tests passed!")
        sys.exit(0)
    else:
        print("✗ Some tests failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
