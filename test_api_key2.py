#!/usr/bin/env python3
"""Test if Groq API key is accessible and working."""

import os

print("=" * 60)
print("API KEY DIAGNOSTIC")
print("=" * 60)

# Check environment variables
print("\n1. Environment Variables:")
groq_key = os.getenv("GROQ_API_KEY")
openai_key = os.getenv("OPENAI_API_KEY")

print(f"   GROQ_API_KEY: {'SET (' + groq_key[:10] + '...)' if groq_key else 'NOT SET'}")
print(f"   OPENAI_API_KEY: {'SET (' + openai_key[:10] + '...)' if openai_key else 'NOT SET'}")

# Check packages
print("\n2. Package Availability:")
try:
    import groq
    print("   groq: INSTALLED")
except ImportError:
    print("   groq: NOT INSTALLED (run: pip install groq)")

try:
    import openai
    print("   openai: INSTALLED")
except ImportError:
    print("   openai: NOT INSTALLED")

# Test Groq
print("\n3. API Connection Test:")
if groq_key:
    try:
        from groq import Groq
        client = Groq(api_key=groq_key)
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": "Say PASS in one word."}],
            temperature=0.0,
            max_tokens=10,
        )
        print(f"   Groq: WORKING -> '{resp.choices[0].message.content.strip()}'")
    except Exception as e:
        print(f"   Groq: FAILED -> {e}")
else:
    print("   Groq: SKIPPED (no API key)")

# Test OpenAI
if openai_key:
    try:
        from openai import OpenAI
        import httpx
        client = OpenAI(api_key=openai_key, http_client=httpx.Client())
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Say PASS in one word."}],
            temperature=0.0,
            max_tokens=10,
        )
        print(f"   OpenAI: WORKING -> '{resp.choices[0].message.content.strip()}'")
    except Exception as e:
        print(f"   OpenAI: FAILED -> {e}")
else:
    print("   OpenAI: SKIPPED (no API key)")

print("\n" + "=" * 60)