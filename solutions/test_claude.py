#!/usr/bin/env python3
"""
Test script for Claude-3.5-Sonnet integration
"""

import os
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def test_claude():
    """Test Claude-3.5-Sonnet model"""
    
    # Get API key from environment (checking multiple possible names)
    api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("API_KEY")
    
    if not api_key:
        print("❌ No API key found. Please set ANTHROPIC_API_KEY or API_KEY in your .env file")
        return
    
    # Set the API key
    os.environ["ANTHROPIC_API_KEY"] = api_key
    
    # Create Claude model
    model = ChatAnthropic(
        model="claude-3-5-sonnet-20241022",
        temperature=0
    )
    
    # Test message
    response = model.invoke("What is 2+2? Reply in one sentence.")
    
    print(f"✅ Claude response: {response.content}")

if __name__ == "__main__":
    print("🧪 Testing Claude-3.5-Sonnet integration...")
    print("🔍 Looking for API key in environment variables...")
    
    # Check what keys are available
    if os.getenv("ANTHROPIC_API_KEY"):
        print("   ✅ Found ANTHROPIC_API_KEY")
    elif os.getenv("API_KEY"):
        print("   ✅ Found API_KEY (will use for Anthropic)")
    else:
        print("   ❌ No API key found in environment")
    
    test_claude() 