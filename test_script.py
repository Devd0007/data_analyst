#!/usr/bin/env python3
"""
Test script for the LLM Data Analyst Agent
Run this to verify your deployment is working correctly
"""

import requests
import json
import time

def test_api(base_url):
    """Test the data analyst API with various scenarios"""
    
    print(f"🧪 Testing Data Analyst Agent at: {base_url}")
    print("=" * 60)
    
    # Test 1: Health Check
    print("\n1️⃣ Testing Health Check...")
    try:
        response = requests.get(f"{base_url}/")
        if response.status_code == 200:
            print("✅ Health check passed")
            print(f"   Status: {response.json().get('status', 'Unknown')}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Health check error: {e}")
    
    # Test 2: Test Endpoint
    print("\n2️⃣ Testing Test Endpoint...")
    try:
        response = requests.get(f"{base_url}/test")
        if response.status_code == 200:
            print("✅ Test endpoint passed")
            data = response.json()
            print(f"   Datasets loaded: {data.get('datasets_loaded', 0)}")
            print(f"   Questions parsed: {data.get('questions_parsed', 0)}")
        else:
            print(f"❌ Test endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Test endpoint error: {e}")
    
    # Test 3: Movie Analysis (JSON Array Format)
    print("\n3️⃣ Testing Movie Analysis (JSON Array)...")
    movie_question = """Scrape the list of highest grossing films from Wikipedia. It is at the URL:
https://en.wikipedia.org/wiki/List_of_highest-grossing_films

Answer the following questions and respond with a JSON array of strings containing the answer.

1. How many $2 bn movies were released before 2020?
2. Which is the earliest film that grossed over $1.5 bn?
3. What's the correlation between the Rank and Peak?
4. Draw a scatterplot of Rank and Peak along with a dotted red regression line through it.
   Return as a base-64 encoded data URI, `"data:image/png;base64,iVBORw0KG..."` under 100,000 bytes."""
    
    try:
        response = requests.post(
            f"{base_url}/api/",
            data=movie_question,
            headers={'Content-Type': 'text/plain'},
            timeout=180  # 3 minutes
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Movie analysis completed")
            print(f"   Response type: {type(result)}")
            print(f"   Response length: {len(result) if isinstance(result, list) else 'N/A'}")
            
            if isinstance(result, list) and len(result) >= 4:
                print(f"   Answer 1 (count): {result[0]}")
                print(f"   Answer 2 (movie): {result[1]}")
                print(f"   Answer 3 (correlation): {result[2]}")
                print(f"   Answer 4 (plot): {'Base64 image' if result[3].startswith('data:image') else 'Invalid'}")
                
                # Check image size
                if result[3].startswith('data:image'):
                    import base64
                    header, data = result[3].split(',', 1)
                    image_bytes = base64.b64decode(data)
                    print(f"   Image size: {len(image_bytes)} bytes ({'✅' if len(image_bytes) < 100000 else '❌'} <100KB)")
            else:
                print(f"   ❌ Unexpected response format: {result}")
        else:
            print(f"❌ Movie analysis failed: {response.status_code}")
            print(f"   Response: {response.text[:200]}...")
    except Exception as e:
        print(f"❌ Movie analysis error: {e}")
    
    # Test 4: Court Data Analysis (JSON Object Format)
    print("\n4️⃣ Testing Court Data Analysis (JSON Object)...")
    court_question = """The Indian high court judgement dataset contains judgements from the Indian High Courts.

Answer the following questions and respond with a JSON object containing the answer.

{
  "Which high court disposed the most cases from 2019 - 2022?": "...",
  "What's the regression slope of the date_of_registration - decision_date by year in the court=33_10?": "...",
  "Plot the year and # of days of delay from the above question as a scatterplot with a regression line. Encode as a base64 data URI under 100,000 characters": "data:image/webp:base64,..."
}"""
    
    try:
        response = requests.post(
            f"{base_url}/api/",
            json={"question": court_question},
            timeout=180
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Court analysis completed")
            print(f"   Response type: {type(result)}")
            
            if isinstance(result, dict):
                for key, value in result.items():
                    if key.endswith('?'):
                        print(f"   {key}: {value}")
                    elif 'plot' in key.lower() or 'chart' in key.lower():
                        print(f"   Visualization: {'✅ Valid' if str(value).startswith('data:image') else '❌ Invalid'}")
            else:
                print(f"   Response: {result}")
        else:
            print(f"❌ Court analysis failed: {response.status_code}")
            print(f"   Response: {response.text[:200]}...")
    except Exception as e:
        print(f"❌ Court analysis error: {e}")
    
    # Test 5: Generic Analysis
    print("\n5️⃣ Testing Generic Analysis...")
    generic_question = "Analyze some sample data and create a visualization with regression analysis"
    
    try:
        response = requests.post(
            f"{base_url}/api/",
            data=generic_question,
            headers={'Content-Type': 'text/plain'},
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Generic analysis completed")
            print(f"   Response: {str(result)[:100]}...")
        else:
            print(f"❌ Generic analysis failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Generic analysis error: {e}")
    
    print("\n" + "=" * 60)
    print("🎉 Testing completed!")
    print("\n💡 Tips for deployment:")
    print("   - Make sure your repl stays awake during evaluation")
    print("   - Test the /api/ endpoint with curl before submission")
    print("   - Verify images are under 100KB and properly base64 encoded")
    print("   - Check that JSON responses match expected format exactly")

def main():
    """Main test function"""
    print("🚀 LLM Data Analyst Agent Tester")
    print("Enter your API base URL (without /api/ at the end)")
    print("Example: https://your-repl.your-username.repl.co")
    
    base_url = input("\nAPI Base URL: ").strip().rstrip('/')
    
    if not base_url:
        print("❌ No URL provided, using localhost for testing")
        base_url = "http://localhost:5000"
    
    if not base_url.startswith('http'):
        base_url = 'https://' + base_url
    
    print(f"\n🔗 Testing: {base_url}")
    
    # Give user a chance to start their server
    input("Press Enter when your server is running...")
    
    test_api(base_url)

if __name__ == "__main__":
    main()