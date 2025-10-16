# test_apis.py - Run this to diagnose the issue
import os
import requests
import json

def test_newsapi():
    api_key = os.getenv("NEWSAPI_KEY")
    if not api_key:
        print("❌ NEWSAPI_KEY not found")
        return
    
    print(f"Testing NewsAPI with key: {api_key[:8]}...")
    
    # Test 1: Top Headlines
    url = f"https://newsapi.org/v2/top-headlines?country=us&apiKey={api_key}"
    response = requests.get(url)
    print(f"Top Headlines Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Articles found: {len(data.get('articles', []))}")
        if data.get('articles'):
            print(f"Sample headline: {data['articles'][0].get('title')}")
    else:
        print(f"❌ Error: {response.text}")
    
    # Test 2: Everything endpoint
    url = f"https://newsapi.org/v2/everything?q=war&apiKey={api_key}&pageSize=5"
    response = requests.get(url)
    print(f"\nEverything API Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Articles found: {len(data.get('articles', []))}")
    else:
        print(f"❌ Error: {response.text}")

def test_finnhub():
    api_key = os.getenv("FINNHUB_KEY")
    if not api_key:
        print("❌ FINNHUB_KEY not found")
        return
    
    print(f"\nTesting Finnhub with key: {api_key[:8]}...")
    url = f"https://finnhub.io/api/v1/news?category=general&token={api_key}"
    response = requests.get(url)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Articles found: {len(data)}")
        if data:
            print(f"Sample headline: {data[0].get('headline')}")
    else:
        print(f"❌ Error: {response.text}")

if __name__ == "__main__":
    test_newsapi()
    test_finnhub()
