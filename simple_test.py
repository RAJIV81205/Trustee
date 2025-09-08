#!/usr/bin/env python3
"""
Simple production test for the internship matching API
"""

import requests
import json

def test_api():
    """Test the production API"""
    base_url = "http://localhost:8000"
    
    print("🧪 Testing Production Internship Matching API")
    print("=" * 50)
    
    # Test 1: Health check
    print("1. Health Check...")
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ API is healthy - Model trained: {result['model_trained']}")
        else:
            print(f"   ❌ Health check failed: {response.status_code}")
            return
    except requests.exceptions.ConnectionError:
        print("   ❌ Connection Error: Make sure the API server is running")
        return
    
    # Test 2: Match internships
    print("\n2. Testing Internship Matching...")
    user_data = {
        "skills": ["Python", "Machine Learning", "SQL"],
        "experience": 1,
        "mode": "Hybrid",
        "location": "Karnataka",
        "sector": "tech",
        "stipend": 25000,
        "duration": 6
    }
    
    try:
        response = requests.post(f"{base_url}/match-internships", json=user_data)
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ Found {result['total_matches']} matches")
            
            # Show top 3 matches
            print("   Top 3 matches:")
            for i, match in enumerate(result['matches'][:3], 1):
                print(f"   {i}. {match['name']} - Score: {match['matching_score']}/10")
                print(f"      Location: {match['location']}, Stipend: ₹{match['stipend']}")
        else:
            print(f"   ❌ Matching failed: {response.status_code}")
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 3: Get all internships
    print("\n3. Testing Get All Internships...")
    try:
        response = requests.get(f"{base_url}/internships")
        if response.status_code == 200:
            result = response.json()
            if result['success']:
                print(f"   ✅ Retrieved {result['data']['total']} internships")
            else:
                print("   ❌ Failed to get internships")
        else:
            print(f"   ❌ Get internships failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 Production API test completed!")

if __name__ == "__main__":
    test_api()