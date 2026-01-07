"""
Test script for Customer Financial Risk Prediction API
"""

import requests
import json
import time

# API Configuration
BASE_URL = "http://localhost:8000"
HEADERS = {"Content-Type": "application/json"}

def print_section(title):
    """Print formatted section header"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def test_health_check():
    """Test health check endpoint"""
    print_section("1. Health Check")
    
    try:
        response = requests.get(f"{BASE_URL}/health", headers=HEADERS)
        print(f"✅ Status Code: {response.status_code}")
        print(f"✅ Response: {json.dumps(response.json(), indent=2)}")
        return True
    except requests.exceptions.ConnectionError:
        print("❌ API not running. Start with: python api_main.py")
        return False

def test_single_prediction():
    """Test single customer prediction"""
    print_section("2. Single Customer Prediction")
    
    customer_data = {
        "customer_id": "TEST001",
        "age": 32,
        "monthly_expenditure": 125000.0,
        "credit_score": 680,
        "transaction_count": 30,
        "avg_transaction_value": 4166.67,
        "uses_pos": 1,
        "uses_web": 0,
        "uses_ussd": 1,
        "uses_mobile_app": 1,
        "income_level": "Middle",
        "saving_behavior": "Average",
        "location": "Nairobi",
        "feedback": "Good service but mobile app crashes sometimes"
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            json=customer_data,
            headers=HEADERS
        )
        
        print(f"✅ Status Code: {response.status_code}")
        print(f"✅ Customer ID: {customer_data['customer_id']}")
        
        result = response.json()
        print(f"✅ Cluster: {result['cluster_name']} (ID: {result['cluster_id']})")
        print(f"✅ Risk: {result['risk_category']} (Score: {result['risk_score']})")
        print(f"✅ Digital Adoption: {result['digital_adoption_score']}/4.0")
        print(f"✅ Processing Time: {result['processing_time']} ms")
        print(f"✅ Recommendations:")
        for i, rec in enumerate(result['recommendations'], 1):
            print(f"   {i}. {rec}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_batch_prediction():
    """Test batch prediction"""
    print_section("3. Batch Prediction")
    
    batch_data = {
        "customers": [
            {
                "customer_id": "BATCH001",
                "age": 45,
                "monthly_expenditure": 300000.0,
                "credit_score": 750,
                "transaction_count": 15,
                "avg_transaction_value": 20000.0,
                "uses_pos": 0,
                "uses_web": 1,
                "uses_ussd": 0,
                "uses_mobile_app": 1,
                "income_level": "High",
                "saving_behavior": "Good",
                "location": "Johannesburg",
                "feedback": "Excellent digital banking platform"
            },
            {
                "customer_id": "BATCH002",
                "age": 28,
                "monthly_expenditure": 75000.0,
                "credit_score": 580,
                "transaction_count": 45,
                "avg_transaction_value": 1666.67,
                "uses_pos": 1,
                "uses_web": 0,
                "uses_ussd": 1,
                "uses_mobile_app": 0,
                "income_level": "Low",
                "saving_behavior": "Poor",
                "location": "Kampala",
                "feedback": "Charges are too high and unclear"
            }
        ]
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict/batch",
            json=batch_data,
            headers=HEADERS
        )
        
        print(f"✅ Status Code: {response.status_code}")
        print(f"✅ Number of predictions: {len(response.json())}")
        
        for i, prediction in enumerate(response.json(), 1):
            print(f"\n   Customer {i}: {prediction['customer_id']}")
            print(f"   • Cluster: {prediction['cluster_name']}")
            print(f"   • Risk: {prediction['risk_category']}")
            print(f"   • Top Recommendation: {prediction['recommendations'][0]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_clusters_info():
    """Test cluster information endpoint"""
    print_section("4. Cluster Information")
    
    try:
        response = requests.get(f"{BASE_URL}/clusters", headers=HEADERS)
        
        print(f"✅ Status Code: {response.status_code}")
        result = response.json()
        
        print(f"✅ Total Clusters: {result['total_clusters']}")
        print(f"✅ Clustering Method: {result['clustering_method']}")
        
        print(f"\n📊 Cluster Details:")
        for cluster_id, info in result['clusters'].items():
            print(f"\n   {info['name']} ({cluster_id}):")
            print(f"   • Description: {info['description']}")
            print(f"   • Typical Customers: {', '.join(info['typical_customers'][:2])}")
            print(f"   • Products: {', '.join(info['recommended_products'])}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_demo_endpoint():
    """Test demo endpoint"""
    print_section("5. Demo Endpoint")
    
    try:
        response = requests.get(f"{BASE_URL}/demo", headers=HEADERS)
        
        print(f"✅ Status Code: {response.status_code}")
        result = response.json()
        
        print(f"✅ Demo Customer: {result['customer_id']}")
        print(f"✅ Assigned to: {result['cluster_name']}")
        print(f"✅ Risk Assessment: {result['risk_category']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_api_performance():
    """Test API performance"""
    print_section("6. Performance Test")
    
    try:
        customer_data = {
            "customer_id": "PERF001",
            "age": 35,
            "monthly_expenditure": 150000.0,
            "credit_score": 650,
            "transaction_count": 20,
            "avg_transaction_value": 7500.0,
            "uses_pos": 1,
            "uses_web": 1,
            "uses_ussd": 0,
            "uses_mobile_app": 1,
            "income_level": "Middle",
            "saving_behavior": "Average",
            "location": "Accra",
            "feedback": "Average experience"
        }
        
        # Test 10 requests
        times = []
        for i in range(10):
            start_time = time.time()
            response = requests.post(
                f"{BASE_URL}/predict",
                json=customer_data,
                headers=HEADERS
            )
            end_time = time.time()
            times.append((end_time - start_time) * 1000)  # Convert to ms
        
        avg_time = sum(times) / len(times)
        max_time = max(times)
        min_time = min(times)
        
        print(f"✅ Number of requests: 10")
        print(f"✅ Average response time: {avg_time:.2f} ms")
        print(f"✅ Minimum response time: {min_time:.2f} ms")
        print(f"✅ Maximum response time: {max_time:.2f} ms")
        print(f"✅ Performance: {'Excellent' if avg_time < 100 else 'Good' if avg_time < 500 else 'Slow'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def run_all_tests():
    """Run all API tests"""
    print("🚀 Starting API Tests...")
    print(f"📡 API Base URL: {BASE_URL}")
    
    tests = [
        ("Health Check", test_health_check),
        ("Single Prediction", test_single_prediction),
        ("Batch Prediction", test_batch_prediction),
        ("Cluster Info", test_clusters_info),
        ("Demo", test_demo_endpoint),
        ("Performance", test_api_performance)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} failed with error: {e}")
            results.append((test_name, False))
    
    # Summary
    print_section("TEST SUMMARY")
    
    total_tests = len(results)
    passed_tests = sum(1 for _, success in results if success)
    
    print(f"📊 Total Tests: {total_tests}")
    print(f"✅ Passed: {passed_tests}")
    print(f"❌ Failed: {total_tests - passed_tests}")
    print(f"📈 Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    
    print(f"\n📋 Detailed Results:")
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {status} - {test_name}")
    
    if passed_tests == total_tests:
        print(f"\n🎉 All tests passed! API is ready for deployment.")
    else:
        print(f"\n⚠️  Some tests failed. Please check API implementation.")

if __name__ == "__main__":
    run_all_tests()