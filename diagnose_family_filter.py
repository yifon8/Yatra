"""
Diagnostic script for family-friendly filter issue with Mumbai beaches
"""

import os
import sys
import json
from agent.dataset_handler import DatasetHandler
from agent.tools import TravelTools

# Set up Google API key
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "")

def main():
    print("=" * 80)
    print("DIAGNOSTIC TEST: Family-Friendly Filter for Mumbai Beaches")
    print("=" * 80)

    # Initialize dataset and tools
    print("\n1. Initializing dataset handler...")
    dataset = DatasetHandler()
    tools = TravelTools(dataset)

    # Step 1: Search for beaches
    print("\n2. Searching for beaches...")
    search_result = tools.search_destinations_quantitative(destination_type="beach")
    print(f"   - Found {search_result.get('count', 0)} beaches")

    # Step 2: Filter by Mumbai
    print("\n3. Filtering by Mumbai...")
    destinations = search_result.get('destinations', [])
    print(f"   - Destinations to filter: {len(destinations)}")

    # Find Mumbai beaches in the results
    mumbai_beaches = [d for d in destinations if d.get('city', '').lower() == 'mumbai']
    print(f"   - Mumbai beaches found: {len(mumbai_beaches)}")
    for beach in mumbai_beaches:
        print(f"     * {beach.get('name')} - Rating: {beach.get('google_review_rating')}")

    city_filter_result = tools.filter_by_city(city="Mumbai", destinations=destinations)
    print(f"   - After city filter: {city_filter_result.get('count', 0)} destinations")
    print(f"   - Cities included: {city_filter_result.get('city_names', [])}")

    filtered_destinations = city_filter_result.get('destinations', [])
    for dest in filtered_destinations[:5]:  # Show first 5
        print(f"     * {dest.get('name')} ({dest.get('city')})")

    # Step 3: Test family-friendly filter with detailed logging
    print("\n4. Testing family-friendly filter...")

    if not filtered_destinations:
        print("   ERROR: No destinations to filter after city filter!")
        return

    # Test with just destination names (strings)
    destination_names = [d.get('name') for d in filtered_destinations]
    print(f"   - Testing with {len(destination_names)} destination names")
    print(f"   - Sample names: {destination_names[:3]}")

    # First, verify we can look up these destinations
    print("\n5. Verifying destination lookup...")
    for name in destination_names[:3]:  # Test first 3
        details = dataset.get_destination_details(name)
        if details:
            print(f"   ✓ Found: {name}")
            print(f"     - City: {details.get('city')}, Type: {details.get('type')}")
        else:
            print(f"   ✗ NOT FOUND: {name}")

    # Now test the family-friendly filter with detailed output
    print("\n6. Running family-friendly filter...")
    print("   (This will use LLM with web search, may take a moment...)")

    family_filter_result = tools.filter_by_family_friendly(destination_names[:3])  # Test with first 3

    print(f"\n7. Family-friendly filter results:")
    print(f"   - Success: {family_filter_result.get('success')}")
    print(f"   - Count: {family_filter_result.get('count')}")
    print(f"   - Method: {family_filter_result.get('method')}")

    # Show analysis log
    analysis_log = family_filter_result.get('analysis_log', [])
    print(f"\n8. Analysis log ({len(analysis_log)} entries):")
    for entry in analysis_log:
        print(f"\n   Destination: {entry.get('destination')}")
        print(f"   Status: {entry.get('status')}")

        if entry.get('status') == 'analyzed':
            print(f"   Is Family-Friendly: {entry.get('is_family_friendly')}")
            print(f"   Confidence: {entry.get('confidence')}")
            print(f"   Reasoning: {entry.get('reasoning')}")
            print(f"   Key Activities: {entry.get('key_activities', [])}")
            print(f"   Concerns: {entry.get('concerns', [])}")
        elif entry.get('status') == 'error':
            print(f"   Error: {entry.get('reason')}")
            if 'raw_response' in entry:
                print(f"   Raw Response: {entry.get('raw_response')}")
        elif entry.get('status') == 'not_found':
            print(f"   Reason: {entry.get('reason')}")

    print("\n" + "=" * 80)
    print("DIAGNOSTIC TEST COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
