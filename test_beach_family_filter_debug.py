"""
Enhanced diagnostic script for testing family-friendly filter with 23 beach destinations
Uses comprehensive logging to debug the web search process
"""

import os
import sys
import logging
from agent.dataset_handler import DatasetHandler
from agent.tools import TravelTools

# Set up environment
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "")

def main():
    print("=" * 100)
    print("ENHANCED DIAGNOSTIC TEST: Family-Friendly Filter for Beach Destinations")
    print("WITH COMPREHENSIVE LOGGING")
    print("=" * 100)
    print()
    print("This script will:")
    print("1. Search for beach type destinations")
    print("2. Test the family-friendly filter with detailed logging")
    print("3. Write comprehensive debug logs to 'yatra_debug.log'")
    print("4. Show real-time progress to console")
    print()
    print("=" * 100)
    print()

    # Initialize dataset and tools
    print("Initializing dataset handler and tools...")
    dataset = DatasetHandler()
    tools = TravelTools(dataset)
    print("✓ Initialization complete\n")

    # Step 1: Search for beaches
    print("Step 1: Searching for beach type destinations...")
    search_result = tools.search_destinations_quantitative(destination_type="beach")

    if not search_result.get('success'):
        print(f"✗ Error searching for beaches: {search_result.get('error')}")
        return

    beach_destinations = search_result.get('destinations', [])
    print(f"✓ Found {len(beach_destinations)} beach destinations\n")

    if len(beach_destinations) == 0:
        print("No beach destinations found. Exiting.")
        return

    # Limit to 23 destinations as mentioned
    test_count = min(23, len(beach_destinations))
    beach_names = [d.get('name') for d in beach_destinations[:test_count]]

    print(f"Step 2: Testing family-friendly filter with {test_count} beach destinations...")
    print(f"Beach destinations to test:")
    for i, name in enumerate(beach_names, 1):
        print(f"  {i:2d}. {name}")
    print()

    print("=" * 100)
    print("STARTING FAMILY-FRIENDLY FILTER ANALYSIS")
    print("This may take several minutes as each destination requires a web search...")
    print("Check 'yatra_debug.log' for detailed progress")
    print("=" * 100)
    print()

    # Step 3: Run family-friendly filter with comprehensive logging
    result = tools.filter_by_family_friendly(beach_names)

    # Display results
    print("\n" + "=" * 100)
    print("RESULTS SUMMARY")
    print("=" * 100)

    if result.get("success"):
        print(f"\n✓ Filter completed successfully!")
        print(f"  Method: {result.get('method', 'unknown')}")
        print(f"  Total destinations analyzed: {len(beach_names)}")
        print(f"  Family-friendly destinations found: {result['count']}")
        print(f"  Excluded destinations: {len(beach_names) - result['count']}")

        print("\n" + "-" * 100)
        print("ANALYSIS LOG")
        print("-" * 100)

        analyzed_count = 0
        error_count = 0
        not_found_count = 0

        for log_entry in result.get('analysis_log', []):
            status = log_entry['status']

            if status == 'analyzed':
                analyzed_count += 1
                print(f"\n[{analyzed_count}] {log_entry['destination']}")
                print(f"    Status: ✓ Analyzed")
                print(f"    Family-Friendly: {'✓ YES' if log_entry['is_family_friendly'] else '✗ NO'}")
                print(f"    Confidence: {log_entry['confidence']}")
                print(f"    Reasoning: {log_entry['reasoning'][:150]}...")
                if log_entry.get('concerns'):
                    print(f"    Concerns: {', '.join(log_entry['concerns'])}")

            elif status == 'error':
                error_count += 1
                print(f"\n[ERROR] {log_entry['destination']}")
                print(f"    Reason: {log_entry.get('reason', 'Unknown error')}")

            elif status == 'not_found':
                not_found_count += 1
                print(f"\n[NOT FOUND] {log_entry['destination']}")
                print(f"    Reason: {log_entry.get('reason', 'Unknown')}")

        print("\n" + "-" * 100)
        print(f"Summary: {analyzed_count} analyzed, {error_count} errors, {not_found_count} not found")
        print("-" * 100)

        if result['count'] > 0:
            print("\n" + "=" * 100)
            print("FAMILY-FRIENDLY BEACH DESTINATIONS")
            print("=" * 100)

            for i, dest in enumerate(result.get('destinations', []), 1):
                print(f"\n{i}. {dest.get('name', 'Unknown')}")
                print(f"   Location: {dest.get('city', '')}, {dest.get('state', '')}")
                print(f"   Rating: {dest.get('google_review_rating', 'N/A')}")

                if 'family_friendly_analysis' in dest:
                    analysis = dest['family_friendly_analysis']
                    print(f"   Confidence: {analysis['confidence']}")
                    print(f"   Reasoning: {analysis['reasoning'][:100]}...")
                    if analysis.get('key_activities'):
                        print(f"   Activities: {', '.join(analysis['key_activities'][:3])}")

        print("\n" + "=" * 100)
        print("Check 'yatra_debug.log' for complete detailed logs")
        print("=" * 100)

    else:
        print(f"\n✗ Error: {result.get('error', 'Unknown error')}")
        print("Check 'yatra_debug.log' for detailed error information")

if __name__ == "__main__":
    main()
