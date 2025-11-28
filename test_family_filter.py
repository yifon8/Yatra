"""
Test script for the family-friendly filter with LLM and web search
"""

import os
import sys
from agent.dataset_handler import DatasetHandler
from agent.tools import TravelTools

def test_family_filter():
    """Test the family-friendly filter with real destinations"""

    # Initialize dataset handler
    dataset = DatasetHandler('data/destinations.csv')

    # Initialize tools
    tools = TravelTools(dataset)

    # Test with a mix of destination types including beaches
    test_destinations = [
        "India Gate",           # War memorial - should be family-friendly
        "Goa Beaches",          # Beach - testing the beach filter fix
        "Andaman Islands"       # Beach - testing the beach filter fix
    ]

    print("Testing family-friendly filter with LLM and web search...")
    print(f"Analyzing {len(test_destinations)} destinations:")
    for dest in test_destinations:
        print(f"  - {dest}")
    print("\nThis may take a minute as the LLM searches the web for each destination...\n")

    # Run the filter
    result = tools.filter_by_family_friendly(test_destinations)

    # Display results
    if result.get("success"):
        print("="*70)
        print("RESULTS")
        print("="*70)
        print(f"\nFamily-friendly destinations found: {result['count']} out of {len(test_destinations)}")
        print(f"Method: {result.get('method', 'unknown')}")

        print("\n" + "-"*70)
        print("DETAILED ANALYSIS LOG")
        print("-"*70)

        for log_entry in result.get('analysis_log', []):
            print(f"\nDestination: {log_entry['destination']}")
            print(f"Status: {log_entry['status']}")

            if log_entry['status'] == 'analyzed':
                print(f"Family-Friendly: {log_entry['is_family_friendly']}")
                print(f"Confidence: {log_entry['confidence']}")
                print(f"Reasoning: {log_entry['reasoning']}")
                if log_entry.get('found_terms'):
                    print(f"Found Terms: {', '.join(log_entry['found_terms'])}")
                if log_entry.get('key_activities'):
                    print(f"Key Activities: {', '.join(log_entry['key_activities'])}")
                if log_entry.get('concerns'):
                    print(f"Concerns: {', '.join(log_entry['concerns'])}")
            else:
                print(f"Reason: {log_entry.get('reason', 'Unknown')}")
            print("-"*70)

        print("\n" + "="*70)
        print("FILTERED DESTINATIONS")
        print("="*70)

        for i, dest in enumerate(result.get('destinations', []), 1):
            print(f"\n{i}. {dest.get('name', 'Unknown')}")
            print(f"   Type: {dest.get('type', 'Unknown')}")
            print(f"   Location: {dest.get('city', '')}, {dest.get('state', '')}")
            print(f"   Rating: {dest.get('google_review_rating', 'N/A')}")

            if 'family_friendly_analysis' in dest:
                analysis = dest['family_friendly_analysis']
                print(f"   Analysis Confidence: {analysis['confidence']}")
                print(f"   Reasoning: {analysis['reasoning']}")
                if analysis.get('found_terms'):
                    print(f"   Found Terms: {', '.join(analysis['found_terms'])}")
                if analysis.get('key_activities'):
                    print(f"   Key Activities: {', '.join(analysis['key_activities'][:3])}")

        print("\n" + "="*70)

    else:
        print(f"Error: {result.get('error', 'Unknown error')}")

if __name__ == "__main__":
    test_family_filter()
