#!/usr/bin/env python3
"""
Test script to verify pandas filtering functionality
"""

import sys
import os
import importlib.util

# Load the dataset_handler module directly without triggering __init__.py
spec = importlib.util.spec_from_file_location(
    "dataset_handler",
    os.path.join(os.path.dirname(__file__), "agent", "dataset_handler.py")
)
dataset_handler = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dataset_handler)

DatasetHandler = dataset_handler.DatasetHandler

def test_filtering():
    """Test the filtering functionality"""
    print("=" * 60)
    print("Testing Pandas Filtering Functionality")
    print("=" * 60)

    # Initialize dataset handler
    print("\n1. Loading dataset...")
    handler = DatasetHandler()
    print(f"   Total destinations: {len(handler.df)}")

    # Test rating filter
    print("\n2. Testing rating filter (null or > 4.0)...")
    filtered = handler.filter_by_rating(min_rating=4.0)
    print(f"   Destinations with rating > 4.0 or null: {len(filtered)}")

    # Show some examples
    if len(filtered) > 0:
        print("\n   Sample destinations:")
        for i, row in filtered.head(5).iterrows():
            rating = row.get('google_review_rating', 'N/A')
            print(f"   - {row.get('name', 'Unknown')}: Rating {rating}")

    # Test family-friendly filter
    print("\n3. Testing family-friendly filter...")
    filtered = handler.filter_by_family_friendly()
    print(f"   Family-friendly destinations: {len(filtered)}")

    # Show some examples
    if len(filtered) > 0:
        print("\n   Sample destinations:")
        for i, row in filtered.head(5).iterrows():
            print(f"   - {row.get('name', 'Unknown')} ({row.get('type', 'Unknown type')})")

    # Test combined search
    print("\n4. Testing combined search (beach, budget 100, 2 hours)...")
    filtered = handler.search_quantitative(
        destination_type="beach",
        max_budget=100,
        max_hours=2
    )
    print(f"   Matching destinations: {len(filtered)}")

    if len(filtered) > 0:
        print("\n   Results:")
        for i, row in filtered.head(5).iterrows():
            name = row.get('name', 'Unknown')
            rating = row.get('google_review_rating', 'N/A')
            fee = row.get('entrance_fee_in_inr', 'N/A')
            hours = row.get('time_needed_to_visit_in_hrs', 'N/A')
            dest_type = row.get('type', 'Unknown')
            print(f"   - {name}")
            print(f"     Type: {dest_type}, Rating: {rating}, Fee: ₹{fee}, Hours: {hours}")

    # Test budget filter specifically
    print("\n6. Testing budget filter (entrance_fee_in_inr <= 50)...")
    filtered = handler.filter_by_budget(max_budget=50)
    print(f"   Destinations within budget: {len(filtered)}")

    # Test duration filter specifically
    print("\n7. Testing duration filter (<= 1.5 hours)...")
    filtered = handler.filter_by_duration(max_hours=1.5)
    print(f"   Destinations within duration: {len(filtered)}")

    print("\n" + "=" * 60)
    print("Testing Complete!")
    print("=" * 60)

if __name__ == '__main__':
    test_filtering()
