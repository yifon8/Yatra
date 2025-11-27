#!/usr/bin/env python3
"""
Test script to verify query generation for city-based searches
"""

from typing import Optional

def _create_query(destination_type: Optional[str],
                 city: Optional[str],
                 hours: Optional[float],
                 budget: Optional[float]) -> str:
    """Create a natural language query from user parameters"""

    # Build filter description
    filter_parts = []
    if destination_type:
        filter_parts.append(f"{destination_type} destinations")
    else:
        filter_parts.append("destinations")

    if hours is not None:
        hours_str = f"{hours:g}"
        filter_parts.append(f"visit duration within {hours_str} hours")

    if budget is not None:
        if budget == 0:
            filter_parts.append("free (no cost)")
        else:
            filter_parts.append(f"budget around ₹{budget:,.0f}")

    filters_desc = ", ".join(filter_parts)

    # Create query based on whether city is specified
    if city is not None:
        query = f"""I need help finding {filters_desc} in or near {city}, India.

IMPORTANT: Use this workflow:
1. Use filter_by_city with city="{city}" and destination_type="{destination_type if destination_type else 'not specified'}"
2. The tool will automatically:
   - Find adjacent cities to {city} using LLM with web search
   - Filter destinations using pandas by those city names
   - Apply system filters (rating >4.0 or null, family-friendly)
   - Sort by rating descending and return top 25
3. Provide your top 3 recommendations with reasoning

Note: Additional filters for hours={hours} and budget={budget} should be mentioned in your recommendations."""
    else:
        query = f"I'm looking for travel destination recommendations in India: {filters_desc}."
        query += "\n\nPlease use the available tools to search the dataset and provide me with your top 3 recommendations with reasoning."

    return query

def test_query_generation():
    """Test the query generation with different parameters"""
    print("=" * 80)
    print("Testing Query Generation")
    print("=" * 80)

    # Test 1: City-based query with type
    print("\n1. Query with city='Mumbai', type='heritage':")
    print("-" * 80)
    query = _create_query(
        destination_type="heritage",
        city="Mumbai",
        hours=None,
        budget=None
    )
    print(query)

    # Test 2: City-based query with type, hours, and budget
    print("\n2. Query with city='Mumbai', type='heritage', hours=3, budget=500:")
    print("-" * 80)
    query = _create_query(
        destination_type="heritage",
        city="Mumbai",
        hours=3,
        budget=500
    )
    print(query)

    # Test 3: Non-city query
    print("\n3. Query without city, type='beach', hours=2:")
    print("-" * 80)
    query = _create_query(
        destination_type="beach",
        city=None,
        hours=2,
        budget=None
    )
    print(query)

    # Test 4: City only
    print("\n4. Query with only city='Delhi':")
    print("-" * 80)
    query = _create_query(
        destination_type=None,
        city="Delhi",
        hours=None,
        budget=None
    )
    print(query)

    print("\n" + "=" * 80)
    print("Testing Complete!")
    print("=" * 80)

if __name__ == '__main__':
    test_query_generation()
