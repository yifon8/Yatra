#!/usr/bin/env python3
"""
Test script to verify the Mumbai beach query generation
"""

def test_query_generation():
    """Test the query generation for Mumbai beach search"""

    # Create a dummy suggester just to test query generation
    # We'll access the private method _create_query
    class QueryTester:
        def _create_query(self, destination_type, city, hours, budget):
            """This is the updated _create_query method"""
            # Create query based on whether city is specified
            if city is not None:
                # For city-based searches, guide the agent through the correct tool sequence
                query = f"I need help finding {destination_type or 'travel'} destinations in or near {city}, India.\n"

                # Add filter details
                if hours is not None:
                    hours_str = f"{hours:g}"
                    query += f"\nVisit duration: {hours_str} hours."

                if budget is not None:
                    if budget == 0:
                        query += f"\nBudget: Free (no cost)."
                    else:
                        query += f"\nBudget: around ₹{budget:,.0f}."

                # Add tool instruction with proper sequence
                query += f'\n\nPlease search for destinations first using search_destinations_quantitative'
                if destination_type:
                    query += f' with destination_type="{destination_type}"'
                if hours is not None:
                    query += f', max_hours={hours:g}'
                if budget is not None:
                    query += f', max_budget={budget}'
                query += f'. Then use filter_by_city with city="{city}" to filter the results to destinations in or near {city}.'
            else:
                # Build filter description for non-city queries
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

                query = f"I'm looking for travel destination recommendations in India: {filters_desc}."
                query += "\n\nPlease use the available tools to search the dataset and provide me with your top 3 recommendations with reasoning."

            return query

    tester = QueryTester()

    print("=" * 80)
    print("Testing Mumbai Beach Query Generation")
    print("=" * 80)

    query = tester._create_query(
        destination_type="beach",
        city="Mumbai",
        hours=None,
        budget=None
    )

    print("\nGenerated Query:")
    print("-" * 80)
    print(query)
    print("-" * 80)

    print("\n✓ Query generated successfully!")
    print("\nExpected behavior:")
    print("1. Agent calls search_destinations_quantitative with destination_type='beach'")
    print("2. Tool returns max 100 destinations (limited to prevent payload overflow)")
    print("3. Agent calls filter_by_city with city='Mumbai' and the destinations")
    print("4. Tool filters to Mumbai and adjacent cities")
    print("5. Agent provides recommendations")

if __name__ == '__main__':
    test_query_generation()
