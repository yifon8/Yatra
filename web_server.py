#!/usr/bin/env python3
"""
Yatra Web Server - Flask API for the web interface
Handles form submissions and connects to the DestinationSuggester agent
"""

from flask import Flask, request, jsonify, send_from_directory, session
from flask_cors import CORS
import os
import sys
from typing import Dict, Any, List
import pandas as pd

from agent import DestinationSuggester

app = Flask(__name__, static_folder='.')
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'yatra-secret-key-change-in-production')
CORS(app, supports_credentials=True)

# Initialize the agent
agent = None


def initialize_agent():
    """Initialize the Yatra agent"""
    global agent

    # Get API key from environment
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        print("❌ Error: Google API key required!")
        print("\nSet it using: export GOOGLE_API_KEY='your-key'")
        print("Get your API key at: https://makersuite.google.com/app/apikey\n")
        sys.exit(1)

    try:
        print("Initializing Yatra agent...")
        agent = DestinationSuggester(api_key=api_key)
        print("✓ Agent initialized successfully!")
    except FileNotFoundError as e:
        print(f"\n❌ {str(e)}")
        print("\nPlease download the dataset from:")
        print("https://www.kaggle.com/datasets/saketk511/travel-dataset-guide-to-indias-must-see-places")
        print("\nAnd place it in the data/ directory as 'destinations.csv'\n")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error initializing agent: {str(e)}\n")
        sys.exit(1)


@app.route('/')
def serve_index():
    """Serve the main HTML page"""
    return send_from_directory('.', 'index.html')


def extract_destinations_from_tools(tools_used: List[Dict]) -> List[Dict]:
    """
    Extract all destinations from tool results

    Args:
        tools_used: List of tool execution results

    Returns:
        List of all destination dictionaries
    """
    all_destinations = []

    for tool_call in tools_used:
        if tool_call.get('tool') == 'search_destinations_quantitative':
            result = tool_call.get('result', {})
            destinations = result.get('destinations', [])
            all_destinations.extend(destinations)

    return all_destinations


def sort_destinations_by_rating(destinations: List[Dict]) -> List[Dict]:
    """
    Sort destinations by rating in descending order
    Destinations without ratings are placed at the end

    Args:
        destinations: List of destination dictionaries

    Returns:
        Sorted list of destinations
    """
    # Try to find rating column
    rating_columns = ['google_review_rating', 'rating', 'review_rating',
                     'user_rating', 'average_rating']

    def get_rating(dest: Dict) -> float:
        """Extract rating from destination, return -1 if not found or invalid"""
        for col in rating_columns:
            if col in dest:
                try:
                    rating_value = dest[col]
                    # Handle empty strings
                    if rating_value == '' or rating_value is None:
                        continue
                    # Convert to float
                    rating = float(rating_value)
                    # Check if it's a valid number (not NaN)
                    if rating == rating:  # NaN != NaN
                        return rating
                except (ValueError, TypeError):
                    continue
        return -1  # No valid rating found

    # Sort by rating descending, destinations without ratings go to the end
    sorted_destinations = sorted(destinations, key=get_rating, reverse=True)

    return sorted_destinations


@app.route('/api/suggest', methods=['POST'])
def suggest_destinations():
    """
    Handle destination suggestion requests

    Expected JSON body:
    {
        "destinationType": "beach|mountain|heritage|wildlife",
        "duration": float (hours, optional) - e.g., 0.5, 1.5, 8.25,
        "budget": integer (Rupees, optional) - 0 allowed for free destinations
    }
    """
    try:
        # Get form data from request
        data = request.get_json()

        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400

        # Extract parameters
        destination_type = data.get('destinationType')
        duration = data.get('duration')
        budget = data.get('budget')

        # Validate required fields
        if not destination_type:
            return jsonify({
                'success': False,
                'error': 'Destination type is required'
            }), 400

        # Duration is optional, but if provided, must be > 0
        if duration is not None and duration <= 0:
            return jsonify({
                'success': False,
                'error': 'Duration must be greater than 0 hours (e.g., 0.5, 1.5, 8.25)'
            }), 400

        # Budget is optional, but if provided, must be >= 0 (0 allowed for free destinations)
        if budget is not None and budget < 0:
            return jsonify({
                'success': False,
                'error': 'Budget cannot be negative'
            }), 400

        # Log the request
        print(f"\n📝 Received request:")
        print(f"   Destination Type: {destination_type}")
        print(f"   Duration: {duration} hours" if duration is not None else "   Duration: Not specified")
        print(f"   Budget: ₹{budget:,}" if budget is not None else "   Budget: Not specified")

        # Call the agent
        result = agent.suggest_destinations(
            destination_type=destination_type,
            hours=duration,
            budget=budget
        )

        # Check if agent returned an error (e.g., no matches found)
        if not result.get('success', True):
            return jsonify({
                'success': False,
                'error': result.get('error', 'An error occurred while processing your request'),
                'query': result.get('query'),
                'tools_used': len(result.get('tools_used', [])),
                'iterations': result.get('iterations', 0)
            }), 400

        # Extract destinations from tool results
        all_destinations = extract_destinations_from_tools(result.get('tools_used', []))

        # Sort destinations by rating (descending)
        sorted_destinations = sort_destinations_by_rating(all_destinations)

        # Store sorted destinations in session for pagination
        session['sorted_destinations'] = sorted_destinations
        session['current_page'] = 0

        # Get first 3 destinations
        page_size = 3
        displayed_destinations = sorted_destinations[:page_size]
        has_more = len(sorted_destinations) > page_size

        # Return the response
        return jsonify({
            'success': True,
            'destinations': displayed_destinations,
            'has_more': has_more,
            'total_count': len(sorted_destinations),
            'query': result['query'],
            'tools_used': len(result.get('tools_used', [])),
            'iterations': result.get('iterations', 0)
        })

    except Exception as e:
        print(f"❌ Error processing request: {str(e)}")
        import traceback
        traceback.print_exc()

        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/more', methods=['POST'])
def get_more_destinations():
    """
    Get next batch of destinations from the current search results

    Returns:
        Next 3 destinations from the sorted list
    """
    try:
        # Get sorted destinations from session
        sorted_destinations = session.get('sorted_destinations', [])
        current_page = session.get('current_page', 0)

        if not sorted_destinations:
            return jsonify({
                'success': False,
                'error': 'No active search. Please submit a new search first.'
            }), 400

        # Calculate next page
        page_size = 3
        next_page = current_page + 1
        start_idx = next_page * page_size
        end_idx = start_idx + page_size

        # Check if there are more destinations
        if start_idx >= len(sorted_destinations):
            return jsonify({
                'success': True,
                'destinations': [],
                'has_more': False,
                'end_of_results': True,
                'message': 'End of Suggestions, feel free to Restart your search.'
            })

        # Get next batch
        displayed_destinations = sorted_destinations[start_idx:end_idx]
        has_more = end_idx < len(sorted_destinations)

        # Check if this is the last batch
        is_last_batch = not has_more and len(displayed_destinations) > 0

        # Update session
        session['current_page'] = next_page

        return jsonify({
            'success': True,
            'destinations': displayed_destinations,
            'has_more': has_more,
            'end_of_results': is_last_batch,
            'message': 'End of Suggestions, feel free to Restart your search.' if is_last_batch else None,
            'total_count': len(sorted_destinations)
        })

    except Exception as e:
        print(f"❌ Error getting more destinations: {str(e)}")
        import traceback
        traceback.print_exc()

        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'agent_initialized': agent is not None
    })


if __name__ == '__main__':
    # Initialize the agent before starting the server
    initialize_agent()

    # Start the Flask server
    print("\n" + "="*60)
    print("🌐 Starting Yatra Web Server")
    print("="*60)
    print(f"\n🔗 Open your browser to: http://localhost:5000")
    print("\n💡 Press Ctrl+C to stop the server\n")

    app.run(host='0.0.0.0', port=5000, debug=True)
