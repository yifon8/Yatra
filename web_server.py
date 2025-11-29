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
import uuid
from datetime import datetime, timedelta
import requests
from urllib.parse import quote

from agent import DestinationSuggester

app = Flask(__name__, static_folder='.')
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'yatra-secret-key-change-in-production')
CORS(app, supports_credentials=True)

# Initialize the agent
agent = None

# Server-side cache for storing destination results
# Format: {session_id: {'destinations': [...], 'current_page': 0, 'expires': datetime}}
destinations_cache = {}

# Track active processing sessions and their cancellation flags
# Format: {session_id: {'cancelled': bool, 'started': datetime}}
processing_sessions = {}

# Cache cleanup interval
CACHE_EXPIRY_HOURS = 2


def clean_expired_cache():
    """Remove expired entries from the cache"""
    now = datetime.now()
    expired_keys = [
        key for key, value in destinations_cache.items()
        if value['expires'] < now
    ]
    for key in expired_keys:
        del destinations_cache[key]


def is_processing_cancelled(session_id: str) -> bool:
    """
    Check if processing has been cancelled for a session

    Args:
        session_id: Session ID to check

    Returns:
        True if processing should be cancelled, False otherwise
    """
    if session_id in processing_sessions:
        return processing_sessions[session_id].get('cancelled', False)
    return False


def cancel_processing(session_id: str):
    """
    Cancel ongoing processing for a session

    Args:
        session_id: Session ID to cancel
    """
    if session_id in processing_sessions:
        processing_sessions[session_id]['cancelled'] = True
        print(f"🛑 Cancelled processing for session: {session_id[:8]}...")


def start_processing(session_id: str):
    """
    Mark that processing has started for a session

    Args:
        session_id: Session ID starting processing
    """
    processing_sessions[session_id] = {
        'cancelled': False,
        'started': datetime.now()
    }
    print(f"🚀 Started processing for session: {session_id[:8]}...")


def end_processing(session_id: str):
    """
    Mark that processing has ended for a session

    Args:
        session_id: Session ID ending processing
    """
    if session_id in processing_sessions:
        del processing_sessions[session_id]
        print(f"✅ Ended processing for session: {session_id[:8]}...")


def get_or_create_session_id():
    """Get existing session ID or create a new one"""
    if 'search_session_id' not in session:
        session['search_session_id'] = str(uuid.uuid4())
    return session['search_session_id']


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
        tool_name = tool_call.get('tool')
        result = tool_call.get('result', {})

        # Extract destinations from various tool types
        if tool_name in ['search_destinations_quantitative', 'filter_by_family_friendly', 'filter_by_city']:
            destinations = result.get('destinations', [])
            if destinations:
                all_destinations = destinations  # Replace with latest filtered results

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
        "city": string (city name, optional, max 30 chars),
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
        city = data.get('city')
        duration = data.get('duration')
        budget = data.get('budget')

        # Validate required fields
        if not destination_type:
            return jsonify({
                'success': False,
                'error': 'Destination type is required'
            }), 400

        # City is optional, but if provided, must be <= 30 characters
        if city is not None and len(city) > 30:
            return jsonify({
                'success': False,
                'error': 'City name must be 30 characters or less'
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
        print(f"   City: {city}" if city is not None else "   City: Not specified")
        print(f"   Duration: {duration} hours" if duration is not None else "   Duration: Not specified")
        print(f"   Budget: ₹{budget:,}" if budget is not None else "   Budget: Not specified")

        # Get or create session ID before starting processing
        session_id = get_or_create_session_id()

        # Mark processing as started for this session
        start_processing(session_id)

        try:
            # Call the agent with cancellation checker
            result = agent.suggest_destinations(
                destination_type=destination_type,
                city=city,
                hours=duration,
                budget=budget,
                cancellation_checker=lambda: is_processing_cancelled(session_id)
            )
        finally:
            # Always end processing when done (success or failure)
            end_processing(session_id)

        # Check if processing was cancelled
        if result.get('cancelled', False):
            return jsonify({
                'success': False,
                'message': 'Session restarted. Please submit a new search.',
                'cancelled': True
            }), 400

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

        # Clean up expired cache entries
        clean_expired_cache()

        # Store results in server-side cache (session_id already obtained above)
        # Only cache if processing wasn't cancelled
        if not is_processing_cancelled(session_id):
            destinations_cache[session_id] = {
                'destinations': sorted_destinations,
                'current_page': 0,
                'expires': datetime.now() + timedelta(hours=CACHE_EXPIRY_HOURS)
            }

        # Get first 3 destinations
        page_size = 3
        displayed_destinations = sorted_destinations[:page_size]
        has_more = len(sorted_destinations) > page_size

        # Check if this is the only batch (first batch is also last batch)
        is_single_batch = len(sorted_destinations) > 0 and not has_more

        # Return the response
        response_data = {
            'success': True,
            'destinations': displayed_destinations,
            'has_more': has_more,
            'total_count': len(sorted_destinations),
            'query': result['query'],
            'tools_used': len(result.get('tools_used', [])),
            'iterations': result.get('iterations', 0)
        }

        # Add end of results flag and message if this is a single batch
        if is_single_batch:
            response_data['end_of_results'] = True
            response_data['message'] = "End of Suggestions, feel free to Restart your search."

        return jsonify(response_data)

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
        # Get session ID
        session_id = session.get('search_session_id')

        if not session_id or session_id not in destinations_cache:
            return jsonify({
                'success': False,
                'error': 'No active search. Please submit a new search first.'
            }), 400

        # Get cached data
        cache_entry = destinations_cache[session_id]
        sorted_destinations = cache_entry['destinations']
        current_page = cache_entry['current_page']

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

        # Update cache
        cache_entry['current_page'] = next_page

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


@app.route('/api/restart', methods=['POST'])
def restart_search():
    """
    Restart search - cancels ongoing processing, clears backend cache, and generates new session ID

    This ensures the backend stops all processing and is ready for a fresh form submission.
    """
    try:
        # Get current session ID
        old_session_id = session.get('search_session_id')

        if old_session_id:
            # CRITICAL: Cancel any ongoing processing for this session
            cancel_processing(old_session_id)

            # Clear the cache for the current session if it exists
            if old_session_id in destinations_cache:
                del destinations_cache[old_session_id]
                print(f"🔄 Cleared cache for session: {old_session_id[:8]}...")

        # Generate a new session ID to ensure a clean slate
        session['search_session_id'] = str(uuid.uuid4())
        new_session_id = session['search_session_id']
        print(f"✨ Created new session: {new_session_id[:8]}...")

        return jsonify({
            'success': True,
            'message': 'Backend reset successfully. Ongoing processing cancelled. Ready for new search.'
        })

    except Exception as e:
        print(f"❌ Error restarting search: {str(e)}")
        import traceback
        traceback.print_exc()

        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/image', methods=['GET'])
def get_destination_image():
    """
    Proxy endpoint to fetch destination images without CORS issues

    Query parameters:
    - query: Search query for the image (e.g., "Taj Mahal India")

    Returns:
        Image data with proper CORS headers or base64 encoded image
    """
    try:
        query = request.args.get('query', '')

        if not query:
            return jsonify({
                'success': False,
                'error': 'Query parameter is required'
            }), 400

        # Check for Unsplash API key
        unsplash_access_key = os.getenv('UNSPLASH_ACCESS_KEY')

        if unsplash_access_key:
            # Use Unsplash API for destination-specific images
            try:
                search_url = f"https://api.unsplash.com/search/photos"
                params = {
                    'query': query,
                    'per_page': 1,
                    'orientation': 'landscape',
                    'content_filter': 'high'
                }
                headers = {
                    'Authorization': f'Client-ID {unsplash_access_key}'
                }

                response = requests.get(search_url, params=params, headers=headers, timeout=10)
                response.raise_for_status()

                data = response.json()

                if data.get('results') and len(data['results']) > 0:
                    # Get the regular size image URL
                    image_url = data['results'][0]['urls']['regular']

                    # Fetch the actual image
                    img_response = requests.get(image_url, timeout=10)
                    img_response.raise_for_status()

                    # Return image as base64
                    import base64
                    img_base64 = base64.b64encode(img_response.content).decode('utf-8')

                    return jsonify({
                        'success': True,
                        'image': f"data:image/jpeg;base64,{img_base64}",
                        'source': 'unsplash'
                    })
                else:
                    # No results from Unsplash, fall through to backup
                    print(f"⚠️ No Unsplash results for query: {query}")
            except Exception as e:
                print(f"⚠️ Unsplash API error: {str(e)}")
                # Fall through to backup service

        # Fallback to Picsum (random but reliable images)
        # Use a deterministic seed based on query for consistent images
        seed = abs(hash(query)) % 1000
        picsum_url = f"https://picsum.photos/seed/{seed}/800/600"

        try:
            img_response = requests.get(picsum_url, timeout=10)
            img_response.raise_for_status()

            # Return image as base64
            import base64
            img_base64 = base64.b64encode(img_response.content).decode('utf-8')

            return jsonify({
                'success': True,
                'image': f"data:image/jpeg;base64,{img_base64}",
                'source': 'picsum'
            })
        except Exception as e:
            print(f"❌ Error fetching from Picsum: {str(e)}")
            return jsonify({
                'success': False,
                'error': 'Failed to fetch image'
            }), 500

    except Exception as e:
        print(f"❌ Error in image proxy: {str(e)}")
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
