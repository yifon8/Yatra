#!/usr/bin/env python3
"""
Yatra Web Server - Flask API for the web interface
Handles form submissions and connects to the DestinationSuggester agent
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import sys
from typing import Dict, Any

from agent import DestinationSuggester

app = Flask(__name__, static_folder='.')
CORS(app)

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


@app.route('/api/suggest', methods=['POST'])
def suggest_destinations():
    """
    Handle destination suggestion requests

    Expected JSON body:
    {
        "destinationType": "beach|mountain|heritage|wildlife",
        "smallChild": true|false,
        "duration": integer (hours),
        "budget": integer (Rupees)
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
        small_child = data.get('smallChild', False)
        duration = data.get('duration')
        budget = data.get('budget')

        # Validate required fields
        if not destination_type:
            return jsonify({
                'success': False,
                'error': 'Destination type is required'
            }), 400

        if not duration or duration <= 0:
            return jsonify({
                'success': False,
                'error': 'Valid duration (positive hours) is required'
            }), 400

        if not budget or budget <= 0:
            return jsonify({
                'success': False,
                'error': 'Valid budget (positive amount) is required'
            }), 400

        # Log the request
        print(f"\n📝 Received request:")
        print(f"   Destination Type: {destination_type}")
        print(f"   Small Child: {small_child}")
        print(f"   Duration: {duration} hours")
        print(f"   Budget: ₹{budget:,}")

        # Call the agent
        result = agent.suggest_destinations(
            destination_type=destination_type,
            has_small_child=small_child,
            hours=duration,
            budget=budget
        )

        # Return the response
        return jsonify({
            'success': True,
            'recommendations': result['recommendations'],
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
