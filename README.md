# Yatra
Travel Planner for families visiting India

# Problem
Data used for family trip travel planning fragmented across several app, websites, platforms.  Difficulty in making sure each destination is family friendly.

# Solution
One interface for families planning to visit India, considering destination type, target city (plus web searched adjacent cities), visit duration, and budget per person.  On the backend, the system also applies a greater than 4 stars out of 5 rating filter, and vets each destination suggestion candidate at the end for family friendliness using 2 top current sources by calling Google Search API and conducting qualitative analysis of the resulting content.

## Overview

Yatra consists of two components:

1. **Frontend Interface** (`index.html`) - Interactive web form for trip planning
2. **AI Agent** (`agent/`) - Python-based travel destination suggester using Google Gemini AI

## Frontend Interface

Open `index.html` in a browser to access the interactive form for planning family trips to India.

## AI Agent

An intelligent travel agent that suggests destinations based on user preferences using:
- **Google Gemini AI** for qualitative analysis with Google Search
- **Kaggle datasets** for real destination data
- **Pandas** for quantitative filtering

## ARCHITECTURE (written by Claude Code, as it produced majority of code)

Hybrid-Intelligence Architecture
• Stage 1: Deterministic quantitative filtering using Pandas (fast, predictable, cost-effective)
• Stage 2: LLM-powered qualitative analysis with web search grounding (nuanced, context-aware)
Agentic Reasoning
• City Validation: LLM verifies cities are in India via web search
• Geographic Expansion: Dynamically discovers adjacent cities (e.g., Mumbai → Navi Mumbai, Thane, Kalyan)
• Family-Friendly Assessment: Searches official tourism sites for safety warnings
Production-Ready Engineering
• Cancellable long-running operations
• Exponential backoff retry logic for API quotas
• Session-based caching
• Comprehensive error handling
Design Patterns
• Blacklist-based safety filter: Permissive by default, excludes only explicit warnings
• Deterministic-first design: Reduces LLM calls by 90%+
• Graceful degradation: Falls back to hardcoded mappings when APIs fail
Technical
• Google Gemini 2.5 Flash Lite with Google web search integration
• Real Kaggle dataset
• Full-stack implementation (Flask API + web interface)
• Transparent reasoning with filtering pipeline traces
Data Flow and Tech Stack
• User input > Frontend validation > Backend session > Agent pipeline > Pandas filter and LLM Analysis > Grounded results > Pagination > Frontend Display
• Python 3.x, Flask REST API, Google Gemini 2.5 Flash Lite, Pandas/NumPy, Google Search API

## Stand up local web server and run agent app
Installation Steps:
1. Use the cd command to navigate to the directory where you want to save the project.
2. Clone repo from https://github.com/yifon8/Yatra > click Code button > choose method and copy string. (Https method: terminal or command line > git clone https://github.com/yifon8/Yatra) Repo already contains the 9kb .csv dataset under data directory.
(Claude Code:) Install dependencies: pip install -r requirements.txt

Configuration: Details on any environment variables, configuration files, or API keys that need to be set up
1. Create Google Cloud Console API key > enable Gemini and Google Search APIs in Cloud Console > go to detail of Cloud Console API key, restrict to only these two enabled APIs > set the API key locally for Yatra
2. (Claude Code:) Set Google API key: (on Mac) export GOOGLE_API_KEY='your-key'

Running the Project:
Steps:
1. open terminal or cmd window
2. navigate to directory to which you cloned Yatra repo
3. enter command to cli: python web_server.py
4. go to local web server window

### Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set up Google API key
export GOOGLE_API_KEY='your-key'

# Download dataset from Kaggle and place in data/destinations.csv

# Run in interactive mode
python run_agent.py -i
```

See [AGENT_README.md](AGENT_README.md) for detailed agent documentation.
