# Yatra
Travel Planner for families visiting India

## Overview

Yatra consists of two components:

1. **Frontend Interface** (`index.html`) - Interactive web form for trip planning
2. **AI Agent** (`agent/`) - Python-based travel destination suggester using Google Gemini AI

## Frontend Interface

Open `index.html` in a browser to access the interactive form for planning family trips to India.

## AI Agent

An intelligent travel agent that suggests destinations based on user preferences using:
- **Google Gemini AI** for qualitative analysis
- **Kaggle datasets** for real destination data
- **Pandas** for quantitative filtering

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
