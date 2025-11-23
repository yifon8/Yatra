# Yatra Travel Agent 🛕

An AI-powered travel destination suggester for India using Google's Gemini AI (Google ADK) and real Kaggle datasets.

## Overview

Yatra is a family-friendly travel agent that helps you find the perfect Indian destinations based on your preferences. It combines:

- **Quantitative filtering**: Budget, duration, and destination type using pandas
- **Qualitative analysis**: LLM-powered evaluation of family-friendliness, atmosphere, and suitability
- **Smart recommendations**: Context-aware suggestions with reasoning

## Features

- 🔍 **Smart Search**: Filter destinations by type, budget, and duration
- 👨‍👩‍👧 **Family-Friendly**: Special considerations for families with small children
- 🤖 **AI-Powered**: Uses Google's Gemini AI for qualitative judgments
- 📊 **Data-Driven**: Works with real Kaggle travel datasets
- 💬 **Multiple Modes**: Interactive, chat, or command-line interface

## Project Structure

```
Yatra/
├── agent/
│   ├── __init__.py                # Package initialization
│   ├── destination_suggester.py   # Main agent logic (Google ADK)
│   ├── system_prompts.py           # LLM prompts for analysis
│   ├── tools.py                    # Tool definitions and implementations
│   └── dataset_handler.py          # Pandas operations on CSV data
├── data/
│   └── destinations.csv            # Kaggle dataset (you need to download)
├── evaluation/
│   └── travel_agent.evalset.json  # Test cases for evaluation
├── run_agent.py                    # CLI entry point
├── requirements.txt                # Python dependencies
└── index.html                      # Frontend interface
```

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Get Google API Key

1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Set it as an environment variable:

```bash
export GOOGLE_API_KEY='your-api-key-here'
```

Or pass it directly when running:

```bash
python run_agent.py --api-key 'your-api-key-here'
```

### 3. Download the Dataset

1. Go to [Kaggle: Travel Dataset - Guide to India's Must See Places](https://www.kaggle.com/datasets/saketk511/travel-dataset-guide-to-indias-must-see-places)
2. Download the CSV file
3. Place it in the `data/` directory as `destinations.csv`

```bash
# Expected path
data/destinations.csv
```

**Note**: You'll need a Kaggle account to download the dataset.

### 4. Run the Agent

**Interactive Mode** (Recommended):
```bash
python run_agent.py -i
```

**Chat Mode**:
```bash
python run_agent.py -c
```

**Command Line Mode**:
```bash
python run_agent.py --type beach --child --hours 8 --budget 5000
```

## Usage Examples

### Interactive Mode

```bash
$ python run_agent.py -i

╔══════════════════════════════════════════════════════════╗
║                                                          ║
║   🛕  YATRA - Your Family's Travel Agent for India  🛕   ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝

Please answer a few questions:

1. What type of destination are you interested in?
   a) Beach & Coastal
   b) Mountain & Hill Stations
   c) Heritage & Cultural
   d) Wildlife & Nature
   e) Any type

Your choice (a/b/c/d/e): a

2. Are you traveling with a small child?
   (yes/no): yes

3. How many hours do you have for the visit?
   Hours (or press Enter to skip): 8

4. What is your budget in Indian Rupees?
   Budget ₹ (or press Enter to skip): 5000

🔍 Searching for the best destinations...
```

### Chat Mode

```bash
$ python run_agent.py -c

You: I want to visit a peaceful mountain destination with my family

Yatra: I'd love to help you find a peaceful mountain destination!
Let me search through our database of Indian hill stations...

[Agent provides recommendations with reasoning]
```

### Command Line Mode

```bash
# Beach destination for family with child, 8 hours, ₹5000 budget
python run_agent.py --type beach --child --hours 8 --budget 5000

# Heritage site without time/budget constraints
python run_agent.py --type heritage

# Wildlife destination with specific budget
python run_agent.py --type wildlife --budget 10000
```

## How It Works

### 1. Quantitative Filtering (Pandas)

The agent first filters destinations using hard criteria:

```python
# In dataset_handler.py
filtered = dataset.search_quantitative(
    destination_type="beach",
    max_budget=5000,
    max_hours=8
)
```

### 2. Qualitative Analysis (LLM)

Then uses Google's Gemini AI to evaluate:

- Family-friendliness based on descriptions
- Atmosphere and vibe
- Suitability for specific user needs
- Practical considerations

### 3. Tool Calling (Google ADK)

The agent uses Google ADK's function calling to:

1. Search destinations with filters
2. Get detailed information
3. Analyze family-friendliness
4. Provide recommendations with reasoning

```python
# Tools available to the agent
- search_destinations_quantitative()
- get_destination_details()
- get_dataset_summary()
- filter_by_family_friendly()
```

## Dataset Structure

The agent expects a CSV with columns related to Indian destinations. Common columns include:

- `name` / `place`: Destination name
- `type` / `category`: Type (beach, mountain, heritage, wildlife)
- `description`: Detailed description (for qualitative analysis)
- `budget` / `cost`: Estimated cost
- `duration` / `hours`: Recommended visit duration
- `state` / `location`: Geographic location
- `activities`: Available activities
- `amenities`: Facilities available

The `DatasetHandler` class automatically detects and adapts to different column names.

## Customization

### Using Your Own Dataset

```bash
python run_agent.py -i --csv path/to/your/dataset.csv
```

### Modifying System Prompts

Edit `agent/system_prompts.py` to customize how the agent analyzes destinations:

```python
FAMILY_FRIENDLINESS_PROMPT = """
Analyze this destination for family-friendliness...
[Your custom criteria]
"""
```

### Adding New Tools

Add new tools in `agent/tools.py`:

```python
def get_tool_declarations(self):
    return [
        # ... existing tools
        {
            "name": "your_new_tool",
            "description": "What your tool does",
            "parameters": {
                # parameter definitions
            }
        }
    ]
```

### Changing the Model

Edit `agent/destination_suggester.py`:

```python
self.model = genai.GenerativeModel(
    model_name='gemini-1.5-pro',  # or 'gemini-1.5-flash'
    system_instruction=AGENT_SYSTEM_PROMPT
)
```

- `gemini-1.5-flash`: Faster, cheaper
- `gemini-1.5-pro`: More capable, better analysis

## Evaluation

Test the agent with the evaluation set:

```python
from agent import DestinationSuggester
import json

# Load evaluation cases
with open('evaluation/travel_agent.evalset.json') as f:
    eval_set = json.load(f)

# Initialize agent
agent = DestinationSuggester()

# Run test cases
for test_case in eval_set['test_cases']:
    print(f"\nTest: {test_case['description']}")

    result = agent.suggest_destinations(
        **test_case['input']
    )

    print(f"Recommendations:\n{result['recommendations']}")
    print(f"Expected criteria: {test_case['expected_criteria']}")
```

## API Reference

### DestinationSuggester

Main agent class for suggesting destinations.

```python
from agent import DestinationSuggester

agent = DestinationSuggester(
    api_key='your-google-api-key',
    csv_path='data/destinations.csv'
)
```

#### Methods

**suggest_destinations()**
```python
result = agent.suggest_destinations(
    destination_type='beach',    # 'beach', 'mountain', 'heritage', 'wildlife'
    has_small_child=True,        # Boolean
    hours=8,                     # Integer
    budget=5000                  # Float (Rupees)
)
```

Returns:
```python
{
    'success': True,
    'recommendations': 'Formatted recommendation text',
    'query': 'Original query',
    'tools_used': [...],
    'iterations': 3
}
```

**chat()**
```python
response = agent.chat("I want a peaceful beach destination")
```

**analyze_destination()**
```python
analysis = agent.analyze_destination(
    destination_name="Goa",
    has_small_child=True
)
```

### DatasetHandler

Handles pandas operations on the dataset.

```python
from agent import DatasetHandler

dataset = DatasetHandler('data/destinations.csv')

# Filter by type
beach_destinations = dataset.filter_by_type('beach')

# Filter by budget
affordable = dataset.filter_by_budget(max_budget=5000)

# Combined search
results = dataset.search_quantitative(
    destination_type='mountain',
    max_budget=10000,
    max_hours=12
)
```

## Troubleshooting

### "Dataset not found" error

Make sure you've downloaded the Kaggle dataset and placed it at `data/destinations.csv`.

### "Google API key required" error

Set your API key:
```bash
export GOOGLE_API_KEY='your-key'
```

### Import errors

Install dependencies:
```bash
pip install -r requirements.txt
```

### Column not found errors

The dataset structure might be different. Check columns:
```python
from agent import DatasetHandler
dataset = DatasetHandler()
print(dataset.get_column_names())
```

Then adjust `dataset_handler.py` to match your dataset's column names.

## Contributing

Ideas for improvements:

1. **Add more tools**: Weather data, distance calculations, user reviews
2. **Multi-city itineraries**: Plan entire trips with multiple destinations
3. **Seasonal recommendations**: Consider best time to visit
4. **Image generation**: Generate images of destinations
5. **Cost breakdown**: Detailed budget planning
6. **Accessibility features**: Consider mobility constraints

## License

This project uses:
- Google Generative AI SDK
- Kaggle dataset (see dataset license on Kaggle)

## Resources

- [Google AI Studio](https://makersuite.google.com/)
- [Google Generative AI Python SDK](https://github.com/google/generative-ai-python)
- [Kaggle Dataset](https://www.kaggle.com/datasets/saketk511/travel-dataset-guide-to-indias-must-see-places)
- [Pandas Documentation](https://pandas.pydata.org/)

## Support

For issues or questions:
1. Check this README
2. Review the evaluation examples in `evaluation/`
3. Check dataset column names match your CSV
4. Ensure Google API key is set correctly

Happy travels with Yatra! 🌏✈️
