# Yatra Web UI - Usage Guide

## Overview

The Yatra web UI provides a user-friendly interface to get personalized travel destination recommendations for India based on your preferences.

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Your Google API Key

You need a Google API key to use the Gemini AI model:

```bash
export GOOGLE_API_KEY='your-api-key-here'
```

Get your API key at: https://makersuite.google.com/app/apikey

### 3. Ensure Dataset is Available

Make sure you have the destinations dataset at `data/destinations.csv`.

Download from: https://www.kaggle.com/datasets/saketk511/travel-dataset-guide-to-indias-must-see-places

## Running the Web Server

Start the Flask web server:

```bash
python web_server.py
```

This will start the server at `http://localhost:5000`

## Using the Web Interface

### Form Inputs

The web UI captures the following user inputs:

1. **Destination Type** (dropdown menu)
   - Variable name: `destinationType`
   - Options:
     - Beach & Coastal (`beach`)
     - Mountain & Hill Stations (`mountain`)
     - Heritage & Cultural (`heritage`)
     - Wildlife & Nature (`wildlife`)

2. **Small Child** (checkbox)
   - Variable name: `smallChild`
   - Type: Boolean (true/false)
   - Indicates if you're traveling with a small child

3. **Visit Duration** (positive integer)
   - Variable name: `duration`
   - Units: hours
   - Type: Positive integer
   - How many hours you have for the visit

4. **Budget** (positive integer)
   - Variable name: `budget`
   - Units: Indian Rupees (₹)
   - Type: Positive integer
   - Your budget for the trip

### How It Works

1. Fill in all the form fields:
   - Select a destination type from the dropdown
   - Check the "Small Child" box if traveling with a small child
   - Enter the visit duration in hours
   - Enter your budget in Rupees

2. Click the "Help me, Yatra!" button

3. The form data is validated:
   - Destination type must be selected
   - Duration must be a positive number
   - Budget must be a positive number

4. If validation passes, the data is sent to the backend API at `/api/suggest`

5. The Yatra agent processes your request using:
   - Google Gemini AI model
   - The travel destinations dataset
   - Intelligent tool calling to search and filter destinations

6. Recommendations appear in the results panel on the right

7. Use the "Retry" button to clear the form and start over

## API Endpoint

### POST /api/suggest

Request body (JSON):
```json
{
  "destinationType": "beach",
  "smallChild": true,
  "duration": 8,
  "budget": 5000
}
```

Response (JSON):
```json
{
  "success": true,
  "recommendations": "Based on your preferences...",
  "query": "I'm looking for travel destination recommendations...",
  "tools_used": 2,
  "iterations": 2
}
```

## JavaScript Functions

The following JavaScript functions are available in the web UI:

- `getFormData()` - Extracts and returns all form values as an object
- `validateFormData(data)` - Validates form inputs and returns array of errors
- `submitForm()` - Handles form submission and API communication
- `displayRecommendations(recommendations)` - Formats and displays AI recommendations

## Variable Mapping

| UI Element | HTML ID | JavaScript Variable | Backend Parameter |
|------------|---------|---------------------|-------------------|
| Destination dropdown | `destination` | `destinationType` | `destination_type` |
| Small child checkbox | `smallChild` | `smallChild` | `has_small_child` |
| Hours input | `hours` | `duration` | `hours` |
| Budget input | `budget` | `budget` | `budget` |

## Troubleshooting

### "Failed to get recommendations"
- Make sure the Flask server is running (`python web_server.py`)
- Check that your Google API key is set correctly
- Verify the dataset is available at `data/destinations.csv`

### Validation errors
- Ensure all required fields are filled
- Duration and budget must be positive numbers
- A destination type must be selected

### Server errors
- Check the terminal where the server is running for error messages
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Verify your Google API key is valid

## Example Usage

1. Select "Beach & Coastal" from the destination dropdown
2. Check "Small Child" if traveling with a child
3. Enter "8" in the Visit Duration field (8 hours)
4. Enter "5000" in the Budget field (₹5,000)
5. Click "Help me, Yatra!"
6. View personalized recommendations in the results panel
