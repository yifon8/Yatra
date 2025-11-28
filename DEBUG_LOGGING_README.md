# Enhanced Debugging Infrastructure for Family-Friendly Filter

## Overview

This update adds comprehensive logging infrastructure to help diagnose issues with the family-friendly filter web search functionality, specifically for analyzing 23 beach type destinations.

## What's Been Added

### 1. Comprehensive Logging in `agent/tools.py`

The `TravelTools` class now includes detailed logging throughout the `filter_by_family_friendly` method:

**Global Logger Setup:**
```python
import logging
import sys

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),      # Console output
        logging.FileHandler('yatra_debug.log', mode='w')  # File output
    ]
)
```

**What Gets Logged:**

1. **Initialization Phase:**
   - Total destinations to analyze
   - Normalized destination names
   - API key validation
   - Gemini client initialization

2. **Per-Destination Processing:**
   - Current progress (e.g., "Processing 5/23")
   - Location context for web search
   - Each API call attempt
   - API call duration timing
   - Response text length and preview
   - JSON parsing success/failure
   - Family-friendly decision with reasoning
   - Total processing time per destination

3. **Error Handling:**
   - JSON parsing errors with raw response
   - API quota errors with retry logic
   - Network failures
   - Fatal errors with stack traces

4. **Summary Statistics:**
   - Total processing time
   - Average time per destination
   - Number of family-friendly destinations found

### 2. New Diagnostic Script: `test_beach_family_filter_debug.py`

A comprehensive test script that:
- Searches for beach-type destinations
- Tests family-friendly filter with 23 beaches
- Provides real-time console feedback
- Generates detailed logs in `yatra_debug.log`
- Shows formatted results with analysis

## How to Use

### Prerequisites

1. **Set your Google API key:**
   ```bash
   export GOOGLE_API_KEY="your-api-key-here"
   ```

2. **Install dependencies:**
   ```bash
   pip install google-genai google-generativeai pandas numpy
   ```

### Running the Diagnostic Test

```bash
python3 test_beach_family_filter_debug.py
```

### Monitoring Progress

**Option 1: Watch the console**
The script provides real-time progress updates showing:
- Which destination is being processed
- Current progress count (e.g., 5/23)
- Family-friendly decisions as they're made

**Option 2: Monitor the log file**
```bash
# In another terminal
tail -f yatra_debug.log
```

**Option 3: Check logs after completion**
```bash
cat yatra_debug.log
```

## Log File Format

The `yatra_debug.log` file contains entries like:

```
2025-11-28 17:52:54,585 - agent.tools - INFO - ================================================================================
2025-11-28 17:52:54,585 - agent.tools - INFO - STARTING FAMILY-FRIENDLY FILTER
2025-11-28 17:52:54,586 - agent.tools - INFO - Total destinations to analyze: 23
2025-11-28 17:52:54,586 - agent.tools - INFO - ================================================================================
2025-11-28 17:52:54,586 - agent.tools - INFO - Normalized destination names: ['Chowpatty Beach', ...]
2025-11-28 17:52:54,586 - agent.tools - INFO - Initializing Gemini client...
2025-11-28 17:52:54,586 - agent.tools - INFO - Gemini client initialized successfully
...
2025-11-28 17:52:55,123 - agent.tools - INFO - Processing destination 1/23: Chowpatty Beach
2025-11-28 17:52:55,124 - agent.tools - INFO - Location context: Chowpatty Beach, Mumbai, Maharashtra, India
2025-11-28 17:52:55,124 - agent.tools - INFO - Attempt 1/4 for Chowpatty Beach
2025-11-28 17:52:55,124 - agent.tools - INFO - Calling Gemini API (model: gemini-2.5-flash-lite) for Chowpatty Beach...
2025-11-28 17:52:57,342 - agent.tools - INFO - API call completed in 2.22 seconds
2025-11-28 17:52:57,342 - agent.tools - INFO - Successfully parsed JSON for Chowpatty Beach
2025-11-28 17:52:57,342 - agent.tools - INFO - Analysis result: is_family_friendly=True, confidence=high
2025-11-28 17:52:57,342 - agent.tools - INFO - ✓ Chowpatty Beach is family-friendly! Adding to results.
2025-11-28 17:52:57,343 - agent.tools - INFO - Completed Chowpatty Beach in 2.22 seconds
2025-11-28 17:52:57,343 - agent.tools - INFO - Progress: 1/23 destinations processed, 1 family-friendly so far
...
```

## Troubleshooting

### If the process doesn't start:
- Check that `GOOGLE_API_KEY` environment variable is set
- Verify the API key is valid
- Ensure all dependencies are installed

### If the process hangs on a specific destination:
- Check the log file for the last processed destination
- Look for API quota errors
- Check for network connectivity issues

### If you see quota errors:
The system automatically retries with exponential backoff:
- Attempt 1: immediate
- Attempt 2: after 2 seconds
- Attempt 3: after 4 seconds
- Attempt 4: after 8 seconds

If quota is exhausted, consider:
- Waiting a few minutes before retrying
- Using a different API key
- Processing destinations in smaller batches

## About Google ADK Usage

**Note:** While the request mentioned `google.adk` with `InMemoryRunner` and `LoggingPlugin`, the actual Google ADK is accessed via:
- `from google import genai` (already imported in tools.py)
- `from google.genai import types`

The `google.genai` package **is** the Google AI Development Kit (ADK) / Google GenAI SDK. There isn't a separate `google.adk` module in the current version. The comprehensive logging infrastructure we've added using Python's built-in `logging` module provides the same debugging capabilities.

## Next Steps

1. **Set your API key** and run the diagnostic script
2. **Monitor the logs** to identify where the process might be hanging
3. **Analyze the timing** data to see if specific destinations take longer
4. **Check for patterns** in failed vs successful API calls

The enhanced logging will help pinpoint:
- Which destination is causing issues
- Whether it's an API timeout
- Whether it's a quota limit
- Whether it's a parsing error
- How long each web search is taking
