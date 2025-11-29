"""
Tools for Yatra Travel Agent using Google ADK
Function declarations and implementations for the agent
"""

from typing import List, Dict, Any, Optional
from .dataset_handler import DatasetHandler
import json
from google import genai
from google.genai import types
import os
import pandas as pd
import logging
import time
import sys

# Configure comprehensive logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('yatra_debug.log', mode='w')
    ]
)

logger = logging.getLogger(__name__)

# Fallback dictionary for neighboring cities when API is unavailable
# This ensures city filtering still works even when Google API key is missing or API fails
NEIGHBORING_CITIES_FALLBACK = {
    # National Capital Region (NCR)
    "delhi": ["Gurgaon", "Gurugram", "Noida", "Ghaziabad", "Faridabad"],
    "new delhi": ["Gurgaon", "Gurugram", "Noida", "Ghaziabad", "Faridabad"],
    "gurgaon": ["Delhi", "Faridabad", "Noida"],
    "gurugram": ["Delhi", "Faridabad", "Noida"],
    "noida": ["Delhi", "Ghaziabad", "Greater Noida"],
    "ghaziabad": ["Delhi", "Noida", "Meerut"],
    "faridabad": ["Delhi", "Gurgaon", "Gurugram"],
    "greater noida": ["Noida", "Delhi"],

    # Mumbai Metropolitan Region (MMR)
    "mumbai": ["Navi Mumbai", "Thane", "Kalyan", "Panvel"],
    "navi mumbai": ["Mumbai", "Thane", "Panvel"],
    "thane": ["Mumbai", "Navi Mumbai", "Kalyan", "Bhiwandi"],
    "kalyan": ["Mumbai", "Thane", "Dombivli"],
    "panvel": ["Mumbai", "Navi Mumbai"],
    "vasai-virar": ["Mumbai", "Thane"],
    "mira-bhayandar": ["Mumbai", "Thane"],
    "bhiwandi": ["Thane", "Kalyan"],

    # Bangalore (Bengaluru) Metropolitan Area
    "bangalore": ["Mysore", "Tumkur", "Hosur"],
    "bengaluru": ["Mysore", "Tumkur", "Hosur"],
    "mysore": ["Bangalore", "Bengaluru"],
    "mysuru": ["Bangalore", "Bengaluru"],

    # Hyderabad Metropolitan Area
    "hyderabad": ["Secunderabad", "Cyberabad"],
    "secunderabad": ["Hyderabad"],

    # Chennai Metropolitan Area
    "chennai": ["Kanchipuram", "Tiruvallur", "Chengalpattu"],

    # Kolkata Metropolitan Area
    "kolkata": ["Howrah", "Salt Lake"],
    "howrah": ["Kolkata"],

    # Pune Metropolitan Region
    "pune": ["Pimpri-Chinchwad", "Lonavala", "Khadki"],
    "pimpri-chinchwad": ["Pune"],

    # Ahmedabad Metropolitan Area
    "ahmedabad": ["Gandhinagar"],
    "gandhinagar": ["Ahmedabad"],

    # Jaipur Metropolitan Area
    "jaipur": ["Ajmer"],

    # Lucknow Metropolitan Area
    "lucknow": ["Kanpur"],

    # Kochi Metropolitan Area
    "kochi": ["Ernakulam"],
    "ernakulam": ["Kochi"],
}


class TravelTools:
    """Collection of tools for the travel agent to use with Google ADK"""

    def __init__(self, dataset_handler: DatasetHandler):
        """
        Initialize tools with dataset handler

        Args:
            dataset_handler: Instance of DatasetHandler for data operations
        """
        self.dataset = dataset_handler

    @staticmethod
    def _get_google_search_tool():
        """
        Create Google Search tool for use with Gemini models

        Returns:
            List containing configured Google Search tool
        """
        return [types.Tool(google_search=types.GoogleSearch())]

    @staticmethod
    def _get_neighboring_cities_fallback(city: str) -> List[str]:
        """
        Get neighboring cities from hardcoded fallback dictionary
        Used when Google API is unavailable or fails

        Args:
            city: Target city name

        Returns:
            List of neighboring cities, or empty list if city not in fallback
        """
        city_lower = city.strip().lower()
        return NEIGHBORING_CITIES_FALLBACK.get(city_lower, [])

    def get_tool_declarations(self) -> List[Dict[str, Any]]:
        """
        Get Google ADK tool declarations for all available tools

        Returns:
            List of tool declarations in Google ADK format
        """
        return [
            {
                "name": "search_destinations_quantitative",
                "description": "Search for destinations using quantitative filters. REQUIRED: destination_type must be specified. SYSTEM FILTERS: Automatically filters by rating >=4.0. Returns a list of matching destinations.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "destination_type": {
                            "type": "string",
                            "description": "Type of destination: 'beach', 'mountain', 'heritage', 'wildlife'. REQUIRED parameter.",
                            "enum": ["beach", "mountain", "heritage", "wildlife"]
                        },
                        "max_budget": {
                            "type": "number",
                            "description": "Maximum budget in Indian Rupees (₹)"
                        },
                        "max_hours": {
                            "type": "number",
                            "description": "Maximum visit duration in hours (can be decimal, e.g., 0.5, 1.5, 8.25)"
                        }
                    },
                    "required": ["destination_type"]
                }
            },
            {
                "name": "get_destination_details",
                "description": "Get detailed information about a specific destination by name. Use this to retrieve full details for qualitative analysis.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "destination_name": {
                            "type": "string",
                            "description": "Name of the destination to get details for"
                        }
                    },
                    "required": ["destination_name"]
                }
            },
            {
                "name": "get_dataset_summary",
                "description": "Get summary statistics about the available destinations dataset, including total count and categories.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            },
            {
                "name": "filter_by_family_friendly",
                "description": "Filter destinations to show only those suitable for families with small children. Uses a blacklist approach - keeps ALL destinations except those with explicit danger warnings or age restrictions like 'adults only', 'no children allowed', or 'dangerous terrain'.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "destination_names": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of destination names to filter"
                        }
                    },
                    "required": ["destination_names"]
                }
            },
            {
                "name": "filter_by_city",
                "description": "Filter a list of destinations by city using LLM with web search to find at most 4 adjacent cities. First uses LLM with web search to find at most 4 cities adjacent to the input city, then filters the provided destinations list to return only those in the input city or adjacent cities (max 5 cities total: target city + 4 bordering cities). This tool should be used AFTER search_destinations_quantitative to filter the results by city.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "Target city name to filter by (e.g., 'Delhi', 'Mumbai', 'Bangalore')"
                        },
                        "destinations": {
                            "type": "array",
                            "items": {"type": "object"},
                            "description": "List of destination objects to filter (typically from search_destinations_quantitative results)"
                        }
                    },
                    "required": ["city", "destinations"]
                }
            }
        ]

    def execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a tool by name with given parameters

        Args:
            tool_name: Name of the tool to execute
            parameters: Parameters for the tool

        Returns:
            Tool execution result
        """
        if tool_name == "search_destinations_quantitative":
            return self.search_destinations_quantitative(**parameters)
        elif tool_name == "get_destination_details":
            return self.get_destination_details(**parameters)
        elif tool_name == "get_dataset_summary":
            return self.get_dataset_summary(**parameters)
        elif tool_name == "filter_by_family_friendly":
            return self.filter_by_family_friendly(**parameters)
        elif tool_name == "filter_by_city":
            return self.filter_by_city(**parameters)
        else:
            return {"error": f"Unknown tool: {tool_name}"}

    def search_destinations_quantitative(self,
                                        destination_type: str,
                                        max_budget: Optional[float] = None,
                                        max_hours: Optional[float] = None) -> Dict[str, Any]:
        """
        Search destinations using quantitative filters with system-level filtering

        SYSTEM FILTERS (always applied):
        - Rating must be >= 4.0
        - Destination type filtering (REQUIRED parameter)

        Args:
            destination_type: Type of destination to filter by (REQUIRED: beach, mountain, heritage, wildlife)
            max_budget: Maximum budget in rupees
            max_hours: Maximum duration in hours (can be decimal, e.g., 0.5, 1.5, 8.25)

        Returns:
            Dictionary with search results
        """
        try:
            # Perform quantitative search with system-level filters
            results_df = self.dataset.search_quantitative(
                destination_type=destination_type,
                max_budget=max_budget,
                max_hours=max_hours
            )

            # Convert to list of dictionaries
            destinations = self.dataset.to_dict_list(results_df)

            # Add note to destinations with unknown entry fees
            for dest in destinations:
                entry_fee_columns = ['entrance_fee_in_inr', 'budget', 'cost', 'price',
                                   'estimated_cost', 'average_cost', 'budget_per_person']

                # Check if entry fee is unknown (empty string or missing)
                has_unknown_fee = True
                for col in entry_fee_columns:
                    if col in dest and dest[col] not in ['', None]:
                        try:
                            # Try to convert to float - if successful and not NaN, fee is known
                            fee_value = float(dest[col])
                            if not (fee_value != fee_value):  # Check for NaN (NaN != NaN is True)
                                has_unknown_fee = False
                                break
                        except (ValueError, TypeError):
                            continue

                # Add note to description if entry fee is unknown
                if has_unknown_fee:
                    description = dest.get('description', '')
                    if description and description != '':
                        dest['description'] = f"{description} [Note: Entry fee for this destination is unknown]"
                    else:
                        dest['description'] = "[Note: Entry fee for this destination is unknown]"

            # Return all matching destinations with compact format to manage payload size
            # Only include essential fields to keep the response compact
            # Note: Agent will filter by city and then select top 25 by rating
            compact_destinations = []
            for dest in destinations:
                compact_dest = {
                    'name': dest.get('name', dest.get('place', 'Unknown')),
                    'type': dest.get('type', ''),
                    'city': dest.get('city', ''),
                    'state': dest.get('state', ''),
                    'description': dest.get('description', '')[:300] if dest.get('description') else '',  # Limit to 300 chars
                    'google_review_rating': dest.get('google_review_rating', ''),
                    'entrance_fee_in_inr': dest.get('entrance_fee_in_inr', ''),
                    'time_needed_to_visit_in_hrs': dest.get('time_needed_to_visit_in_hrs', '')
                }
                compact_destinations.append(compact_dest)

            # Limit to 100 destinations to prevent payload overflow (finish_reason: 12 error)
            # This gives the agent enough options while keeping the response size manageable
            total_matching = len(compact_destinations)
            max_results = 100
            if len(compact_destinations) > max_results:
                compact_destinations = compact_destinations[:max_results]
                note_suffix = f" Limited to {max_results} destinations to manage payload size (total matching: {total_matching})."
            else:
                note_suffix = ""

            return {
                "success": True,
                "count": len(compact_destinations),
                "total_matching": total_matching,
                "destinations": compact_destinations,
                "filters_applied": {
                    "system_filters": "Rating >= 4.0",
                    "type": destination_type,
                    "max_budget": max_budget or "unlimited",
                    "max_hours": max_hours or "unlimited"
                },
                "note": f"Returning {len(compact_destinations)} matching destinations with filters applied (type: {destination_type}, rating >= 4.0).{note_suffix}"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "destinations": []
            }

    def get_destination_details(self, destination_name: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific destination

        Args:
            destination_name: Name of the destination

        Returns:
            Dictionary with destination details
        """
        try:
            details = self.dataset.get_destination_details(destination_name)

            if details:
                return {
                    "success": True,
                    "destination": details
                }
            else:
                return {
                    "success": False,
                    "error": f"Destination '{destination_name}' not found",
                    "destination": {}
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "destination": {}
            }

    def get_dataset_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics about the dataset

        Returns:
            Dictionary with dataset summary
        """
        try:
            summary = self.dataset.get_summary_stats()
            return {
                "success": True,
                "summary": summary
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "summary": {}
            }

    def filter_by_family_friendly(self, destination_names: List[str], cancellation_checker: Optional[callable] = None) -> Dict[str, Any]:
        """
        Filter destinations for family-friendliness using LLM with web search

        This implementation uses an LLM to search the web for information about
        each destination and determine if it's suitable for families with small children.

        BLACKLIST APPROACH (EXCLUSION-BASED):
        The LLM keeps ALL destinations by default and only excludes destinations that have
        explicit warnings or restrictions such as:
        - "No children allowed" or "adults only"
        - Explicit danger warnings for children (dangerous terrain, unsafe for kids)
        - Age restrictions that exclude young children
        - Extreme activities that explicitly state they're unsafe for children
        - Nightlife-only venues (bars, clubs, casinos) with no family-appropriate activities

        The filter is VERY PERMISSIVE - it only excludes destinations when there are
        clear, explicit reasons why children cannot or should not visit. When in doubt,
        the filter defaults to INCLUDING the destination.

        The LLM prioritizes official sources like state tourism board websites and Wikipedia.

        Args:
            destination_names: List of destination names to evaluate (can also accept list of dicts with 'name' key)
            cancellation_checker: Optional callable that returns True if processing should be cancelled

        Returns:
            Dictionary with family-friendly destinations and analysis
        """
        try:
            logger.info(f"=" * 80)
            logger.info(f"STARTING FAMILY-FRIENDLY FILTER")
            logger.info(f"Total destinations to analyze: {len(destination_names)}")
            logger.info(f"=" * 80)

            # Check for cancellation at the start
            if cancellation_checker and cancellation_checker():
                logger.info("🛑 Processing cancelled before family filter started")
                return {
                    'success': False,
                    'cancelled': True,
                    'error': 'Processing was cancelled',
                    'destinations': [],
                    'analysis_log': []
                }

            # Normalize destination_names - handle both strings and dicts
            normalized_names = []
            for item in destination_names:
                if isinstance(item, str):
                    normalized_names.append(item)
                elif isinstance(item, dict):
                    # Extract name from dict (try multiple possible keys)
                    name = item.get('name') or item.get('place') or item.get('destination')
                    if name:
                        normalized_names.append(name)
                    else:
                        return {
                            'success': False,
                            'error': f'Invalid destination object: {item}. Must have a "name", "place", or "destination" field.',
                            'destinations': [],
                            'analysis_log': []
                        }
                else:
                    return {
                        'success': False,
                        'error': f'Invalid destination_names parameter. Expected list of strings or dicts, got: {type(item)}',
                        'destinations': [],
                        'analysis_log': []
                    }

            destination_names = normalized_names

            if not destination_names:
                logger.warning("No destinations to filter")
                return {
                    'success': True,
                    'count': 0,
                    'destinations': [],
                    'analysis_log': [],
                    'method': 'llm_with_web_search'
                }

            logger.info(f"Normalized destination names: {destination_names}")

            # Configure Gemini client with API key
            api_key = os.environ.get('GOOGLE_API_KEY')
            if not api_key:
                logger.error("GOOGLE_API_KEY environment variable not set!")
                return {
                    'success': False,
                    'error': 'GOOGLE_API_KEY environment variable not set',
                    'destinations': [],
                    'analysis_log': []
                }

            logger.info("Initializing Gemini client...")
            client = genai.Client(api_key=api_key)
            logger.info("Gemini client initialized successfully")

            family_friendly = []
            analysis_log = []

            start_time = time.time()

            for idx, name in enumerate(destination_names, 1):
                # Check for cancellation before processing each destination
                if cancellation_checker and cancellation_checker():
                    logger.info(f"🛑 Processing cancelled at destination {idx}/{len(destination_names)}")
                    return {
                        'success': False,
                        'cancelled': True,
                        'error': f'Processing was cancelled after analyzing {idx-1}/{len(destination_names)} destinations',
                        'destinations': family_friendly,  # Return what we have so far
                        'analysis_log': analysis_log
                    }

                logger.info(f"\n{'=' * 80}")
                logger.info(f"Processing destination {idx}/{len(destination_names)}: {name}")
                logger.info(f"{'=' * 80}")
                dest_start_time = time.time()

                details = self.dataset.get_destination_details(name)

                if not details:
                    logger.warning(f"Destination '{name}' not found in dataset")
                    analysis_log.append({
                        "destination": name,
                        "status": "not_found",
                        "reason": "Destination not found in dataset"
                    })
                    continue

                # Get location context for better web search
                city = details.get('city', '')
                state = details.get('state', '')
                location_context = f"{name}, {city}, {state}, India" if city and state else f"{name}, India"
                logger.info(f"Location context: {location_context}")

                # Create prompt for LLM to analyze family-friendliness
                prompt = f"""You are analyzing tourist destinations in India to identify those that should be EXCLUDED from a family-friendly list.

Destination to analyze: {location_context}

Please search the web for information about this destination.

EXCLUSION CRITERIA - Mark as NOT family-friendly ONLY if you find:
1. Explicit age restrictions: "adults only", "no children allowed", "18+ only"
2. Explicit danger warnings for children: "dangerous terrain for children", "unsafe for kids", "not suitable for small children"
3. Age-restricted venues: bars, nightclubs, casinos with no family-appropriate activities
4. Explicit safety warnings that children cannot visit safely

PRIORITIZE information from:
- Official tourism websites (incredibleindia.org, state tourism sites)
- Wikipedia and travel guides
- Official destination websites
- Reputable travel review sites

DECISION LOGIC (BLACKLIST/EXCLUSION APPROACH):
- Mark as "is_family_friendly": TRUE by default for ALL destinations
- Mark as "is_family_friendly": FALSE ONLY if you find explicit exclusion criteria listed above
- When in doubt or information is unclear, default to TRUE
- General tourist destinations, landmarks, beaches, mountains, heritage sites, wildlife areas should be TRUE unless there are explicit restrictions
- Do NOT exclude destinations just because they don't explicitly mention being family-friendly
- Do NOT exclude destinations just because of general safety precautions that apply to all visitors

IMPORTANT: Be VERY PERMISSIVE. Only exclude destinations with clear, explicit restrictions or danger warnings specifically about children. Most tourist destinations in India are suitable for families by default.

Based on the web search results, determine if this destination should be EXCLUDED from a family-friendly list.

Respond ONLY with a JSON object in this exact format:
{{
    "is_family_friendly": true/false,
    "confidence": "high/medium/low",
    "reasoning": "One short sentence (max 10-15 words) explaining why this is suitable/unsuitable for families",
    "key_activities": ["activity1", "activity2", "activity3"],
    "concerns": ["concern1", "concern2"] or []
}}

Do not include any text before or after the JSON object."""

                # Retry logic with exponential backoff for API quota errors
                max_retries = 3
                retry_delays = [2, 4, 8]  # seconds
                analysis_success = False

                for attempt in range(max_retries + 1):
                    try:
                        logger.info(f"Attempt {attempt + 1}/{max_retries + 1} for {name}")
                        logger.info("Creating GenerateContentConfig with Google Search tool...")

                        # Generate response with grounding (web search)
                        config = types.GenerateContentConfig(
                            temperature=0.0,  # Deterministic responses
                            tools=TravelTools._get_google_search_tool()  # Enable web search with Google Search tool
                        )
                        logger.debug(f"Config created: temperature=0.0, tools=Google Search")

                        logger.info(f"Calling Gemini API (model: gemini-2.5-flash-lite) for {name}...")
                        api_call_start = time.time()

                        response = client.models.generate_content(
                            model='gemini-2.5-flash-lite',
                            contents=prompt,
                            config=config
                        )

                        api_call_duration = time.time() - api_call_start
                        logger.info(f"API call completed in {api_call_duration:.2f} seconds")

                        # Parse the LLM response
                        logger.debug("Parsing LLM response...")
                        response_text = response.text.strip()
                        logger.debug(f"Response text length: {len(response_text)} characters")
                        logger.debug(f"Response preview: {response_text[:200]}...")

                        # Extract JSON from response (handle markdown code blocks if present)
                        if '```json' in response_text:
                            logger.debug("Extracting JSON from markdown code block (```json)")
                            response_text = response_text.split('```json')[1].split('```')[0].strip()
                        elif '```' in response_text:
                            logger.debug("Extracting JSON from markdown code block (```)")
                            response_text = response_text.split('```')[1].split('```')[0].strip()

                        logger.debug(f"Cleaned response text: {response_text[:300]}...")
                        logger.info("Parsing JSON response...")
                        analysis = json.loads(response_text)
                        logger.info(f"Successfully parsed JSON for {name}")
                        logger.info(f"Analysis result: is_family_friendly={analysis.get('is_family_friendly')}, "
                                   f"confidence={analysis.get('confidence')}")

                        # Log the analysis
                        analysis_log.append({
                            "destination": name,
                            "status": "analyzed",
                            "is_family_friendly": analysis.get("is_family_friendly", False),
                            "confidence": analysis.get("confidence", "unknown"),
                            "reasoning": analysis.get("reasoning", ""),
                            "key_activities": analysis.get("key_activities", []),
                            "concerns": analysis.get("concerns", [])
                        })

                        # Add to family-friendly list if deemed suitable
                        if analysis.get("is_family_friendly", False):
                            logger.info(f"✓ {name} is family-friendly! Adding to results.")
                            # Enrich destination details with LLM analysis
                            details['family_friendly_analysis'] = {
                                "confidence": analysis.get("confidence", "unknown"),
                                "reasoning": analysis.get("reasoning", ""),
                                "key_activities": analysis.get("key_activities", []),
                                "concerns": analysis.get("concerns", [])
                            }
                            family_friendly.append(details)
                        else:
                            logger.info(f"✗ {name} is NOT family-friendly. Excluding from results.")
                            logger.info(f"Reasoning: {analysis.get('reasoning', 'N/A')}")

                        dest_duration = time.time() - dest_start_time
                        logger.info(f"Completed {name} in {dest_duration:.2f} seconds")
                        logger.info(f"Progress: {idx}/{len(destination_names)} destinations processed, "
                                   f"{len(family_friendly)} family-friendly so far")

                        analysis_success = True
                        break  # Success! Exit retry loop

                    except json.JSONDecodeError as je:
                        logger.error(f"JSON parsing error for {name}: {str(je)}")
                        logger.error(f"Raw response: {response_text[:500] if 'response_text' in locals() else 'No response'}")
                        analysis_log.append({
                            "destination": name,
                            "status": "error",
                            "reason": f"Failed to parse LLM response: {str(je)}",
                            "raw_response": response_text[:200] if 'response_text' in locals() else "No response"
                        })
                        break  # Don't retry JSON errors

                    except Exception as e:
                        error_str = str(e)
                        logger.error(f"Error processing {name}: {error_str}")

                        # Check if this is a quota error (429)
                        is_quota_error = '429' in error_str or 'quota' in error_str.lower() or 'rate limit' in error_str.lower()

                        if is_quota_error and attempt < max_retries:
                            # Retry with exponential backoff
                            delay = retry_delays[attempt]
                            logger.warning(f"API quota error for destination {name} (attempt {attempt + 1}/{max_retries + 1}): {error_str}")
                            logger.info(f"Retrying in {delay} seconds...")
                            time.sleep(delay)
                            continue
                        else:
                            # Either not a quota error, or we've exhausted retries
                            if is_quota_error:
                                logger.error(f"API quota exceeded for destination {name} after {max_retries + 1} attempts")
                                analysis_log.append({
                                    "destination": name,
                                    "status": "error",
                                    "reason": f"API quota exceeded after {max_retries + 1} attempts: {error_str}"
                                })
                            else:
                                logger.error(f"LLM analysis failed for destination {name}: {error_str}")
                                analysis_log.append({
                                    "destination": name,
                                    "status": "error",
                                    "reason": f"LLM analysis failed: {error_str}"
                                })
                            break

            total_duration = time.time() - start_time
            logger.info(f"\n{'=' * 80}")
            logger.info(f"FAMILY-FRIENDLY FILTER COMPLETE")
            logger.info(f"Total time: {total_duration:.2f} seconds")
            logger.info(f"Destinations processed: {len(destination_names)}")
            logger.info(f"Family-friendly destinations found: {len(family_friendly)}")
            logger.info(f"Average time per destination: {total_duration / len(destination_names):.2f} seconds")
            logger.info(f"{'=' * 80}\n")

            return {
                "success": True,
                "count": len(family_friendly),
                "destinations": family_friendly,
                "analysis_log": analysis_log,
                "method": "llm_with_web_search"
            }
        except Exception as e:
            logger.error(f"FATAL ERROR in filter_by_family_friendly: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "destinations": [],
                "analysis_log": []
            }

    def filter_by_city(self, city: str, destinations: List[Dict[str, Any]], cancellation_checker: Optional[callable] = None) -> Dict[str, Any]:
        """
        Filter a list of destinations by city using LLM with web search to find adjacent cities

        Workflow:
        1. Use LLM with web search to find at most 4 cities adjacent to the input city
        2. Add the input city to the list (city_names will have at most 5 cities: target + 4 bordering)
        3. Filter the provided destinations list to only include those in the city or adjacent cities
        4. Return filtered destinations

        Args:
            city: Target city name (e.g., 'Delhi', 'Mumbai')
            destinations: List of destination objects to filter (from search_destinations_quantitative)
            cancellation_checker: Optional callable that returns True if processing should be cancelled

        Returns:
            Dictionary with filtered destinations and city_names list (max 5 cities)
        """
        try:
            # Check for cancellation at the start
            if cancellation_checker and cancellation_checker():
                logger = logging.getLogger(__name__)
                logger.info("🛑 Processing cancelled before city filter started")
                return {
                    'success': False,
                    'cancelled': True,
                    'error': 'Processing was cancelled',
                    'destinations': [],
                    'city_names': [],
                    'target_city': city
                }

            # Check if GOOGLE_API_KEY is set
            google_api_key = os.environ.get('GOOGLE_API_KEY')
            use_fallback = False
            fallback_reason = None

            if not google_api_key:
                use_fallback = True
                fallback_reason = "Google API key not configured"
                logger = logging.getLogger(__name__)
                logger.warning(f"⚠️  GOOGLE_API_KEY not set - using hardcoded neighboring cities for {city}")

            # Try to use fallback first if API key is missing
            if use_fallback:
                adjacent_cities = self._get_neighboring_cities_fallback(city)
                city_names = [city.strip()]
                city_names.extend([c.strip() for c in adjacent_cities if c.strip()])
                city_names = list(dict.fromkeys(city_names))

                logger = logging.getLogger(__name__)
                logger.info(f"Using fallback neighboring cities for {city}: {adjacent_cities}")
            else:
                # Configure Gemini client with API key
                client = genai.Client(api_key=google_api_key)

                # Create prompt for LLM to find adjacent cities
                prompt = f"""You are identifying cities that are geographically adjacent to or part of the metropolitan area of a target city in India.

Target City: {city}, India

IMPORTANT: Search the web and identify AT MOST 4 cities that are:
1. Part of the {city} metropolitan area or region (e.g., for Mumbai: Navi Mumbai, Thane, Kalyan, Mira-Bhayandar)
2. Neighboring cities that share a border with {city}
3. Satellite cities or suburbs of {city}
4. Cities within 50km that are commonly considered part of the greater {city} area

Prioritize the most significant/populous bordering cities. Return a MAXIMUM of 4 cities.

SEARCH STRATEGY:
- Search for "{city} metropolitan area cities"
- Search for "{city} neighboring cities"
- Search for "cities near {city}"
- Search for "{city} suburbs satellite cities"

PRIORITIZE information from:
- Wikipedia articles about the {city} metropolitan area or region
- Official state tourism websites
- Geographic databases and maps
- Government urban planning documents

EXAMPLES of what we're looking for:
- For Mumbai: Navi Mumbai, Thane, Kalyan, Panvel, Mira-Bhayandar, Vasai-Virar, Bhiwandi
- For Delhi: Gurgaon, Noida, Ghaziabad, Faridabad, Greater Noida
- For Bangalore: Mysore (nearby major city)

Based on the web search results, provide a comprehensive list of city names that are geographically adjacent to or part of the {city} metropolitan area.

Respond ONLY with a JSON object in this exact format:
{{
    "adjacent_cities": ["City1", "City2", "City3", ...],
    "reasoning": "Brief explanation of why these cities are considered adjacent based on web search",
    "source_info": "Brief note about information sources used"
}}

CRITICAL: Return AT MOST 4 adjacent cities. Prioritize the most important/populous ones if there are more than 4 options.

Do not include any text before or after the JSON object."""

                # Retry logic with exponential backoff for API quota errors
                max_retries = 3
                retry_delays = [2, 4, 8]  # seconds
                response = None
                response_text = None

                for attempt in range(max_retries + 1):
                    try:
                        # Generate response with grounding (web search)
                        config = types.GenerateContentConfig(
                            temperature=0.0,  # Deterministic responses
                            tools=TravelTools._get_google_search_tool()  # Enable web search with Google Search tool
                        )
                        response = client.models.generate_content(
                            model='gemini-2.5-flash-lite',
                            contents=prompt,
                            config=config
                        )

                        # Parse the LLM response
                        response_text = response.text.strip()

                        # Extract JSON from response (handle markdown code blocks if present)
                        if '```json' in response_text:
                            response_text = response_text.split('```json')[1].split('```')[0].strip()
                        elif '```' in response_text:
                            response_text = response_text.split('```')[1].split('```')[0].strip()

                        analysis = json.loads(response_text)

                        # Get adjacent cities from LLM response
                        adjacent_cities = analysis.get("adjacent_cities", [])

                        # Limit to at most 4 bordering cities
                        adjacent_cities = adjacent_cities[:4]

                        # Log the LLM analysis for debugging
                        logger = logging.getLogger(__name__)
                        logger.info(f"LLM found adjacent cities for {city}: {adjacent_cities}")
                        logger.info(f"LLM reasoning: {analysis.get('reasoning', 'N/A')}")

                        # Always include the input city
                        city_names = [city.strip()]
                        city_names.extend([c.strip() for c in adjacent_cities if c.strip()])

                        # Remove duplicates while preserving order
                        city_names = list(dict.fromkeys(city_names))

                        logger.info(f"Final city_names list: {city_names}")
                        break  # Success! Exit retry loop

                    except json.JSONDecodeError as je:
                        # If LLM fails to return valid JSON, use fallback
                        logger = logging.getLogger(__name__)
                        logger.error(f"Failed to parse LLM response for city {city}: {str(je)}")
                        logger.error(f"Raw response: {response_text if response_text else 'N/A'}")
                        logger.warning(f"Falling back to hardcoded neighboring cities for {city}")

                        # Try fallback dictionary
                        use_fallback = True
                        fallback_reason = "LLM returned invalid JSON"
                        adjacent_cities = self._get_neighboring_cities_fallback(city)
                        city_names = [city.strip()]
                        city_names.extend([c.strip() for c in adjacent_cities if c.strip()])
                        city_names = list(dict.fromkeys(city_names))
                        break  # Don't retry JSON errors

                    except Exception as e:
                        error_str = str(e)
                        logger = logging.getLogger(__name__)

                        # Check if this is a quota error (429)
                        is_quota_error = '429' in error_str or 'quota' in error_str.lower() or 'rate limit' in error_str.lower()

                        if is_quota_error and attempt < max_retries:
                            # Retry with exponential backoff
                            delay = retry_delays[attempt]
                            logger.warning(f"API quota error for city {city} (attempt {attempt + 1}/{max_retries + 1}): {error_str}")
                            logger.info(f"Retrying in {delay} seconds...")
                            time.sleep(delay)
                            continue
                        else:
                            # Either not a quota error, or we've exhausted retries
                            # Use fallback dictionary instead of just input city
                            if is_quota_error:
                                logger.error(f"API quota exceeded for city {city} after {max_retries + 1} attempts: {error_str}")
                            else:
                                logger.error(f"LLM analysis failed for city {city}: {error_str}")

                            logger.warning(f"Falling back to hardcoded neighboring cities for {city}")
                            use_fallback = True
                            fallback_reason = "API quota exceeded" if is_quota_error else "LLM API failed"
                            adjacent_cities = self._get_neighboring_cities_fallback(city)
                            city_names = [city.strip()]
                            city_names.extend([c.strip() for c in adjacent_cities if c.strip()])
                            city_names = list(dict.fromkeys(city_names))
                            break

            # Filter the provided destinations list by city names
            # Normalize city names for comparison
            normalized_city_names = [name.strip().lower() for name in city_names]

            filtered_destinations = []
            for dest in destinations:
                # Get the city value from the destination
                dest_city = dest.get('city', '')
                if isinstance(dest_city, str) and dest_city.strip().lower() in normalized_city_names:
                    filtered_destinations.append(dest)

            # Build return dictionary
            result = {
                "success": True,
                "count": len(filtered_destinations),
                "destinations": filtered_destinations,
                "city_names": city_names,
                "target_city": city,
                "filters_applied": {
                    "city_filter": city_names
                },
                "note": f"Found {len(filtered_destinations)} destinations in {', '.join(city_names)}"
            }

            # Add warning if fallback was used
            if use_fallback:
                warning_msg = f"⚠️  Note: Using hardcoded neighboring cities for {city}"
                if fallback_reason:
                    warning_msg += f" (Reason: {fallback_reason})"
                if len(city_names) == 1:
                    warning_msg += f". No neighboring cities found in fallback dictionary."
                result["warning"] = warning_msg
                logger = logging.getLogger(__name__)
                logger.info(f"Returning with warning: {warning_msg}")

            return result
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "destinations": [],
                "city_names": [],
                "target_city": city
            }
            
def format_tool_result(result: Dict[str, Any]) -> str:
    """
    Format tool result for display to the agent

    Args:
        result: Tool execution result

    Returns:
        Formatted string
    """
    if not result.get("success"):
        return f"Error: {result.get('error', 'Unknown error')}"

    # Format based on result type
    if "destinations" in result:
        count = result.get("count", 0)
        if count == 0:
            return "No destinations found matching the criteria."

        dest_list = result["destinations"][:5]  # Show top 5
        formatted = f"Found {count} destinations. Top results:\n\n"

        for i, dest in enumerate(dest_list, 1):
            name = dest.get('name', dest.get('place', 'Unknown'))
            formatted += f"{i}. {name}\n"

            # Show key details
            for key, value in dest.items():
                if key not in ['name', 'place'] and value:
                    formatted += f"   {key}: {value}\n"
            formatted += "\n"

        return formatted

    elif "destination" in result:
        dest = result["destination"]
        formatted = f"Details for: {dest.get('name', 'Unknown')}\n\n"

        for key, value in dest.items():
            if value:
                formatted += f"{key}: {value}\n"

        return formatted

    elif "summary" in result:
        return json.dumps(result["summary"], indent=2)

    return json.dumps(result, indent=2)
