"""
Tools for Yatra Travel Agent using Google ADK
Function declarations and implementations for the agent
"""

from typing import List, Dict, Any, Optional
from .dataset_handler import DatasetHandler
import json
import google.generativeai as genai
from google.generativeai import protos
import os
import pandas as pd
import logging
import time


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
        return [protos.Tool(google_search=protos.GoogleSearch())]

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
                "description": "Filter a list of destinations by city using LLM with web search to find adjacent cities. First uses LLM with web search to find cities adjacent to the input city, then filters the provided destinations list to return only those in the input city or adjacent cities. This tool should be used AFTER search_destinations_quantitative to filter the results by city.",
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

    def filter_by_family_friendly(self, destination_names: List[str]) -> Dict[str, Any]:
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

        Returns:
            Dictionary with family-friendly destinations and analysis
        """
        try:
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
                return {
                    'success': True,
                    'count': 0,
                    'destinations': [],
                    'analysis_log': [],
                    'method': 'llm_with_web_search'
                }

            # Configure Gemini with grounding for web search
            model = genai.GenerativeModel(
                model_name='gemini-2.5-flash-lite',
                generation_config=genai.GenerationConfig(
                    temperature=0.0  # Deterministic responses
                )
            )

            family_friendly = []
            analysis_log = []

            for name in destination_names:
                details = self.dataset.get_destination_details(name)

                if not details:
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
    "reasoning": "Brief explanation of why this destination is included/excluded",
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
                        # Generate response with grounding (web search)
                        response = model.generate_content(
                            prompt,
                            tools=TravelTools._get_google_search_tool()  # Enable web search with Google Search tool
                        )

                        # Parse the LLM response
                        response_text = response.text.strip()

                        # Extract JSON from response (handle markdown code blocks if present)
                        if '```json' in response_text:
                            response_text = response_text.split('```json')[1].split('```')[0].strip()
                        elif '```' in response_text:
                            response_text = response_text.split('```')[1].split('```')[0].strip()

                        analysis = json.loads(response_text)

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
                            # Enrich destination details with LLM analysis
                            details['family_friendly_analysis'] = {
                                "confidence": analysis.get("confidence", "unknown"),
                                "reasoning": analysis.get("reasoning", ""),
                                "key_activities": analysis.get("key_activities", []),
                                "concerns": analysis.get("concerns", [])
                            }
                            family_friendly.append(details)

                        analysis_success = True
                        break  # Success! Exit retry loop

                    except json.JSONDecodeError as je:
                        analysis_log.append({
                            "destination": name,
                            "status": "error",
                            "reason": f"Failed to parse LLM response: {str(je)}",
                            "raw_response": response_text[:200] if 'response_text' in locals() else "No response"
                        })
                        break  # Don't retry JSON errors

                    except Exception as e:
                        error_str = str(e)

                        # Check if this is a quota error (429)
                        is_quota_error = '429' in error_str or 'quota' in error_str.lower() or 'rate limit' in error_str.lower()

                        if is_quota_error and attempt < max_retries:
                            # Retry with exponential backoff
                            delay = retry_delays[attempt]
                            logger = logging.getLogger(__name__)
                            logger.warning(f"API quota error for destination {name} (attempt {attempt + 1}/{max_retries + 1}): {error_str}")
                            logger.info(f"Retrying in {delay} seconds...")
                            time.sleep(delay)
                            continue
                        else:
                            # Either not a quota error, or we've exhausted retries
                            logger = logging.getLogger(__name__)
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

            return {
                "success": True,
                "count": len(family_friendly),
                "destinations": family_friendly,
                "analysis_log": analysis_log,
                "method": "llm_with_web_search"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "destinations": [],
                "analysis_log": []
            }

    def filter_by_city(self, city: str, destinations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Filter a list of destinations by city using LLM with web search to find adjacent cities

        Workflow:
        1. Use LLM with web search to find cities adjacent to the input city
        2. Add the input city to the list
        3. Filter the provided destinations list to only include those in the city or adjacent cities
        4. Return filtered destinations

        Args:
            city: Target city name (e.g., 'Delhi', 'Mumbai')
            destinations: List of destination objects to filter (from search_destinations_quantitative)

        Returns:
            Dictionary with filtered destinations
        """
        try:
            # Configure Gemini with grounding for web search
            model = genai.GenerativeModel(
                model_name='gemini-2.5-flash-lite',
                generation_config=genai.GenerationConfig(
                    temperature=0.0  # Deterministic responses
                )
            )

            # Create prompt for LLM to find adjacent cities
            prompt = f"""You are identifying cities that are geographically adjacent to or part of the metropolitan area of a target city in India.

Target City: {city}, India

IMPORTANT: Search the web and identify ALL cities that are:
1. Part of the {city} metropolitan area or region (e.g., for Mumbai: Navi Mumbai, Thane, Kalyan, Mira-Bhayandar, Vasai-Virar)
2. Neighboring cities that share a border with {city}
3. Satellite cities or suburbs of {city}
4. Cities within 50km that are commonly considered part of the greater {city} area

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

CRITICAL: Return AT LEAST 3-5 adjacent cities if they exist. Do not return an empty list.

Do not include any text before or after the JSON object."""

            # Retry logic with exponential backoff for API quota errors
            max_retries = 3
            retry_delays = [2, 4, 8]  # seconds
            response = None
            response_text = None

            for attempt in range(max_retries + 1):
                try:
                    # Generate response with grounding (web search)
                    response = model.generate_content(
                        prompt,
                        tools=TravelTools._get_google_search_tool()  # Enable web search with Google Search tool
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
                    # If LLM fails to return valid JSON, just use the input city
                    logger = logging.getLogger(__name__)
                    logger.error(f"Failed to parse LLM response for city {city}: {str(je)}")
                    logger.error(f"Raw response: {response_text if response_text else 'N/A'}")
                    city_names = [city.strip()]
                    analysis = {
                        "error": f"Failed to parse LLM response: {str(je)}",
                        "fallback": "Using only input city"
                    }
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
                        if is_quota_error:
                            logger.error(f"API quota exceeded for city {city} after {max_retries + 1} attempts: {error_str}")
                            logger.info(f"Falling back to using only input city: {city}")
                        else:
                            logger.error(f"LLM analysis failed for city {city}: {error_str}")

                        city_names = [city.strip()]
                        analysis = {
                            "error": f"LLM analysis failed: {error_str}",
                            "fallback": "Using only input city",
                            "quota_exceeded": is_quota_error
                        }
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

            return {
                "success": True,
                "count": len(filtered_destinations),
                "destinations": filtered_destinations,
                "city_names": city_names,
                "target_city": city,
#               "analysis": analysis,
                "filters_applied": {
                    "city_filter": city_names
                },
                "note": f"Found {len(filtered_destinations)} destinations in {', '.join(city_names)}"
            }
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
