"""
Tools for Yatra Travel Agent using Google ADK
Function declarations and implementations for the agent
"""

from typing import List, Dict, Any, Optional
from .dataset_handler import DatasetHandler
import json
import google.generativeai as genai
import os


class TravelTools:
    """Collection of tools for the travel agent to use with Google ADK"""

    def __init__(self, dataset_handler: DatasetHandler):
        """
        Initialize tools with dataset handler

        Args:
            dataset_handler: Instance of DatasetHandler for data operations
        """
        self.dataset = dataset_handler

    def get_tool_declarations(self) -> List[Dict[str, Any]]:
        """
        Get Google ADK tool declarations for all available tools

        Returns:
            List of tool declarations in Google ADK format
        """
        return [
            {
                "name": "search_destinations_quantitative",
                "description": "Search for destinations using quantitative filters like budget, duration, and type. SYSTEM FILTERS: Automatically filters for family-friendly destinations with rating null or >4.0. Returns a list of matching destinations.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "destination_type": {
                            "type": "string",
                            "description": "Type of destination: 'beach', 'mountain', 'heritage', 'wildlife'. Optional - omit this parameter to search all types",
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
                    "required": []
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
                "description": "Filter destinations to show only those suitable for families with small children. Considers safety, amenities, and child-friendly activities.",
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
                "description": "Returns a list of city names that are in or near a specified target city. Uses web search to determine proximity and geographic relationships. Takes a list of destination names, checks which cities those destinations are in, and returns only the city names (not destination objects) that are in or near the target city. Use this to get applicable city names for final filtering.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "destination_names": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of destination names to check (their cities will be evaluated for proximity)"
                        },
                        "city": {
                            "type": "string",
                            "description": "Target city name to filter by (e.g., 'Delhi', 'Mumbai', 'Bangalore')"
                        }
                    },
                    "required": ["destination_names", "city"]
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
                                        destination_type: Optional[str] = None,
                                        max_budget: Optional[float] = None,
                                        max_hours: Optional[float] = None) -> Dict[str, Any]:
        """
        Search destinations using quantitative filters with system-level filtering

        SYSTEM FILTERS (always applied):
        - Rating must be null or > 4.0
        - Destinations must be family-friendly

        Args:
            destination_type: Type of destination to filter by
            max_budget: Maximum budget in rupees
            max_hours: Maximum duration in hours (can be decimal, e.g., 0.5, 1.5, 8.25)

        Returns:
            Dictionary with search results
        """
        try:
            # Perform quantitative search with system-level filters
            results_df = self.dataset.search_quantitative(
                destination_type=destination_type if destination_type else None,
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

            # Limit to top 12 destinations and use compact format to manage payload size
            # Reduced from 25 to 12 to prevent Gemini API finish_reason: 12 (UNEXPECTED_TOOL_CALL) errors
            # Only include essential fields to keep the response compact
            compact_destinations = []
            for dest in destinations[:12]:
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

            return {
                "success": True,
                "count": len(destinations),
                "total_matching": len(destinations),
                "destinations": compact_destinations,
                "filters_applied": {
                    "system_filters": "Rating >4.0 or null, Family-friendly",
                    "type": destination_type or "all",
                    "max_budget": max_budget or "unlimited",
                    "max_hours": max_hours or "unlimited"
                },
                "note": f"Showing top {len(compact_destinations)} of {len(destinations)} matching destinations"
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
                model_name='gemini-2.0-flash-exp',
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
                prompt = f"""You are analyzing tourist destinations in India for family-friendliness.

Destination to analyze: {location_context}

Please search the web for information about this destination, focusing on:
1. Activities available at this destination
2. Safety for children
3. Amenities for families (restrooms, food options, accessibility)
4. Educational or entertainment value for children

PRIORITIZE information from:
- Official state tourism board websites (e.g., incredibleindia.org, state tourism sites)
- Wikipedia
- Official destination websites
- Reputable travel sites

Based on the web search results, determine if this destination is suitable for families with small children (ages 3-12).

Respond ONLY with a JSON object in this exact format:
{{
    "is_family_friendly": true/false,
    "confidence": "high/medium/low",
    "reasoning": "Brief explanation based on web search findings",
    "key_activities": ["activity1", "activity2", "activity3"],
    "concerns": ["concern1", "concern2"] or []
}}

Do not include any text before or after the JSON object."""

                try:
                    # Generate response with grounding (web search)
                    response = model.generate_content(
                        prompt,
                        tools=[{'google_search_retrieval': {}}]  # Enable web search grounding
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

                except json.JSONDecodeError as je:
                    analysis_log.append({
                        "destination": name,
                        "status": "error",
                        "reason": f"Failed to parse LLM response: {str(je)}",
                        "raw_response": response_text[:200] if 'response_text' in locals() else "No response"
                    })
                except Exception as e:
                    analysis_log.append({
                        "destination": name,
                        "status": "error",
                        "reason": f"LLM analysis failed: {str(e)}"
                    })

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

    def filter_by_city(self, destination_names: List[str], city: str) -> Dict[str, Any]:
        """
        Filter destinations by city proximity and return applicable city names using LLM with web search

        This implementation uses an LLM to search the web for information about
        each destination's city and determine if it's located in or near the specified city.
        Returns only the city names (not full destination objects) that are in or near
        the target city, which can then be used to filter the destination list.

        Args:
            destination_names: List of destination names to evaluate (can also accept list of dicts with 'name' key)
            city: City name to filter by (e.g., 'Delhi', 'Mumbai')

        Returns:
            Dictionary with city names that are in or near the target city and analysis
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
                            'city_names': [],
                            'analysis_log': [],
                            'target_city': city
                        }
                else:
                    return {
                        'success': False,
                        'error': f'Invalid destination_names parameter. Expected list of strings or dicts, got: {type(item)}',
                        'city_names': [],
                        'analysis_log': [],
                        'target_city': city
                    }

            destination_names = normalized_names

            if not destination_names:
                return {
                    'success': True,
                    'count': 0,
                    'city_names': [],
                    'analysis_log': [],
                    'target_city': city,
                    'method': 'llm_with_web_search'
                }

            # Configure Gemini with grounding for web search
            model = genai.GenerativeModel(
                model_name='gemini-2.0-flash-exp',
                generation_config=genai.GenerationConfig(
                    temperature=0.0  # Deterministic responses
                )
            )

            nearby_city_names = set()  # Use set to avoid duplicates
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

                # Get location context from dataset
                dest_city = details.get('city', '')
                dest_state = details.get('state', '')
                location_context = f"{name}, {dest_city}, {dest_state}, India" if dest_city and dest_state else f"{name}, India"

                # Create prompt for LLM to analyze city proximity
                prompt = f"""You are analyzing tourist destinations in India for geographic proximity.

Target City: {city}
Destination to analyze: {location_context}

Please search the web for information about this destination's location, focusing on:
1. Which city/cities the destination is located in or near
2. Distance from {city} (if the destination is not in {city})
3. Geographic/administrative relationship to {city} (same metro area, neighboring district, etc.)
4. Common travel accessibility from {city}

PRIORITIZE information from:
- Official state tourism board websites (e.g., incredibleindia.org, state tourism sites)
- Wikipedia
- Official destination websites
- Google Maps or similar mapping services

Based on the web search results, determine if this destination is in or reasonably near {city}.
Consider "near" to mean:
- Within the same city/metro area
- Within approximately 100km radius
- Commonly visited as a day trip or short excursion from {city}
- In neighboring districts or administrative areas that are closely connected

Respond ONLY with a JSON object in this exact format:
{{
    "is_near_city": true/false,
    "confidence": "high/medium/low",
    "reasoning": "Brief explanation based on web search findings",
    "actual_location": "City/district where the destination is actually located",
    "distance_info": "Distance from target city if available, or 'N/A'"
}}

Do not include any text before or after the JSON object."""

                try:
                    # Generate response with grounding (web search)
                    response = model.generate_content(
                        prompt,
                        tools=[{'google_search_retrieval': {}}]  # Enable web search grounding
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
                        "is_near_city": analysis.get("is_near_city", False),
                        "confidence": analysis.get("confidence", "unknown"),
                        "reasoning": analysis.get("reasoning", ""),
                        "actual_location": analysis.get("actual_location", "Unknown"),
                        "distance_info": analysis.get("distance_info", "N/A")
                    })

                    # Add city name to nearby list if deemed close to the target city
                    if analysis.get("is_near_city", False):
                        # Extract the city name from destination details
                        dest_city = details.get('city', '')
                        if dest_city and dest_city.strip():
                            nearby_city_names.add(dest_city.strip())

                except json.JSONDecodeError as je:
                    analysis_log.append({
                        "destination": name,
                        "status": "error",
                        "reason": f"Failed to parse LLM response: {str(je)}",
                        "raw_response": response_text[:200] if 'response_text' in locals() else "No response"
                    })
                except Exception as e:
                    analysis_log.append({
                        "destination": name,
                        "status": "error",
                        "reason": f"LLM analysis failed: {str(e)}"
                    })

            # Convert set to sorted list for consistent output
            city_names_list = sorted(list(nearby_city_names))

            return {
                "success": True,
                "count": len(city_names_list),
                "city_names": city_names_list,
                "analysis_log": analysis_log,
                "target_city": city,
                "method": "llm_with_web_search"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "city_names": [],
                "analysis_log": [],
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
