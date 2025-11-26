"""
Tools for Yatra Travel Agent using Google ADK
Function declarations and implementations for the agent
"""

from typing import List, Dict, Any, Optional
from .dataset_handler import DatasetHandler
import json


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

            return {
                "success": True,
                "count": len(destinations),
                "destinations": destinations[:25],  # Limit to top 25 for context
                "filters_applied": {
                    "system_filters": "Rating >4.0 or null, Family-friendly",
                    "type": destination_type or "all",
                    "max_budget": max_budget or "unlimited",
                    "max_hours": max_hours or "unlimited"
                }
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
        Filter destinations for family-friendliness based on qualitative criteria

        Note: This is a simplified version. In a full implementation, this would
        use the LLM to analyze each destination's description and features.

        Args:
            destination_names: List of destination names to evaluate

        Returns:
            Dictionary with family-friendly destinations
        """
        try:
            family_friendly = []

            for name in destination_names:
                details = self.dataset.get_destination_details(name)

                if details:
                    # Simple heuristic - in practice, use LLM for qualitative analysis
                    # Look for family-friendly keywords in description
                    description = str(details.get('description', '')).lower()
                    keywords = ['family', 'child', 'kid', 'safe', 'park',
                               'museum', 'educational', 'garden']

                    if any(keyword in description for keyword in keywords):
                        family_friendly.append(details)

            return {
                "success": True,
                "count": len(family_friendly),
                "destinations": family_friendly
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "destinations": []
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
