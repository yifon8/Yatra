"""
Yatra Destination Suggester Agent
Main agent logic using Google ADK (Gemini) for travel recommendations
"""

import google.generativeai as genai
from typing import Dict, Any, List, Optional
import os
import json
import logging
import time
import pandas as pd

from .dataset_handler import DatasetHandler
from .tools import TravelTools, format_tool_result
from .system_prompts import (
    AGENT_SYSTEM_PROMPT,
    get_recommendation_prompt,
    get_destination_analysis_prompt
)

# Configure logging
logger = logging.getLogger(__name__)


class DestinationSuggester:
    """Main agent class for suggesting travel destinations using Google ADK"""

    def __init__(self, api_key: str = None, csv_path: str = None):
        """
        Initialize the destination suggester agent

        Args:
            api_key: Google API key (or set GOOGLE_API_KEY environment variable)
            csv_path: Path to the CSV dataset
        """
        # Set up API key
        if api_key is None:
            api_key = os.getenv('GOOGLE_API_KEY')

        if not api_key:
            raise ValueError(
                "Google API key required. Pass as parameter or set GOOGLE_API_KEY environment variable"
            )

        genai.configure(api_key=api_key)

        # Initialize dataset and tools
        self.dataset = DatasetHandler(csv_path)
        self.tools = TravelTools(self.dataset)

        # Initialize the model with function calling
        # Set temperature=0 for deterministic, reproducible responses
        self.model = genai.GenerativeModel(
            model_name='gemini-2.5-flash-lite',
            system_instruction=AGENT_SYSTEM_PROMPT,
            generation_config=genai.GenerationConfig(
                temperature=0.0
            )
        )

        # Convert tool declarations to Google ADK format
        self.tool_config = self._create_tool_config()

        print("✓ Yatra agent initialized successfully!")

    def _create_tool_config(self) -> List:
        """Create Google ADK tool configuration from tool declarations"""
        tool_declarations = self.tools.get_tool_declarations()

        # Convert to Google ADK format
        tools = []
        for tool_decl in tool_declarations:
            # Convert parameters dict to Schema object (required on Windows)
            params_dict = tool_decl["parameters"]
            params_schema = genai.protos.Schema(
                type=genai.protos.Type.OBJECT,
                properties={
                    key: genai.protos.Schema(**self._convert_property_to_schema(value))
                    for key, value in params_dict.get("properties", {}).items()
                },
                required=params_dict.get("required", [])
            )

            tools.append(genai.protos.Tool(
                function_declarations=[
                    genai.protos.FunctionDeclaration(
                        name=tool_decl["name"],
                        description=tool_decl["description"],
                        parameters=params_schema
                    )
                ]
            ))

        return tools

    def _convert_property_to_schema(self, property_def: dict) -> dict:
        """Convert a property definition to Schema format"""
        schema_dict = {}

        # Map JSON schema types to protobuf types
        type_mapping = {
            "string": genai.protos.Type.STRING,
            "number": genai.protos.Type.NUMBER,
            "integer": genai.protos.Type.INTEGER,
            "boolean": genai.protos.Type.BOOLEAN,
            "array": genai.protos.Type.ARRAY,
            "object": genai.protos.Type.OBJECT
        }

        prop_type = property_def.get("type", "string")
        schema_dict["type"] = type_mapping.get(prop_type, genai.protos.Type.STRING)

        if "description" in property_def:
            schema_dict["description"] = property_def["description"]

        if "enum" in property_def:
            schema_dict["enum"] = property_def["enum"]

        # Handle array items
        if prop_type == "array" and "items" in property_def:
            items_def = property_def["items"]
            items_type = items_def.get("type", "string")
            schema_dict["items"] = genai.protos.Schema(
                type=type_mapping.get(items_type, genai.protos.Type.STRING)
            )

        return schema_dict

    def suggest_destinations(self,
                           destination_type: str,
                           city: Optional[str] = None,
                           hours: Optional[float] = None,
                           budget: Optional[float] = None,
                           max_iterations: int = 5) -> Dict[str, Any]:
        """
        Suggest destinations based on user preferences using a deterministic filtering pipeline

        The filtering pipeline follows these steps:
        1. Filter by destination type (REQUIRED user input)
        2. Filter by rating (>= 4 stars, system required)
        3. Filter by budget (if provided)
        4. Filter by visit duration (if provided)
        5. Store resulting destinations list

        Then TWO BRANCHES:

        Branch A (if city is provided):
        6. Generate city_names list (input city + adjacent cities) via LLM web search
        7. Filter destinations by city_names
        8. Call family-friendly filter via LLM web search
        9. Filter by family_friendly = true
        10. Generate recommendations

        Branch B (if city is NOT provided):
        6. Cap to top 25 destinations sorted by rating
        7. Call family-friendly filter via LLM web search
        8. Filter by family_friendly = true
        9. Generate recommendations

        Args:
            destination_type: Type of destination (beach, mountain, heritage, wildlife) - REQUIRED
            city: City name to find destinations in or near (optional, max 30 chars)
            hours: Available time in hours (can be decimal, e.g., 0.5, 1.5, 8.25)
            budget: Budget in rupees
            max_iterations: Maximum tool calling iterations (deprecated - kept for compatibility)

        Returns:
            Dictionary with recommendations and reasoning
        """
        print(f"\n🔍 Starting deterministic filtering pipeline...\n")

        filtering_steps = []
        initial_count = len(self.dataset.df)

        # STEP 1: Filter by destination type (REQUIRED)
        print(f"🏖️  Step 1: Filtering by destination type ({destination_type})")
        filtered_df = self.dataset._apply_type_filter_on_df(self.dataset.df, destination_type)
        after_type = len(filtered_df)
        filtering_steps.append(f"Type filter ({destination_type}): {initial_count} → {after_type} destinations")
        print(f"   ✓ Filtered from {initial_count} to {after_type} destinations\n")

        # STEP 2: Filter by rating >= 4 stars (SYSTEM REQUIRED)
        print("📊 Step 2: Filtering by rating (>= 4 stars)")
        filtered_df = self.dataset._apply_rating_filter_on_df(filtered_df, min_rating=4.0)
        after_rating = len(filtered_df)
        filtering_steps.append(f"Rating filter (>= 4.0): {after_type} → {after_rating} destinations")
        print(f"   ✓ Filtered to {after_rating} destinations\n")

        # STEP 3: Filter by budget (if provided)
        if budget is not None:
            print(f"💰 Step 3: Filtering by budget (<= ₹{budget:,.0f})")
            filtered_df = self.dataset._apply_budget_filter_on_df(filtered_df, budget)
            after_budget = len(filtered_df)
            filtering_steps.append(f"Budget filter (<= ₹{budget:,.0f}): {after_rating} → {after_budget} destinations")
            print(f"   ✓ Filtered to {after_budget} destinations\n")
            after_rating = after_budget
        else:
            print("💰 Step 3: No budget filter specified (skipped)\n")

        # STEP 4: Filter by visit duration (if provided)
        if hours is not None:
            print(f"⏱️  Step 4: Filtering by visit duration (<= {hours:g} hours)")
            filtered_df = self.dataset._apply_duration_filter_on_df(filtered_df, hours)
            after_duration = len(filtered_df)
            filtering_steps.append(f"Duration filter (<= {hours:g} hrs): {after_rating} → {after_duration} destinations")
            print(f"   ✓ Filtered to {after_duration} destinations\n")
            after_rating = after_duration
        else:
            print("⏱️  Step 4: No duration filter specified (skipped)\n")

        # STEP 5: Check if we have any destinations left
        if len(filtered_df) == 0:
            return {
                'success': False,
                'error': 'No destinations found matching your criteria. Try adjusting your filters.',
                'query': f"destination_type={destination_type}, hours={hours}, budget={budget}, city={city}",
                'tools_used': [],
                'iterations': 0,
                'filtering_steps': filtering_steps
            }

        # STEP 5: Store resulting destinations list
        destinations_list = filtered_df.to_dict('records')
        print(f"✅ Step 5: Stored {len(destinations_list)} destinations after initial filtering\n")
        filtering_steps.append(f"Initial filtering complete: {len(destinations_list)} destinations ready for next stage")

        tools_used = []

        # TWO-BRANCH LOGIC BASED ON CITY INPUT
        if city:
            # ========== BRANCH A: CITY PROVIDED ==========
            print(f"🌆 BRANCH A: City-based filtering (city: {city})\n")

            # STEP 6 (Branch A): Generate city_names list (input city + adjacent cities)
            print(f"🏙️  Step 6: Generating city_names list using LLM web search")
            print(f"   Finding cities adjacent to {city}...")

            try:
                # Use filter_by_city to get adjacent cities and filter destinations
                city_result = self.tools.filter_by_city(city, destinations_list)

                if not city_result.get('success', False):
                    logger.error(f"City filter failed: {city_result.get('error')}")
                    return {
                        'success': False,
                        'error': f"City filter failed: {city_result.get('error', 'Unknown error')}",
                        'query': f"destination_type={destination_type}, hours={hours}, budget={budget}, city={city}",
                        'tools_used': [{'tool': 'filter_by_city', 'result': city_result}],
                        'iterations': 1,
                        'filtering_steps': filtering_steps
                    }

                city_names = city_result.get('city_names', [city])
                print(f"   ✓ Found city_names: {', '.join(city_names)}\n")

                # STEP 7 (Branch A): Filter by city_names
                print(f"📍 Step 7: Filtering destinations by city_names")
                city_filtered_destinations = city_result.get('destinations', [])
                after_city = len(city_filtered_destinations)
                filtering_steps.append(f"City filter ({city} + adjacent): {len(destinations_list)} → {after_city} destinations")
                print(f"   ✓ Filtered to {after_city} destinations in {', '.join(city_names)}\n")

                tools_used.append({'tool': 'filter_by_city', 'params': {'city': city}, 'result': city_result})

                # Check if we have any destinations left after city filter
                if len(city_filtered_destinations) == 0:
                    return {
                        'success': False,
                        'error': f'No destinations found in or near {city} matching your criteria. Try a different city or adjust your filters.',
                        'query': f"destination_type={destination_type}, hours={hours}, budget={budget}, city={city}",
                        'tools_used': tools_used,
                        'iterations': len(tools_used),
                        'filtering_steps': filtering_steps
                    }

            except Exception as e:
                logger.error(f"City filter error: {e}")
                return {
                    'success': False,
                    'error': f"Error applying city filter: {str(e)}",
                    'query': f"destination_type={destination_type}, hours={hours}, budget={budget}, city={city}",
                    'tools_used': tools_used,
                    'iterations': len(tools_used),
                    'filtering_steps': filtering_steps
                }

            # STEP 8 (Branch A): Call family-friendly filter
            print(f"👨‍👩‍👧 Step 8: Filtering for family-friendly destinations using LLM web search")
            print(f"   Analyzing {len(city_filtered_destinations)} destinations...")

            try:
                family_result = self.tools.filter_by_family_friendly(city_filtered_destinations)

                if not family_result.get('success', False):
                    logger.error(f"Family filter failed: {family_result.get('error')}")
                    return {
                        'success': False,
                        'error': f"Family-friendly filter failed: {family_result.get('error', 'Unknown error')}",
                        'query': f"destination_type={destination_type}, hours={hours}, budget={budget}, city={city}",
                        'tools_used': tools_used + [{'tool': 'filter_by_family_friendly', 'result': family_result}],
                        'iterations': len(tools_used) + 1,
                        'filtering_steps': filtering_steps
                    }

                family_friendly_destinations = family_result.get('destinations', [])
                after_family = len(family_friendly_destinations)
                filtering_steps.append(f"Family-friendly filter (LLM): {after_city} → {after_family} destinations")
                print(f"   ✓ Filtered to {after_family} family-friendly destinations\n")

                tools_used.append({'tool': 'filter_by_family_friendly', 'params': {'destination_count': len(city_filtered_destinations)}, 'result': family_result})

            except Exception as e:
                logger.error(f"Family filter error: {e}")
                return {
                    'success': False,
                    'error': f"Error applying family-friendly filter: {str(e)}",
                    'query': f"destination_type={destination_type}, hours={hours}, budget={budget}, city={city}",
                    'tools_used': tools_used,
                    'iterations': len(tools_used),
                    'filtering_steps': filtering_steps
                }

            # STEP 9 (Branch A): Filter by family_friendly = true (already done by the tool)
            # Check if we have any destinations left
            if len(family_friendly_destinations) == 0:
                return {
                    'success': False,
                    'error': 'No family-friendly destinations found matching your criteria. Try adjusting your filters.',
                    'query': f"destination_type={destination_type}, hours={hours}, budget={budget}, city={city}",
                    'tools_used': tools_used,
                    'iterations': len(tools_used),
                    'filtering_steps': filtering_steps
                }

            # Store final working set for recommendations
            working_destinations = family_friendly_destinations

        else:
            # ========== BRANCH B: NO CITY PROVIDED ==========
            print(f"🌍 BRANCH B: No city filter (analyzing top destinations by rating)\n")

            # STEP 6 (Branch B): Cap to top 25 destinations sorted by rating
            print(f"📊 Step 6: Capping to top 25 destinations sorted by rating")

            # Sort by rating descending
            rating_column = None
            possible_columns = ['rating', 'google_review_rating', 'review_rating', 'user_rating', 'average_rating']
            for col in possible_columns:
                if col in filtered_df.columns:
                    rating_column = col
                    break

            if rating_column:
                # Convert to numeric and sort
                filtered_df[f'{rating_column}_numeric'] = pd.to_numeric(filtered_df[rating_column], errors='coerce')
                sorted_df = filtered_df.sort_values(by=f'{rating_column}_numeric', ascending=False, na_position='last')
                sorted_df = sorted_df.drop(columns=[f'{rating_column}_numeric'])
            else:
                sorted_df = filtered_df

            # Cap to top 25
            top_destinations = sorted_df.head(25).to_dict('records')
            after_cap = len(top_destinations)
            filtering_steps.append(f"Cap to top 25 by rating: {len(destinations_list)} → {after_cap} destinations")
            print(f"   ✓ Capped to {after_cap} destinations\n")

            # STEP 7 (Branch B): Call family-friendly filter
            print(f"👨‍👩‍👧 Step 7: Filtering for family-friendly destinations using LLM web search")
            print(f"   Analyzing {len(top_destinations)} destinations...")

            try:
                family_result = self.tools.filter_by_family_friendly(top_destinations)

                if not family_result.get('success', False):
                    logger.error(f"Family filter failed: {family_result.get('error')}")
                    return {
                        'success': False,
                        'error': f"Family-friendly filter failed: {family_result.get('error', 'Unknown error')}",
                        'query': f"destination_type={destination_type}, hours={hours}, budget={budget}, city={city}",
                        'tools_used': [{'tool': 'filter_by_family_friendly', 'result': family_result}],
                        'iterations': 1,
                        'filtering_steps': filtering_steps
                    }

                family_friendly_destinations = family_result.get('destinations', [])
                after_family = len(family_friendly_destinations)
                filtering_steps.append(f"Family-friendly filter (LLM): {after_cap} → {after_family} destinations")
                print(f"   ✓ Filtered to {after_family} family-friendly destinations\n")

                tools_used.append({'tool': 'filter_by_family_friendly', 'params': {'destination_count': len(top_destinations)}, 'result': family_result})

            except Exception as e:
                logger.error(f"Family filter error: {e}")
                return {
                    'success': False,
                    'error': f"Error applying family-friendly filter: {str(e)}",
                    'query': f"destination_type={destination_type}, hours={hours}, budget={budget}, city={city}",
                    'tools_used': [],
                    'iterations': 0,
                    'filtering_steps': filtering_steps
                }

            # STEP 8 (Branch B): Filter by family_friendly = true (already done by the tool)
            # Check if we have any destinations left
            if len(family_friendly_destinations) == 0:
                return {
                    'success': False,
                    'error': 'No family-friendly destinations found matching your criteria. Try adjusting your filters.',
                    'query': f"destination_type={destination_type}, hours={hours}, budget={budget}, city={city}",
                    'tools_used': tools_used,
                    'iterations': len(tools_used),
                    'filtering_steps': filtering_steps
                }

            # Store final working set for recommendations
            working_destinations = family_friendly_destinations

        # FINAL STEP: Generate recommendations from filtered results
        if city:
            print(f"✨ Step 10: Generating recommendations from {len(working_destinations)} filtered destinations\n")
        else:
            print(f"✨ Step 9: Generating recommendations from {len(working_destinations)} filtered destinations\n")

        # Limit to max 50 destinations to prevent payload issues
        max_results = 50
        if len(working_destinations) > max_results:
            print(f"   ⚠️  Limiting to {max_results} destinations to manage payload size")
            working_destinations = working_destinations[:max_results]

        # Create recommendation query
        recommendation_query = self._create_recommendation_query(
            destination_type=destination_type,
            city=city,
            hours=hours,
            budget=budget,
            filtered_destinations=working_destinations,
            filtering_steps=filtering_steps
        )

        # Start chat session for recommendations
        chat = self.model.start_chat()

        # Send recommendation query
        max_retries = 3
        retry_delay = 1
        response = None

        for attempt in range(max_retries):
            try:
                # Send without tools - just ask for recommendations
                response = chat.send_message(recommendation_query)

                # Check for finish_reason issues
                if response.candidates:
                    candidate = response.candidates[0]
                    finish_reason = candidate.finish_reason
                    finish_reason_value = finish_reason.value if hasattr(finish_reason, 'value') else finish_reason

                    if finish_reason_value and finish_reason_value != 1:
                        logger.warning(f"Recommendation response finish_reason: {finish_reason} (value: {finish_reason_value})")

                        if finish_reason_value == 3:
                            return {
                                'success': False,
                                'error': 'Content filtered by safety settings. Please try again.',
                                'query': recommendation_query,
                                'tools_used': tools_used,
                                'iterations': len(tools_used),
                                'filtering_steps': filtering_steps
                            }

                        if finish_reason_value == 12:
                            return {
                                'success': False,
                                'error': 'The AI model encountered an issue. Please try with fewer results or more specific filters.',
                                'query': recommendation_query,
                                'tools_used': tools_used,
                                'iterations': len(tools_used),
                                'filtering_steps': filtering_steps
                            }

                # Check for valid response
                if response.candidates and response.candidates[0].content.parts:
                    break
                else:
                    logger.warning(f"Empty response (attempt {attempt + 1}/{max_retries})")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        retry_delay *= 2

            except Exception as e:
                logger.error(f"Error getting recommendations (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    return {
                        'success': False,
                        'error': f'API error: {str(e)}',
                        'query': recommendation_query,
                        'tools_used': tools_used,
                        'iterations': len(tools_used),
                        'filtering_steps': filtering_steps
                    }

        # Check if we got a valid response
        if not response or not response.candidates or not response.candidates[0].content.parts:
            logger.error("Failed to get valid response after all retries")
            return {
                'success': False,
                'error': 'Unable to get recommendations from the AI service. Please try again.',
                'query': recommendation_query,
                'tools_used': tools_used,
                'iterations': len(tools_used),
                'filtering_steps': filtering_steps
            }

        # Extract recommendations
        try:
            final_response = response.text

            if not final_response or not final_response.strip():
                logger.error("Empty recommendation text")
                return {
                    'success': False,
                    'error': 'Unable to generate recommendations. Please try again.',
                    'query': recommendation_query,
                    'tools_used': tools_used,
                    'iterations': len(tools_used),
                    'filtering_steps': filtering_steps
                }

        except Exception as e:
            logger.error(f"Error extracting recommendations: {e}")
            return {
                'success': False,
                'error': 'Unable to process recommendations. Please try again.',
                'query': recommendation_query,
                'tools_used': tools_used,
                'iterations': len(tools_used),
                'filtering_steps': filtering_steps
            }

        print(f"\n✨ Recommendations ready!\n")

        return {
            'success': True,
            'recommendations': final_response,
            'query': recommendation_query,
            'tools_used': tools_used,
            'iterations': len(tools_used),
            'filtering_steps': filtering_steps
        }

    def _create_recommendation_query(self,
                                    destination_type: Optional[str],
                                    city: Optional[str],
                                    hours: Optional[float],
                                    budget: Optional[float],
                                    filtered_destinations: List[Dict[str, Any]],
                                    filtering_steps: List[str]) -> str:
        """
        Create a query for LLM to generate recommendations from pre-filtered destinations

        Args:
            destination_type: Type of destination
            city: City name (if specified)
            hours: Duration (if specified)
            budget: Budget (if specified)
            filtered_destinations: Pre-filtered list of destinations
            filtering_steps: List of filtering steps applied

        Returns:
            Query string for LLM
        """
        # Build context about user preferences
        preferences = []
        if destination_type:
            preferences.append(f"Type: {destination_type}")
        if city:
            preferences.append(f"Location: {city} and nearby cities")
        if hours is not None:
            preferences.append(f"Visit duration: up to {hours:g} hours")
        if budget is not None:
            if budget == 0:
                preferences.append("Budget: Free (no cost)")
            else:
                preferences.append(f"Budget: up to ₹{budget:,.0f}")

        preferences_text = ", ".join(preferences) if preferences else "General family-friendly destinations in India"

        # Create destination list for LLM
        dest_summaries = []
        for i, dest in enumerate(filtered_destinations[:50], 1):  # Limit to 50 for payload size
            name = dest.get('name') or dest.get('place') or dest.get('destination', 'Unknown')
            city_name = dest.get('city', '')
            dest_type = dest.get('type', '')
            rating = dest.get('google_review_rating') or dest.get('rating', '')
            duration = dest.get('time_needed_to_visit_in_hrs', '')
            fee = dest.get('entrance_fee_in_inr', '')

            summary = f"{i}. {name}"
            if city_name:
                summary += f" ({city_name})"
            details = []
            if dest_type:
                details.append(f"Type: {dest_type}")
            if rating:
                details.append(f"Rating: {rating}")
            if duration:
                details.append(f"Duration: {duration} hrs")
            if fee:
                details.append(f"Fee: ₹{fee}")
            if details:
                summary += f" - {', '.join(details)}"

            dest_summaries.append(summary)

        destinations_text = "\n".join(dest_summaries)

        # Build filtering steps summary
        filtering_summary = "\n".join([f"  - {step}" for step in filtering_steps])

        query = f"""I've pre-filtered family-friendly travel destinations in India based on these criteria:

**User Preferences:** {preferences_text}

**Filtering Pipeline Applied:**
{filtering_summary}

**Filtered Destinations ({len(filtered_destinations)} total, showing first {len(dest_summaries)}):**
{destinations_text}

**Your Task:**
Based on the filtered destinations above, please provide your top 3 recommendations that best match the user's preferences. For each recommendation:

1. Explain WHY you chose this destination
2. Highlight what makes it suitable for families with children
3. Mention any special features or attractions
4. Provide practical tips if relevant

Please write in a friendly, helpful tone as a travel advisor."""

        return query

    def _suggest_with_tools(self,
                           query: str,
                           max_iterations: int = 5) -> Dict[str, Any]:
        """
        Legacy method for tool-based suggestions (now replaced by deterministic pipeline)
        This method is kept for backward compatibility but should not be used.
        """
        logger.warning("_suggest_with_tools is deprecated - use suggest_destinations directly")
        return {
            'success': False,
            'error': 'This method is deprecated. Please use suggest_destinations directly.',
            'query': query,
            'tools_used': [],
            'iterations': 0
        }

    def _old_suggest_with_tools_backup(self,
                           query: str,
                           max_iterations: int = 5) -> Dict[str, Any]:
        """
        OLD BACKUP: Original tool-based implementation (DEPRECATED)
        Preserved for reference but should not be used.
        """
        # Start chat session with tools
        chat = self.model.start_chat()

        # Send initial query with tools - retry logic for empty responses
        max_retries = 3
        retry_delay = 1  # seconds
        response = None

        for attempt in range(max_retries):
            try:
                response = chat.send_message(
                    query,
                    tools=self.tool_config
                )

                # Check for finish_reason issues (safety, content filtering, etc.)
                if response.candidates:
                    candidate = response.candidates[0]
                    finish_reason = candidate.finish_reason

                    # Get the finish_reason value (1 = STOP/normal completion)
                    finish_reason_value = finish_reason.value if hasattr(finish_reason, 'value') else finish_reason

                    # Log finish_reason if it's not a normal completion (1 = STOP)
                    if finish_reason_value and finish_reason_value != 1:
                        logger.warning(f"Initial response finish_reason: {finish_reason} (value: {finish_reason_value})")

                        # Check for safety/content filtering (3 = SAFETY)
                        if finish_reason_value == 3:
                            return {
                                'success': False,
                                'error': 'Content filtered by safety settings. Please try rephrasing your query.',
                                'query': query,
                                'tools_used': [],
                                'iterations': 0
                            }

                        # Check for unexpected tool call (12 = UNEXPECTED_TOOL_CALL)
                        if finish_reason_value == 12:
                            return {
                                'success': False,
                                'error': 'The AI model encountered an issue with your request. Please try rephrasing your query or adjusting your search criteria.',
                                'query': query,
                                'tools_used': [],
                                'iterations': 0
                            }

                # Check if we got a valid response
                if response.candidates and response.candidates[0].content.parts:
                    break
                else:
                    logger.warning(f"Empty response from Gemini API (attempt {attempt + 1}/{max_retries})")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        retry_delay *= 2  # exponential backoff
            except Exception as e:
                logger.error(f"Error calling Gemini API (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    return {
                        'success': False,
                        'error': f'API error: {str(e)}',
                        'query': query,
                        'tools_used': [],
                        'iterations': 0
                    }

        # Check if we ever got a valid response after all retries
        if not response or not response.candidates or not response.candidates[0].content.parts:
            logger.error("Failed to get valid response from Gemini API after all retries")
            return {
                'success': False,
                'error': 'Unable to get a response from the AI service. Please try again.',
                'query': query,
                'tools_used': [],
                'iterations': 0
            }

        # Handle tool calling loop
        iteration = 0
        conversation_history = []

        while iteration < max_iterations:
            # Check if response has candidates and parts
            if not response.candidates or not response.candidates[0].content.parts:
                logger.warning(f"Empty response during tool calling loop at iteration {iteration}")
                # This might indicate the search found no results
                # Check if we used the search tool
                search_used = any(h['tool'] == 'search_destinations_quantitative' for h in conversation_history)
                if search_used:
                    return {
                        'success': False,
                        'error': 'No destinations found matching your criteria. Try adjusting your filters.',
                        'query': query,
                        'tools_used': conversation_history,
                        'iterations': iteration
                    }
                else:
                    return {
                        'success': False,
                        'error': 'Unable to complete your request. Please try again.',
                        'query': query,
                        'tools_used': conversation_history,
                        'iterations': iteration
                    }

            # Check if model wants to use tools (check any part, not just parts[0])
            has_function_call = any(
                hasattr(part, 'function_call') and part.function_call
                for part in response.candidates[0].content.parts
            )

            if has_function_call:
                # Extract function call - find the part that has it
                function_call = None
                for part in response.candidates[0].content.parts:
                    if hasattr(part, 'function_call') and part.function_call:
                        function_call = part.function_call
                        break

                if not function_call:
                    logger.error("Function call detected but couldn't extract it")
                    return {
                        'success': False,
                        'error': 'Unable to process the request. Please try again.',
                        'query': query,
                        'tools_used': conversation_history,
                        'iterations': iteration
                    }

                tool_name = function_call.name
                tool_params = dict(function_call.args)

                print(f"🔧 Agent calling tool: {tool_name}")
                print(f"   Parameters: {json.dumps(tool_params, indent=2)}")

                # Execute the tool
                try:
                    tool_result = self.tools.execute_tool(tool_name, tool_params)
                    print(f"✓ Tool executed successfully\n")
                    logger.info(f"Tool {tool_name} executed with {len(tool_result.get('destinations', []))} results")
                except Exception as e:
                    logger.error(f"Error executing tool {tool_name}: {e}")
                    return {
                        'success': False,
                        'error': f'Error executing search: {str(e)}',
                        'query': query,
                        'tools_used': conversation_history,
                        'iterations': iteration
                    }

                # Record tool execution BEFORE sending result back to model
                # This ensures conversation_history is accurate even if sending the result fails
                tool_record = {
                    'tool': tool_name,
                    'params': tool_params,
                    'result': tool_result
                }
                conversation_history.append(tool_record)

                # Log tool result details for debugging
                result_json = json.dumps(tool_result)
                result_size = len(result_json)
                result_size_kb = result_size / 1024
                logger.info(f"Tool result size: {result_size} bytes ({result_size_kb:.2f} KB)")

                # Truncate tool result if it's too large to prevent finish_reason 12 errors
                max_result_size = 150000  # 150KB limit
                truncated_result = tool_result
                final_result_size = result_size
                final_result_size_kb = result_size_kb

                if result_size > max_result_size:
                    logger.warning(f"Large tool result detected ({result_size_kb:.2f} KB) - truncating to prevent finish_reason 12")

                    # If the result contains destinations, truncate the list
                    if isinstance(tool_result, dict) and 'destinations' in tool_result:
                        original_count = len(tool_result['destinations'])
                        truncated_result = tool_result.copy()

                        # Estimate how many destinations we can include
                        # Assume each destination is roughly equal size
                        avg_dest_size = result_size / original_count if original_count > 0 else result_size
                        max_destinations = max(1, int(max_result_size / avg_dest_size * 0.8))  # 80% of limit for safety

                        truncated_result['destinations'] = tool_result['destinations'][:max_destinations]
                        truncated_result['truncated'] = True
                        truncated_result['original_count'] = original_count
                        truncated_result['showing_count'] = max_destinations
                        truncated_result['note'] = f"Results truncated from {original_count} to {max_destinations} destinations due to size limits. Please use more specific filters to see all results."

                        # Recalculate size after truncation
                        truncated_json = json.dumps(truncated_result)
                        final_result_size = len(truncated_json)
                        final_result_size_kb = final_result_size / 1024
                        logger.info(f"Truncated result size: {final_result_size} bytes ({final_result_size_kb:.2f} KB)")
                        logger.info(f"Truncated from {original_count} to {max_destinations} destinations")
                    else:
                        # For non-destination results, just add a warning
                        logger.warning(f"Tool result is too large but doesn't contain destinations list - cannot truncate")

                # Format result for the model
                function_response = genai.protos.Part(
                    function_response=genai.protos.FunctionResponse(
                        name=tool_name,
                        response={'result': truncated_result}
                    )
                )

                # Send tool result back to model with retry logic
                for attempt in range(max_retries):
                    try:
                        response = chat.send_message(function_response)

                        # Check for finish_reason issues (safety, content filtering, etc.)
                        if response.candidates:
                            candidate = response.candidates[0]
                            finish_reason = candidate.finish_reason

                            # Get the finish_reason value (1 = STOP/normal completion)
                            finish_reason_value = finish_reason.value if hasattr(finish_reason, 'value') else finish_reason

                            # Log finish_reason if it's not a normal completion (1 = STOP)
                            if finish_reason_value and finish_reason_value != 1:
                                logger.warning(f"Response finish_reason: {finish_reason} (value: {finish_reason_value})")

                                # Check for safety/content filtering (3 = SAFETY)
                                if finish_reason_value == 3:
                                    return {
                                        'success': False,
                                        'error': 'Content filtered by safety settings. Please try rephrasing your query or adjusting filters.',
                                        'query': query,
                                        'tools_used': conversation_history,
                                        'iterations': iteration
                                    }

                                # Check for unexpected tool call (12 = UNEXPECTED_TOOL_CALL)
                                if finish_reason_value == 12:
                                    logger.warning(f"Finish reason 12 after sending {tool_name} result ({final_result_size_kb:.2f} KB)")
                                    error_msg = f'The AI model encountered an issue processing the tool result from {tool_name}.'
                                    if final_result_size > 100000:
                                        error_msg += f' The result size ({final_result_size_kb:.2f} KB) may be too large.'
                                    error_msg += ' Please try again with more specific filters or search criteria.'
                                    return {
                                        'success': False,
                                        'error': error_msg,
                                        'query': query,
                                        'tools_used': conversation_history,
                                        'iterations': iteration
                                    }

                        if response.candidates and response.candidates[0].content.parts:
                            break
                        else:
                            logger.warning(f"Empty response after tool call (attempt {attempt + 1}/{max_retries})")
                            if attempt < max_retries - 1:
                                time.sleep(1)
                    except Exception as e:
                        error_str = str(e)
                        logger.error(f"Error sending tool result to Gemini (attempt {attempt + 1}/{max_retries}): {error_str}")

                        # Check if the error is related to finish_reason
                        if "finish_reason" in error_str.lower():
                            # Try to extract the finish_reason value from the error message
                            if "finish_reason: 12" in error_str or "finish_reason:12" in error_str:
                                logger.warning(f"Exception with finish_reason 12 after {tool_name} ({final_result_size_kb:.2f} KB): {error_str}")
                                error_msg = f'The AI model encountered an issue processing the tool result from {tool_name}.'
                                if final_result_size > 100000:
                                    error_msg += f' The result size ({final_result_size_kb:.2f} KB) may be too large.'
                                error_msg += ' Please try again with more specific filters or search criteria.'
                                return {
                                    'success': False,
                                    'error': error_msg,
                                    'query': query,
                                    'tools_used': conversation_history,
                                    'iterations': iteration
                                }
                            elif "finish_reason: 3" in error_str or "finish_reason:3" in error_str:
                                return {
                                    'success': False,
                                    'error': 'Content filtered by safety settings. Please try rephrasing your query or adjusting filters.',
                                    'query': query,
                                    'tools_used': conversation_history,
                                    'iterations': iteration
                                }

                        if attempt < max_retries - 1:
                            time.sleep(1)
                        else:
                            return {
                                'success': False,
                                'error': f'API error after tool execution: {error_str}',
                                'query': query,
                                'tools_used': conversation_history,
                                'iterations': iteration
                            }

                # Note: conversation_history.append() moved earlier to ensure
                # tool execution is recorded even if sending the result fails

                iteration += 1
            else:
                # Model has finished and provided final response
                break

        # Check if we hit max iterations with a pending function call
        if response.candidates and response.candidates[0].content.parts:
            has_pending_call = any(
                hasattr(part, 'function_call') and part.function_call
                for part in response.candidates[0].content.parts
            )
            if has_pending_call:
                logger.warning(f"Max iterations ({max_iterations}) reached with pending function call")
                return {
                    'success': False,
                    'error': 'No destinations found matching your criteria. Try adjusting your filters or search parameters.',
                    'query': query,
                    'tools_used': conversation_history,
                    'iterations': iteration
                }

        # Extract final response - check if response contains text
        try:
            # Verify the response has valid text content
            if not response.candidates or not response.candidates[0].content.parts:
                logger.error("Final response has no content parts")
                return {
                    'success': False,
                    'error': 'Unable to process the search results. Please try again with different search criteria.',
                    'query': query,
                    'tools_used': conversation_history,
                    'iterations': iteration
                }

            # Check if any part still has a function_call
            has_function_call = any(
                hasattr(part, 'function_call') and part.function_call
                for part in response.candidates[0].content.parts
            )

            if has_function_call:
                logger.error("Response still contains function calls, cannot extract text")
                return {
                    'success': False,
                    'error': 'Unable to complete your request. Please try again with different search criteria.',
                    'query': query,
                    'tools_used': conversation_history,
                    'iterations': iteration
                }

            # Now safely extract text
            final_response = response.text

            # Verify we got actual text content
            if not final_response or not final_response.strip():
                logger.error("Final response is empty")
                return {
                    'success': False,
                    'error': 'Unable to generate recommendations. Please try again with different search criteria.',
                    'query': query,
                    'tools_used': conversation_history,
                    'iterations': iteration
                }

        except Exception as e:
            logger.error(f"Error extracting final response: {e}")
            return {
                'success': False,
                'error': 'Unable to process the search results. Please try again with different search criteria.',
                'query': query,
                'tools_used': conversation_history,
                'iterations': iteration
            }

        print(f"\n✨ Recommendations ready!\n")

        return {
            'success': True,
            'recommendations': final_response,
            'query': query,
            'tools_used': conversation_history,
            'iterations': iteration
        }

    def _create_query(self,
                     destination_type: str,
                     city: Optional[str],
                     hours: Optional[float],
                     budget: Optional[float]) -> str:
        """Create a natural language query from user parameters (DEPRECATED - used by legacy methods only)"""

        # Create query based on whether city is specified
        if city is not None:
            # For city-based searches, guide the agent through the correct tool sequence
            query = f"I need help finding {destination_type} destinations in or near {city}, India.\n"

            # Add filter details
            if hours is not None:
                hours_str = f"{hours:g}"
                query += f"\nVisit duration: {hours_str} hours."

            if budget is not None:
                if budget == 0:
                    query += f"\nBudget: Free (no cost)."
                else:
                    query += f"\nBudget: around ₹{budget:,.0f}."

            # Add tool instruction with proper sequence
            query += f'\n\nPlease search for destinations first using search_destinations_quantitative with destination_type="{destination_type}"'
            if hours is not None:
                query += f', max_hours={hours:g}'
            if budget is not None:
                query += f', max_budget={budget}'
            query += f'. Then use filter_by_city with city="{city}" to filter the results to destinations in or near {city}.'
        else:
            # Build filter description for non-city queries
            filter_parts = [f"{destination_type} destinations"]

            if hours is not None:
                hours_str = f"{hours:g}"
                filter_parts.append(f"visit duration within {hours_str} hours")

            if budget is not None:
                if budget == 0:
                    filter_parts.append("free (no cost)")
                else:
                    filter_parts.append(f"budget around ₹{budget:,.0f}")

            filters_desc = ", ".join(filter_parts)

            query = f"I'm looking for travel destination recommendations in India: {filters_desc}."
            query += "\n\nPlease use the available tools to search the dataset and provide me with your top 3 recommendations with reasoning."

        return query

    def chat(self, message: str) -> str:
        """
        Have a conversation with the agent

        Args:
            message: User message

        Returns:
            Agent response
        """
        chat = self.model.start_chat()

        # Send message with retry logic
        max_retries = 3
        response = None

        for attempt in range(max_retries):
            try:
                response = chat.send_message(message, tools=self.tool_config)
                if response.candidates and response.candidates[0].content.parts:
                    break
                else:
                    logger.warning(f"Empty response in chat (attempt {attempt + 1}/{max_retries})")
                    if attempt < max_retries - 1:
                        time.sleep(1)
            except Exception as e:
                logger.error(f"Error in chat (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)

        if not response or not response.candidates or not response.candidates[0].content.parts:
            return "I apologize, but I'm having trouble connecting to the AI service. Please try again."

        # Handle tool calls if any
        max_iterations = 5
        iteration = 0

        while iteration < max_iterations:
            # Check if response has candidates and parts
            if not response.candidates or not response.candidates[0].content.parts:
                logger.warning(f"Empty response during chat tool calling at iteration {iteration}")
                return "I apologize, but I couldn't find any relevant information to answer your question."

            # Check if any part has a function call
            has_function_call = any(
                hasattr(part, 'function_call') and part.function_call
                for part in response.candidates[0].content.parts
            )

            if has_function_call:
                # Extract function call - find the part that has it
                function_call = None
                for part in response.candidates[0].content.parts:
                    if hasattr(part, 'function_call') and part.function_call:
                        function_call = part.function_call
                        break

                if not function_call:
                    logger.warning("Function call detected but couldn't extract it in chat")
                    return "I apologize, but I couldn't process the tool call properly. Please try again."

                tool_name = function_call.name
                tool_params = dict(function_call.args)

                tool_result = self.tools.execute_tool(tool_name, tool_params)

                function_response = genai.protos.Part(
                    function_response=genai.protos.FunctionResponse(
                        name=tool_name,
                        response={'result': tool_result}
                    )
                )

                response = chat.send_message(function_response)
                iteration += 1
            else:
                break

        # Check if response still has a pending function call
        if response.candidates and response.candidates[0].content.parts:
            has_pending_call = any(
                hasattr(part, 'function_call') and part.function_call
                for part in response.candidates[0].content.parts
            )
            if has_pending_call:
                logger.warning(f"Max iterations reached in chat with pending function call")
                return "I apologize, but I couldn't complete your request. Please try rephrasing your question or being more specific."

        try:
            # Verify the response has valid text content
            if not response.candidates or not response.candidates[0].content.parts:
                logger.error("Chat response has no content parts")
                return "I apologize, but I couldn't generate a response. Please try again."

            # Check if any part still has a function_call
            has_function_call = any(
                hasattr(part, 'function_call') and part.function_call
                for part in response.candidates[0].content.parts
            )

            if has_function_call:
                logger.error("Chat response still contains function calls, cannot extract text")
                return "I apologize, but I couldn't complete processing your message. Please try again."

            return response.text
        except Exception as e:
            logger.error(f"Error extracting chat response: {e}")
            return "I apologize, but I encountered an error processing your message. Please try again."

    def analyze_destination(self,
                          destination_name: str) -> str:
        """
        Get detailed analysis of a specific destination

        Args:
            destination_name: Name of the destination

        Returns:
            Detailed analysis text
        """
        details = self.dataset.get_destination_details(destination_name)

        if not details:
            return f"Destination '{destination_name}' not found in the dataset."

        # Use LLM to analyze
        prompt = f"""Analyze this Indian travel destination for families:

Destination: {destination_name}

Details:
{json.dumps(details, indent=2)}

Provide:
1. Overview and highlights
2. Why it's good (or not) for families
3. Practical tips and recommendations
4. Best time to visit
"""

        response = self.model.generate_content(prompt)
        return response.text