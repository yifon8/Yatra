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
                           destination_type: Optional[str] = None,
                           city: Optional[str] = None,
                           hours: Optional[float] = None,
                           budget: Optional[float] = None,
                           max_iterations: int = 5) -> Dict[str, Any]:
        """
        Suggest destinations based on user preferences

        Args:
            destination_type: Type of destination (beach, mountain, heritage, wildlife)
            city: City name to find destinations in or near (optional, max 30 chars)
            hours: Available time in hours (can be decimal, e.g., 0.5, 1.5, 8.25)
            budget: Budget in rupees
            max_iterations: Maximum tool calling iterations

        Returns:
            Dictionary with recommendations and reasoning
        """
        # Create the user query
        query = self._create_query(destination_type, city, hours, budget)

        print(f"\n🔍 Processing query: {query}\n")

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

            # Check if model wants to use tools
            if response.candidates[0].content.parts[0].function_call:
                # Extract function call
                function_call = response.candidates[0].content.parts[0].function_call
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

                # Format result for the model
                function_response = genai.protos.Part(
                    function_response=genai.protos.FunctionResponse(
                        name=tool_name,
                        response={'result': tool_result}
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
                                    return {
                                        'success': False,
                                        'error': 'The AI model encountered an issue processing the tool result. This may be due to the data format. Please try again with different search criteria.',
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
                                return {
                                    'success': False,
                                    'error': 'The AI model encountered an issue processing the tool result. This may be due to the data format or size. Please try again with different search criteria or more specific filters.',
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

                conversation_history.append({
                    'tool': tool_name,
                    'params': tool_params,
                    'result': tool_result
                })

                iteration += 1
            else:
                # Model has finished and provided final response
                break

        # Check if we hit max iterations with a pending function call
        if (response.candidates and response.candidates[0].content.parts and
            response.candidates[0].content.parts[0].function_call):
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
                     destination_type: Optional[str],
                     city: Optional[str],
                     hours: Optional[float],
                     budget: Optional[float]) -> str:
        """Create a natural language query from user parameters"""

        # Create query based on whether city is specified
        if city is not None:
            # For city-based searches, always use "cities" in the query
            # But also include destination type and other filters
            query = f"I need help finding cities in or next to {city}, India.\n"

            # Add filter details
            if destination_type:
                query += f"\nI'm interested in {destination_type} destinations."

            if hours is not None:
                hours_str = f"{hours:g}"
                query += f"\nVisit duration: {hours_str} hours."

            if budget is not None:
                if budget == 0:
                    query += f"\nBudget: Free (no cost)."
                else:
                    query += f"\nBudget: around ₹{budget:,.0f}."

            # Add tool instruction
            query += f'\n\nUse the filter_by_city tool with city="{city}".'
        else:
            # Build filter description for non-city queries
            filter_parts = []
            if destination_type:
                filter_parts.append(f"{destination_type} destinations")
            else:
                filter_parts.append("destinations")

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

            if response.candidates[0].content.parts[0].function_call:
                function_call = response.candidates[0].content.parts[0].function_call
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
        if (response.candidates and response.candidates[0].content.parts and
            response.candidates[0].content.parts[0].function_call):
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