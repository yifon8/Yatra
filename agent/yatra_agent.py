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
                           has_small_child: bool = False,
                           hours: Optional[float] = None,
                           budget: Optional[float] = None,
                           max_iterations: int = 5) -> Dict[str, Any]:
        """
        Suggest destinations based on user preferences

        Args:
            destination_type: Type of destination (beach, mountain, heritage, wildlife)
            has_small_child: Whether traveling with a small child
            hours: Available time in hours (can be decimal, e.g., 0.5, 1.5, 8.25)
            budget: Budget in rupees
            max_iterations: Maximum tool calling iterations

        Returns:
            Dictionary with recommendations and reasoning
        """
        # Create the user query
        query = self._create_query(destination_type, has_small_child, hours, budget)

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
                        if response.candidates and response.candidates[0].content.parts:
                            break
                        else:
                            logger.warning(f"Empty response after tool call (attempt {attempt + 1}/{max_retries})")
                            if attempt < max_retries - 1:
                                time.sleep(1)
                    except Exception as e:
                        logger.error(f"Error sending tool result to Gemini (attempt {attempt + 1}/{max_retries}): {e}")
                        if attempt < max_retries - 1:
                            time.sleep(1)
                        else:
                            return {
                                'success': False,
                                'error': f'API error after tool execution: {str(e)}',
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

        # Extract final response
        final_response = response.text

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
                     has_small_child: bool,
                     hours: Optional[float],
                     budget: Optional[float]) -> str:
        """Create a natural language query from user parameters"""
        query_parts = ["I'm looking for travel destination recommendations in India"]

        if destination_type:
            query_parts.append(f"focusing on {destination_type} destinations")

        if has_small_child:
            query_parts.append("that are suitable for families with small children")

        if hours is not None:
            # Format hours nicely (remove .0 for whole numbers)
            hours_str = f"{hours:g}"
            query_parts.append(f"where we can visit within {hours_str} hours")

        if budget is not None:
            if budget == 0:
                query_parts.append("preferably free destinations (no cost)")
            else:
                query_parts.append(f"with a budget of around ₹{budget:,.0f}")

        query = " ".join(query_parts) + "."
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

        return response.text

    def analyze_destination(self,
                          destination_name: str,
                          has_small_child: bool = False) -> str:
        """
        Get detailed analysis of a specific destination

        Args:
            destination_name: Name of the destination
            has_small_child: Whether considering for families with small children

        Returns:
            Detailed analysis text
        """
        details = self.dataset.get_destination_details(destination_name)

        if not details:
            return f"Destination '{destination_name}' not found in the dataset."

        # Use LLM to analyze
        prompt = f"""Analyze this Indian travel destination for families{"with small children" if has_small_child else ""}:

Destination: {destination_name}

Details:
{json.dumps(details, indent=2)}

Provide:
1. Overview and highlights
2. Why it's good (or not) for families{"with small children" if has_small_child else ""}
3. Practical tips and recommendations
4. Best time to visit
"""

        response = self.model.generate_content(prompt)
        return response.text
