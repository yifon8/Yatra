"""
Yatra Destination Suggester Agent
Main agent logic using Google ADK (Gemini) for travel recommendations
"""

import google.generativeai as genai
from typing import Dict, Any, List, Optional
import os
import json

from .dataset_handler import DatasetHandler
from .tools import TravelTools, format_tool_result
from .system_prompts import (
    AGENT_SYSTEM_PROMPT,
    get_recommendation_prompt,
    get_destination_analysis_prompt
)


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
        self.model = genai.GenerativeModel(
            model_name='gemini-1.5-flash',  # or 'gemini-1.5-pro' for better quality
            system_instruction=AGENT_SYSTEM_PROMPT
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
            tools.append(genai.protos.Tool(
                function_declarations=[
                    genai.protos.FunctionDeclaration(
                        name=tool_decl["name"],
                        description=tool_decl["description"],
                        parameters=tool_decl["parameters"]
                    )
                ]
            ))

        return tools

    def suggest_destinations(self,
                           destination_type: Optional[str] = None,
                           has_small_child: bool = False,
                           hours: Optional[int] = None,
                           budget: Optional[float] = None,
                           max_iterations: int = 5) -> Dict[str, Any]:
        """
        Suggest destinations based on user preferences

        Args:
            destination_type: Type of destination (beach, mountain, heritage, wildlife)
            has_small_child: Whether traveling with a small child
            hours: Available time in hours
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

        # Send initial query with tools
        response = chat.send_message(
            query,
            tools=self.tool_config
        )

        # Handle tool calling loop
        iteration = 0
        conversation_history = []

        while iteration < max_iterations:
            # Check if model wants to use tools
            if response.candidates[0].content.parts[0].function_call:
                # Extract function call
                function_call = response.candidates[0].content.parts[0].function_call
                tool_name = function_call.name
                tool_params = dict(function_call.args)

                print(f"🔧 Agent calling tool: {tool_name}")
                print(f"   Parameters: {json.dumps(tool_params, indent=2)}")

                # Execute the tool
                tool_result = self.tools.execute_tool(tool_name, tool_params)

                print(f"✓ Tool executed successfully\n")

                # Format result for the model
                function_response = genai.protos.Part(
                    function_response=genai.protos.FunctionResponse(
                        name=tool_name,
                        response={'result': tool_result}
                    )
                )

                # Send tool result back to model
                response = chat.send_message(function_response)

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
                     hours: Optional[int],
                     budget: Optional[float]) -> str:
        """Create a natural language query from user parameters"""
        query_parts = ["I'm looking for travel destination recommendations in India"]

        if destination_type:
            query_parts.append(f"focusing on {destination_type} destinations")

        if has_small_child:
            query_parts.append("that are suitable for families with small children")

        if hours:
            query_parts.append(f"where we can visit within {hours} hours")

        if budget:
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
        response = chat.send_message(message, tools=self.tool_config)

        # Handle tool calls if any
        max_iterations = 5
        iteration = 0

        while iteration < max_iterations:
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
