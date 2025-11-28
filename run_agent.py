#!/usr/bin/env python3
"""
Yatra Travel Agent - Command Line Interface
Run the destination suggester agent from the command line
"""

import argparse
import os
import sys
from typing import Optional

from agent import DestinationSuggester


def print_banner():
    """Print Yatra banner"""
    print("""
╔══════════════════════════════════════════════════════════╗
║                                                          ║
║   🛕  YATRA - Your Family's Travel Agent for India  🛕   ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
    """)


def print_recommendations(result: dict):
    """Pretty print recommendations"""
    print("\n" + "="*60)
    print("📍 YATRA'S RECOMMENDATIONS")
    print("="*60 + "\n")

    print(result['recommendations'])

    if result.get('tools_used'):
        print("\n" + "-"*60)
        print(f"🔧 Tools used: {result['iterations']} tool calls")
        for i, tool_call in enumerate(result['tools_used'], 1):
            print(f"   {i}. {tool_call['tool']}")

    print("\n" + "="*60 + "\n")


def interactive_mode(agent: DestinationSuggester):
    """Run agent in interactive mode"""
    print_banner()
    print("Welcome to Yatra! Let's find the perfect destination for your family.\n")

    # Collect user preferences
    print("Please answer a few questions:\n")

    # Destination type (REQUIRED)
    print("1. What type of destination are you interested in? (REQUIRED)")
    print("   a) Beach & Coastal")
    print("   b) Mountain & Hill Stations")
    print("   c) Heritage & Cultural")
    print("   d) Wildlife & Nature")

    type_map = {
        'a': 'beach',
        'b': 'mountain',
        'c': 'heritage',
        'd': 'wildlife'
    }

    type_choice = input("\nYour choice (a/b/c/d): ").lower().strip()
    destination_type = type_map.get(type_choice)

    while not destination_type:
        print("❌ Destination type is required! Please choose a, b, c, or d.")
        type_choice = input("Your choice (a/b/c/d): ").lower().strip()
        destination_type = type_map.get(type_choice)

    # City (optional)
    print("\n2. Which city would you like to explore (or nearby)?")
    city_input = input("   City name (or press Enter to skip): ").strip()
    city = city_input if city_input else None

    # Duration
    print("\n3. How many hours do you have for the visit?")
    print("   (You can enter decimal values like 0.5, 1.5, 8.25)")
    hours_input = input("   Hours (or press Enter to skip): ").strip()
    hours = float(hours_input) if hours_input else None

    # Budget
    print("\n4. What is your budget in Indian Rupees?")
    budget_input = input("   Budget ₹ (or press Enter to skip): ").strip()
    budget = float(budget_input) if budget_input else None

    # Run agent
    print("\n🔍 Searching for the best destinations...\n")

    try:
        result = agent.suggest_destinations(
            destination_type=destination_type,
            city=city,
            hours=hours,
            budget=budget
        )

        print_recommendations(result)

    except Exception as e:
        print(f"\n❌ Error: {str(e)}\n")
        sys.exit(1)


def command_line_mode(agent: DestinationSuggester, args):
    """Run agent with command line arguments"""
    print_banner()

    # Check that destination_type is provided (REQUIRED)
    if not args.type:
        print("❌ Error: --type is required!")
        print("   Please specify one of: beach, mountain, heritage, wildlife")
        print("\n   Example: python run_agent.py --type beach --hours 4 --budget 5000\n")
        sys.exit(1)

    try:
        result = agent.suggest_destinations(
            destination_type=args.type,
            city=args.city,
            hours=args.hours,
            budget=args.budget
        )

        print_recommendations(result)

    except Exception as e:
        print(f"\n❌ Error: {str(e)}\n")
        sys.exit(1)


def chat_mode(agent: DestinationSuggester):
    """Run agent in chat mode"""
    print_banner()
    print("Chat with Yatra! Type 'quit' or 'exit' to end the conversation.\n")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("\nYatra: Safe travels! 🌏\n")
            break

        if not user_input:
            continue

        try:
            response = agent.chat(user_input)
            print(f"\nYatra: {response}\n")
        except Exception as e:
            print(f"\nYatra: Sorry, I encountered an error: {str(e)}\n")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Yatra - Travel destination suggester for India',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (recommended for first-time users)
  python run_agent.py -i

  # Chat mode
  python run_agent.py -c

  # Command line mode with specific parameters (hours can be decimal)
  python run_agent.py --type beach --child --hours 8.5 --budget 5000

  # Use custom dataset
  python run_agent.py -i --csv data/my_dataset.csv
        """
    )

    parser.add_argument(
        '-i', '--interactive',
        action='store_true',
        help='Run in interactive mode (asks questions)'
    )

    parser.add_argument(
        '-c', '--chat',
        action='store_true',
        help='Run in chat mode (free-form conversation)'
    )

    parser.add_argument(
        '--type',
        choices=['beach', 'mountain', 'heritage', 'wildlife'],
        help='Type of destination (REQUIRED for non-interactive mode)'
    )

    parser.add_argument(
        '--city',
        type=str,
        help='City name to find destinations in or near (optional)'
    )

    parser.add_argument(
        '--hours',
        type=float,
        help='Available time in hours (can be decimal, e.g., 0.5, 1.5, 8.25)'
    )

    parser.add_argument(
        '--budget',
        type=float,
        help='Budget in Indian Rupees'
    )

    parser.add_argument(
        '--csv',
        type=str,
        help='Path to CSV dataset (default: data/destinations.csv)'
    )

    parser.add_argument(
        '--api-key',
        type=str,
        help='Google API key (or set GOOGLE_API_KEY env variable)'
    )

    args = parser.parse_args()

    # Check for API key
    api_key = args.api_key or os.getenv('GOOGLE_API_KEY')
    if not api_key:
        print("❌ Error: Google API key required!")
        print("\nSet it using one of these methods:")
        print("  1. Environment variable: export GOOGLE_API_KEY='your-key'")
        print("  2. Command line argument: --api-key 'your-key'")
        print("\nGet your API key at: https://aistudio.google.com/app/apikey\n")
        sys.exit(1)

    # Initialize agent
    try:
        print("Initializing Yatra agent...")
        agent = DestinationSuggester(api_key=api_key, csv_path=args.csv)
    except FileNotFoundError as e:
        print(f"\n❌ {str(e)}")
        print("\nPlease download the dataset from:")
        print("https://www.kaggle.com/datasets/saketk511/travel-dataset-guide-to-indias-must-see-places")
        print("\nAnd place it in the data/ directory as 'destinations.csv'\n")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error initializing agent: {str(e)}\n")
        sys.exit(1)

    # Run in appropriate mode
    if args.chat:
        chat_mode(agent)
    elif args.interactive or not any([args.type, args.hours, args.budget]):
        # Default to interactive if no parameters provided
        interactive_mode(agent)
    else:
        command_line_mode(agent, args)


if __name__ == '__main__':
    main()
