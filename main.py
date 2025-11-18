#
# Vercel AI SDK version using the Python AI SDK with web search
#

import os
import time
from dotenv import load_dotenv
from duckduckgo_search import DDGS

from ai_sdk import generate_text, tool, openai

# Load environment variables from .env file
load_dotenv()

api_key = os.getenv("INCEPTION_API_KEY")
if api_key is None:
    raise Exception("INCEPTION_API_KEY not found in environment variables")


def web_search_execute(query: str, max_results: int = 5) -> str:
    """
    Search the web using DuckDuckGo and return formatted results.

    Args:
        query: The search query string
        max_results: Maximum number of results to return (default: 5)

    Returns:
        Formatted string with search results including titles, snippets, and URLs
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))

        if not results:
            return "No search results found."

        # Format results as a readable string
        formatted_results = []
        for i, result in enumerate(results, 1):
            title = result.get('title', 'No title')
            snippet = result.get('body', 'No description')
            url = result.get('href', 'No URL')
            formatted_results.append(
                f"{i}. {title}\n   {snippet}\n   URL: {url}"
            )

        return "\n\n".join(formatted_results)

    except Exception as e:
        return f"Error performing web search: {str(e)}"


# Create the web search tool with JSON schema parameters
web_search = tool(
    name="web_search",
    description="Search the web for current information using DuckDuckGo. Use this when you need up-to-date information or facts that you don't already know.",
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query to look up on the web"
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results to return",
                "default": 5
            }
        },
        "required": ["query"]
    },
    execute=web_search_execute
)


# Custom model wrapper for Inception API
class InceptionModel:
    """
    Wrapper for Inception API to work with ai-sdk-python.
    Uses the OpenAI-compatible interface.
    """
    def __init__(self, api_key: str, model_id: str = "mercury"):
        self.api_key = api_key
        self.model_id = model_id
        self.base_url = "https://api.inceptionlabs.ai/v1"

    def __call__(self, model_name: str = None):
        """Return an OpenAI-compatible model instance"""
        import openai as openai_client

        # Configure OpenAI client to use Inception API
        client = openai_client.OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

        # Use the ai_sdk openai provider with custom configuration
        from ai_sdk import openai as ai_openai
        return ai_openai(model_name or self.model_id)


def main() -> None:
    """
    Main loop for the AI agent with web search capability.
    Continuously prompts user for questions and provides answers using the LLM and web search tool.
    """
    # Set up OpenAI client to use Inception API
    os.environ["OPENAI_API_KEY"] = api_key
    os.environ["OPENAI_BASE_URL"] = "https://api.inceptionlabs.ai/v1"

    # Initialize the model with Inception's mercury model
    model = openai("mercury")

    print("Welcome! Ask me questions (type 'exit' or 'quit' to stop).\n")
    print("I can search the web for current information to answer your questions.\n")

    while True:
        try:
            question = input("Your question: ").strip()

            if question.lower() in ['exit', 'quit', '']:
                print("Goodbye!")
                break

            # Start timing
            start_time = time.time()

            # Generate response with web search tool available
            result = generate_text(
                model=model,
                prompt=question,
                tools=[web_search],
                max_tokens=1000
            )

            # Calculate elapsed time
            elapsed_time = time.time() - start_time

            print(f"\nAnswer: {result.text}")
            print(f"(Response time: {elapsed_time:.2f}s)\n")

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}\n")
            continue


if __name__ == "__main__":
    main()
