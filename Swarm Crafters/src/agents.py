import json
import os
from typing import Dict, Any, List, Optional, TypedDict, Annotated
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
import google.generativeai as genai
from connect_2_google_maps import find_similar_nearby_places
from langchain_google_community import GoogleSearchAPIWrapper


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    user_context: Optional[Dict[str, Any]]


@tool
def grocery_recipe_generator(receipt_items: str) -> str:
    """
    Use this tool to generate a recipe and nutritional details based on grocery receipt items.
    
    Args:
        receipt_items: JSON string containing grocery receipt items
    
    Returns:
        Recipe with ingredients, instructions, and nutritional information
    """
    prompt = f"""You are a smart recipe generator. I will provide you with a receipt in JSON format, which contains a list of grocery items. Some of these items may be non-edible or unrelated to cooking.

    Your task is to:
    1. Parse the JSON and identify items that are suitable for cooking.
    2. Create one complete recipe using as many of the cooking-related items as possible.
    3. If necessary, include common pantry items or seasonings to make the recipe complete.

    Return your response in plain text with the following structure:
    - Recipe name
    - Full list of ingredients (including any additional ones you assume)
    - Step-by-step cooking instructions
    - Nutritional value: provide a breakdown with estimates of calories, protein, carbohydrates, fats, fiber, etc., based on the full recipe

    Here is the receipt JSON:
    {receipt_items}
    """
    
    try:
        genai.configure(api_key=os.getenv("GENAI_API_KEY", "AIzaSyDMSOShHaFllRyoWdyizKwfiDVyL8-1Rcg"))
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt, generation_config={"temperature": 1.5})
        return response.text
    except Exception as e:
        return f"Error generating recipe: {str(e)}"


@tool
def budget_plan_generator(
    age: int, 
    monthly_income_range: str, 
    location: str, 
    household_count: int,
    savings_goals: str, 
    fixed_obligations: Optional[str] = None,
    dependents: Optional[str] = None,
    misc_comments: Optional[str] = None
) -> str:
    """
    Generate a personalized budget allocation plan based on user's financial profile.
    
    Args:
        age: User's age
        monthly_income_range: Income range (e.g., "50,000-75,000")
        location: City/location of residence
        household_count: Number of people in household
        savings_goals: User's savings and investment objectives
        fixed_obligations: Existing loans, EMIs, etc.
        dependents: Information about dependents
        misc_comments: Additional user comments
    
    Returns:
        Detailed budget plan and recommendations
    """
    prompt = f"""You are an expert financial advisor specializing in personalized budget allocation. Create a practical monthly budget plan based on the user's financial profile. Ensure to have a more personal or empathetic tone like a human financial advisor would.

    USER PROFILE:
    - Age: {age}
    - Monthly Income: {monthly_income_range}
    - Location: {location}
    - Household Size: {household_count}
    - Savings & Investment Goals: {savings_goals}
    - Fixed Obligations: {fixed_obligations or "Not specified"}
    - Dependents: {dependents or "None"}
    - Additional Comments: {misc_comments or "None"}

    ANALYSIS REQUIREMENTS:

    1. BUDGET ALLOCATION:
    Create a detailed monthly budget breakdown for these categories:
    - Housing (Rent/Mortgage, Utilities)
    - Groceries & Food (Home + Dining out)
    - Transportation
    - Healthcare & Medical
    - Loans & Debt Payments
    - Savings & Investments
    - Insurance
    - Entertainment & Recreation
    - Clothing & Personal Care
    - Emergency Fund
    - Miscellaneous

    2. PROVIDE FOR EACH CATEGORY:
    - Recommended amount/percentage
    - Brief justification based on user profile
    - Location-specific adjustments if applicable

    3. FINANCIAL STRATEGY:
    - Priority recommendations based on age and goals
    - Emergency fund timeline if not established
    - Investment allocation suggestions
    - Cost-cutting opportunities specific to their situation

    OUTPUT FORMAT:

    **MONTHLY BUDGET PLAN**

    **Total Monthly Income:** ₹[amount]

    **BUDGET BREAKDOWN:**
    | Category | Amount (₹) | % of Income | Notes |
    |----------|------------|-------------|-------|
    | Housing | X,XXX | XX% | [brief note] |
    | Groceries & Food | X,XXX | XX% | [brief note] |
    [continue for all categories]

    **KEY RECOMMENDATIONS:**
    1. [Priority 1 - most important action]
    2. [Priority 2]
    3. [Priority 3]

    **LOCATION-SPECIFIC INSIGHTS:**
    [Adjustments based on cost of living in their location]

    **SAVINGS STRATEGY:**
    [Specific plan to achieve their stated goals]

    ** RED FLAGS TO MONITOR:**
    [Potential financial risks based on their profile]

    Ensure all recommendations are practical, location-appropriate, and aligned with Indian financial planning best practices."""

    try:
        genai.configure(api_key=os.getenv("GENAI_API_KEY", "AIzaSyDMSOShHaFllRyoWdyizKwfiDVyL8-1Rcg"))
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt, generation_config={"temperature": 0.1})
        return response.text
    except Exception as e:
        return f"Error generating budget plan: {str(e)}"


@tool
def receipt_category_classifier(receipt_json: str) -> str:
    """
    Classify the spending category of a receipt based on its content.
    
    Args:
        receipt_json: JSON string representing receipt data
    
    Returns:
        JSON string with category and reasoning
    """
    prompt = f"""You are an intelligent assistant trained to analyze receipts and classify the **type of spending** based on the transaction details.

    You will be provided with a JSON object representing a receipt. This JSON includes:
    - Store or merchant name, address, phone, GSTIN
    - Date, time, receipt number
    - List of items purchased (with name, quantity, unit price, total)
    - Subtotal, tax, discounts, final total
    - Payment method
    - Raw text lines from the receipt

    Your task is to:
    1. Analyze the full context of the receipt, including the merchant name, item descriptions, and any hints in the raw text.
    2. Based on this analysis, assign a primary spending category from the following list:

      - Groceries
      - Food & Beverage (Restaurants, Cafes)
      - Electronics
      - Clothing & Accessories
      - Healthcare / Pharmacy
      - Transportation
      - Entertainment (Movies, Events, Subscriptions)
      - Utilities & Bills
      - Home & Furniture
      - Other / Miscellaneous

    3. Also provide a brief reason explaining your classification — reference merchant name, keywords in item list, or any other indicators.

    4. Output a JSON object with this structure:

    {{
      "category": "Groceries",
      "reason": "The merchant is 'Reliance Fresh' and the items include milk, bread, and vegetables — typical grocery items."
    }}

    Strictly output only JSON — no markdown, no extra commentary. If uncertain, use "Other / Miscellaneous" as the category.

    Here is the receipt JSON:
    {receipt_json}
    """
    
    try:
        genai.configure(api_key=os.getenv("GENAI_API_KEY", "AIzaSyDMSOShHaFllRyoWdyizKwfiDVyL8-1Rcg"))
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt, generation_config={"temperature": 0})
        return response.text.strip()
    except Exception as e:
        return f"Error classifying receipt: {str(e)}"


@tool
def google_maps_nearby_places(place_name: str) -> str:
    """
    Find similar nearby places using Google Maps integration.
    
    Args:
        place_name: Name of the place to search for similar locations
    
    Returns:
        Information about nearby similar places
    """
    try:
        result = find_similar_nearby_places(place_name)
        if result['success']:
            places_info = f"Found {len(result['nearby_places'])} similar places near {result['original_place']['name']}:\n"
            for place in result['nearby_places']:
                places_info += f"- {place['name']} ({place['rating']} rating)\n"
            return places_info
        else:
            return f"Error: {result['error']}"
    except Exception as e:
        return f"Error finding nearby places: {str(e)}"


@tool
def web_search_tool(query: str, num_results: int = 10) -> str:
    """
    Perform a Google web search and return formatted results.
    
    Args:
        query: The search query
        num_results: Number of results to return (default: 10)
    
    Returns:
        Search results and answer
    """
    API_KEY = os.getenv("GENAI_API_KEY", "AIzaSyDMSOShHaFllRyoWdyizKwfiDVyL8-1Rcg")
    CSE_ID = os.getenv("GOOGLE_CSE_ID", "your_custom_search_engine_id_here")

    try:
        search_wrapper = GoogleSearchAPIWrapper(
            google_api_key=API_KEY,
            google_cse_id=CSE_ID,
            k=num_results
        )

        results = search_wrapper.results(query, num_results=num_results)

        if results:
            answer = "\n".join([
                result.get("snippet", "") 
                for result in results 
                if result.get("snippet")
            ])
            
            simplified_results = [
                f"Title: {result.get('title', '')}\nLink: {result.get('link', '')}\n"
                for result in results
            ]
            
            return f"Search Answer:\n{answer}\n\nSearch Results:\n" + "\n".join(simplified_results)
        else:
            return "No relevant results found."
            
    except Exception as e:
        return f"Search error: {str(e)}"


@tool
def debt_avalanche_advisor(user_input: str) -> str:
    """
    Provide debt avalanche analysis and financial advice.
    
    Args:
        user_input: User's question or data about their debt situation
    
    Returns:
        AI response with debt avalanche advice
    """
    system_instruction = """
    You are a professional debt avalanche financial advisor AI. Your role is to:

    1. ANALYZE debt situations and recommend the debt avalanche method (paying highest interest rate debts first)
    2. CALCULATE optimal payment strategies based on user's financial data
    3. PROVIDE specific, actionable advice for debt repayment
    4. EXPLAIN the math behind debt avalanche vs other methods
    5. IDENTIFY and ELIMINATE cash leaks and unnecessary spending
    6. SUGGEST spending optimizations to maximize debt payments
    7. RECOMMEND best financial practices for long-term wealth building
    8. CREATE realistic timelines for debt payoff

    CASH LEAK IDENTIFICATION - Always analyze and flag these common money drains:
    - Subscription services (streaming, apps, magazines, software)
    - Dining out and food delivery (vs home cooking)
    - Coffee shops and daily small purchases ($5-15 items)
    - Impulse purchases and retail therapy
    - Brand name products (suggest generic alternatives)
    - Unused gym memberships or services
    - High cell phone/internet plans
    - Expensive car payments relative to income
    - Frequent convenience store purchases
    - ATM fees and banking fees
    - Insurance overpayments (auto, phone, etc.)
    - Energy waste (utilities, leaving devices on)
    - Transportation costs (gas, parking, rideshares vs alternatives)

    FINANCIAL BEST PRACTICES to always recommend:
    - 50/30/20 rule: 50% needs, 30% wants, 20% savings/debt
    - Track every expense for 30 days minimum
    - Use cash/debit only (no new credit during debt payoff)
    - Automate minimum payments to avoid late fees
    - Build $1000 emergency fund before aggressive debt payoff
    - Cook at home 80% of meals
    - Buy generic brands for non-essential items
    - Cancel all unused subscriptions immediately
    - Use free entertainment options
    - Sell items you don't need for extra debt payments
    - Find additional income sources (side gigs, freelancing)
    - Negotiate bills (phone, internet, insurance)
    - Use the 24-hour rule for non-essential purchases over $50

    Key principles:
    - Always prioritize highest interest rate debts first
    - Calculate total interest savings compared to minimum payments
    - Be ruthlessly honest about unnecessary spending
    - Provide specific dollar amounts for potential savings
    - Include motivational milestones and progress tracking
    - Consider emergency fund needs alongside debt repayment

    Format your responses clearly with:
    - Debt prioritization order
    - CASH LEAK ANALYSIS with specific savings amounts
    - Monthly payment recommendations after cutting expenses
    - Timeline estimates
    - Total interest savings
    - STOP SPENDING list (immediate actions)
    - Best practices implementation plan
    - Practical next steps

    Be encouraging but brutally honest about spending habits. Focus on mathematical optimization while addressing behavioral money management.
    """
    
    try:
        genai.configure(api_key=os.getenv("GENAI_API_KEY", "AIzaSyDMSOShHaFllRyoWdyizKwfiDVyL8-1Rcg"))
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        full_prompt = f"{system_instruction}\n\nUser Question/Data:\n{user_input}\n\nProvide your debt avalanche analysis and recommendations:"
        
        response = model.generate_content(full_prompt, generation_config={"temperature": 0.5})
        return response.text
    except Exception as e:
        return f"Error: Unable to process debt advice request. {str(e)}"


# Create the financial agent class
class FinancialAgent:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=os.getenv("GENAI_API_KEY", "AIzaSyDMSOShHaFllRyoWdyizKwfiDVyL8-1Rcg"),
            temperature=0.1
        )

        self.tools = [
            grocery_recipe_generator,
            budget_plan_generator,
            receipt_category_classifier,
            google_maps_nearby_places,
            web_search_tool,
            debt_avalanche_advisor
        ]
        
        # Bind tools to the LLM
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        
        # Create tool node
        self.tool_node = ToolNode(self.tools)
        
        # Build the graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("agent", self._call_model)
        workflow.add_node("tools", self.tool_node)
        
        # Set entry point
        workflow.set_entry_point("agent")
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "agent",
            self._should_continue,
            {
                "continue": "tools",
                "end": END,
            },
        )
        
        # Add edge from tools back to agent
        workflow.add_edge("tools", "agent")
        
        return workflow.compile()
    
    def _call_model(self, state: AgentState) -> Dict[str, Any]:
        """Call the model with tools"""
        messages = state["messages"]
        
        # Add system message for context
        system_message = """You are a comprehensive financial assistant with access to specialized tools. 
        You can help with:
        - Recipe generation from grocery receipts
        - Budget planning and financial advice
        - Receipt categorization for expense tracking
        - Finding nearby places and services
        - Web search for financial information
        - Debt avalanche strategies and debt management
        
        Always use the appropriate tools when users ask for these services. 
        Be helpful, accurate, and provide detailed financial guidance."""
        
        if not any(isinstance(msg, AIMessage) and msg.content.startswith("You are a comprehensive") for msg in messages):
            messages = [AIMessage(content=system_message)] + messages
        
        response = self.llm_with_tools.invoke(messages)
        return {"messages": [response]}
    
    def _should_continue(self, state: AgentState) -> str:
        """Determine whether to continue or end the conversation"""
        messages = state["messages"]
        last_message = messages[-1]
        
        # If the last message has tool calls, continue to tools
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "continue"
        return "end"
    
    def chat(self, message: str, user_context: Optional[Dict[str, Any]] = None) -> str:
        """Chat with the financial agent"""
        initial_state = {
            "messages": [HumanMessage(content=message)],
            "user_context": user_context or {}
        }
        
        # Run the graph
        result = self.graph.invoke(initial_state)
        
        # Return the last AI message
        return result["messages"][-1].content
    
    def stream_chat(self, message: str, user_context: Optional[Dict[str, Any]] = None):
        """Stream chat responses"""
        initial_state = {
            "messages": [HumanMessage(content=message)],
            "user_context": user_context or {}
        }
        
        for chunk in self.graph.stream(initial_state):
            yield chunk


# Example usage and testing
def main():
    """Example usage of the Financial Agent"""
    
    # Initialize the agent
    agent = FinancialAgent()
    
    print("🏦 Financial Assistant initialized with LangGraph!")
    print("Available tools:")
    for tool in agent.tools:
        print(f"  - {tool.name}: {tool.description}")
        print()
    
    # Example interactions
    examples = [
        "I need help creating a budget. I'm 28 years old, earn 80,000-100,000 per month, live in Mumbai with 2 people in household, and want to save for buying a house.",
        
        "Can you generate a recipe from this grocery receipt: {'items': [{'name': 'Tomatoes', 'quantity': 2, 'price': 40}, {'name': 'Onions', 'quantity': 1, 'price': 30}, {'name': 'Rice', 'quantity': 1, 'price': 120}, {'name': 'Chicken', 'quantity': 0.5, 'price': 200}]}",
        
        "I have credit card debt of ₹200,000 at 24% interest and a personal loan of ₹150,000 at 18% interest. My monthly income is ₹60,000. Help me with a debt repayment strategy.",
        
        "Classify this receipt: {'merchant': 'Reliance Fresh', 'items': [{'name': 'Milk', 'price': 50}, {'name': 'Bread', 'price': 25}, {'name': 'Eggs', 'price': 80}], 'total': 155}",

        "Suggest restaurants that's an alternate for Truffles Indiranagar with a lower budget range"
    ]
    
    print("\n" + "="*60)
    print("EXAMPLE INTERACTIONS:")
    print("="*60)
    
    for i, example in enumerate(examples, 1):
        print(f"\n--- Example {i} ---")
        print(f"User: {example}")
        print("\nAssistant:")
        try:
            response = agent.chat(example)
            print(response)
        except Exception as e:
            print(f"Error: {str(e)}")
        print("-" * 40)


if __name__ == "__main__":
    main()