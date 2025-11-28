"""
MCP (Model Context Protocol) Library
Provides tools that can be called by the LLM.
"""
import json


def get_weather(city: str) -> str:
    """
    Get the weather for a city.
    """
    # TODO: Implement actual weather API call
    return f"{city}의 날씨는 맑고, 기온은 22도입니다."


def get_current_time() -> str:
    """
    Get the current time.
    """
    from datetime import datetime
    now = datetime.now()
    return f"현재 시간은 {now.strftime('%Y년 %m월 %d일 %H시 %M분')}입니다."


def calculate(expression: str) -> str:
    """
    Calculate a mathematical expression.
    """
    try:
        # Safe eval for basic math
        allowed_chars = set('0123456789+-*/.() ')
        if not all(c in allowed_chars for c in expression):
            return "잘못된 수식입니다."
        result = eval(expression)
        return f"계산 결과: {expression} = {result}"
    except Exception as e:
        return f"계산 오류: {str(e)}"


# Tool definitions in Ollama format
TOOLS = [
    {
        'type': 'function',
        'function': {
            'name': 'get_weather',
            'description': 'Get the current weather for a specific city. Use this when user asks about weather.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'city': {
                        'type': 'string',
                        'description': 'The name of the city (e.g., Seoul, Tokyo, New York)'
                    }
                },
                'required': ['city']
            }
        }
    },
    {
        'type': 'function',
        'function': {
            'name': 'get_current_time',
            'description': 'Get the current date and time. Use this when user asks about current time or date.',
            'parameters': {
                'type': 'object',
                'properties': {},
                'required': []
            }
        }
    },
    {
        'type': 'function',
        'function': {
            'name': 'calculate',
            'description': 'Calculate a mathematical expression. Use this when user asks to calculate something.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'expression': {
                        'type': 'string',
                        'description': 'The mathematical expression to calculate (e.g., "2+2", "10*5", "100/4")'
                    }
                },
                'required': ['expression']
            }
        }
    }
]

# Function registry for execution
TOOL_FUNCTIONS = {
    'get_weather': get_weather,
    'get_current_time': get_current_time,
    'calculate': calculate,
}


def get_tools():
    """Returns the list of available tools in Ollama format."""
    return TOOLS


def get_tool_functions():
    """Returns the function registry."""
    return TOOL_FUNCTIONS


def execute_tool(tool_name: str, arguments: dict) -> str:
    """
    Execute a tool by name with given arguments.
    
    Args:
        tool_name: Name of the tool to execute
        arguments: Dictionary of arguments for the tool
        
    Returns:
        Result of the tool execution as string
    """
    if tool_name not in TOOL_FUNCTIONS:
        return f"Unknown tool: {tool_name}"
    
    try:
        func = TOOL_FUNCTIONS[tool_name]
        result = func(**arguments)
        return result
    except Exception as e:
        return f"Tool execution error: {str(e)}"
