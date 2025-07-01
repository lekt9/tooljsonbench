import json
import re
from typing import Any, Dict, List, Optional, Tuple
import jsonschema
from jsonschema import validate, ValidationError


def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """Extract JSON object from text, handling various formats."""
    # Try to parse the entire text as JSON first
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    
    # Look for JSON objects in the text using regex
    json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    matches = re.findall(json_pattern, text, re.DOTALL)
    
    for match in matches:
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue
    
    # Look for JSON arrays
    array_pattern = r'\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\]'
    matches = re.findall(array_pattern, text, re.DOTALL)
    
    for match in matches:
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue
    
    return None


def validate_json_response(response_text: str, expected_schema: Optional[Dict[str, Any]] = None) -> Tuple[bool, Optional[Dict[str, Any]], List[str]]:
    """
    Validate JSON response against schema.
    
    Returns:
        (is_valid_json, parsed_json, validation_errors)
    """
    errors = []
    
    # Extract JSON from response
    parsed_json = extract_json_from_text(response_text)
    
    if parsed_json is None:
        errors.append("No valid JSON found in response")
        return False, None, errors
    
    # Validate against schema if provided
    if expected_schema:
        try:
            validate(instance=parsed_json, schema=expected_schema)
        except ValidationError as e:
            errors.append(f"Schema validation error: {e.message}")
            return False, parsed_json, errors
        except Exception as e:
            errors.append(f"Schema validation error: {str(e)}")
            return False, parsed_json, errors
    
    return True, parsed_json, errors


def extract_tool_calls_from_text(text: str) -> List[Dict[str, Any]]:
    """Extract tool calls from response text."""
    tool_calls = []
    
    # Look for function call patterns
    # Pattern 1: {"function": "name", "arguments": {...}}
    function_pattern = r'\{"function":\s*"([^"]+)",\s*"arguments":\s*(\{[^}]*\})\}'
    matches = re.findall(function_pattern, text, re.DOTALL)
    
    for function_name, args_str in matches:
        try:
            arguments = json.loads(args_str)
            tool_calls.append({
                "function": function_name,
                "arguments": arguments
            })
        except json.JSONDecodeError:
            continue
    
    # Pattern 2: function_name(arg1=value1, arg2=value2)
    call_pattern = r'(\w+)\(([^)]+)\)'
    matches = re.findall(call_pattern, text)
    
    for function_name, args_str in matches:
        try:
            # Parse arguments
            arguments = {}
            arg_pairs = args_str.split(',')
            for pair in arg_pairs:
                if '=' in pair:
                    key, value = pair.split('=', 1)
                    key = key.strip()
                    value = value.strip().strip('"\'')
                    arguments[key] = value
            
            tool_calls.append({
                "function": function_name,
                "arguments": arguments
            })
        except Exception:
            continue
    
    return tool_calls


def validate_tool_calls(response_text: str, expected_tools: List[Dict[str, Any]]) -> Tuple[bool, List[Dict[str, Any]], List[str]]:
    """
    Validate tool calls in response.
    
    Returns:
        (has_valid_tool_calls, extracted_tool_calls, validation_errors)
    """
    errors = []
    tool_calls = extract_tool_calls_from_text(response_text)
    
    if not tool_calls:
        errors.append("No tool calls found in response")
        return False, [], errors
    
    # Validate each tool call
    expected_tool_names = {tool["function"]["name"] for tool in expected_tools}
    
    for tool_call in tool_calls:
        function_name = tool_call.get("function")
        if function_name not in expected_tool_names:
            errors.append(f"Unknown function called: {function_name}")
            continue
        
        # Find the expected tool definition
        expected_tool = next(
            (tool for tool in expected_tools if tool["function"]["name"] == function_name),
            None
        )
        
        if expected_tool:
            # Validate arguments against schema
            expected_params = expected_tool["function"].get("parameters", {})
            if expected_params and "properties" in expected_params:
                try:
                    validate(instance=tool_call.get("arguments", {}), schema=expected_params)
                except ValidationError as e:
                    errors.append(f"Tool call argument validation error for {function_name}: {e.message}")
    
    return len(errors) == 0, tool_calls, errors


def calculate_accuracy_score(expected: Any, actual: Any, test_type: str = "exact_match") -> float:
    """Calculate accuracy score based on test type."""
    if test_type == "exact_match":
        return 1.0 if expected == actual else 0.0
    
    elif test_type == "json_structure":
        if not isinstance(expected, dict) or not isinstance(actual, dict):
            return 0.0
        
        # Check if all expected keys are present
        expected_keys = set(expected.keys())
        actual_keys = set(actual.keys())
        
        if expected_keys.issubset(actual_keys):
            return 1.0
        else:
            # Partial credit based on key overlap
            overlap = len(expected_keys.intersection(actual_keys))
            return overlap / len(expected_keys) if expected_keys else 0.0
    
    elif test_type == "tool_calling":
        if not isinstance(expected, list) or not isinstance(actual, list):
            return 0.0
        
        if len(expected) != len(actual):
            return 0.0
        
        # Check if tool calls match
        matches = 0
        for exp_call, act_call in zip(expected, actual):
            if (exp_call.get("function") == act_call.get("function") and
                exp_call.get("arguments") == act_call.get("arguments")):
                matches += 1
        
        return matches / len(expected) if expected else 0.0
    
    return 0.0


def validate_response_format(response_text: str, expected_format: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate response format compliance."""
    errors = []
    format_type = expected_format.get("type", "")
    
    if format_type == "json_object":
        parsed_json = extract_json_from_text(response_text)
        if parsed_json is None:
            errors.append("Response is not valid JSON")
            return False, errors
    
    elif format_type == "json_schema":
        schema = expected_format.get("json_schema") or expected_format.get("schema")
        if schema:
            is_valid, _, validation_errors = validate_json_response(response_text, schema)
            if not is_valid:
                errors.extend(validation_errors)
                return False, errors
    
    return len(errors) == 0, errors
