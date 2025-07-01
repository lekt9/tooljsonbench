from typing import Any, Dict, List, Optional, Tuple, Union
from pydantic import BaseModel


class ToolFunction(BaseModel):
    """Definition of a tool function."""
    name: str
    description: str
    parameters: Dict[str, Any]  # JSON schema for parameters


class Tool(BaseModel):
    """Tool definition for function calling."""
    type: str = "function"
    function: ToolFunction


class ResponseFormat(BaseModel):
    """Response format specification for structured outputs."""
    type: str  # "json_object" or "json_schema"
    json_schema: Optional[Dict[str, Any]] = None  # JSON schema when type is "json_schema"


class RequestConfig(BaseModel):
    """The configuration for a request to the LLM API.

    Args:
        model: The model to use.
        prompt: The prompt to provide to the LLM API.
        sampling_params: Additional sampling parameters to send with the request.
            For more information see the Router app's documentation for the completions
        llm_api: The name of the LLM API to send the request to.
        metadata: Additional metadata to attach to the request for logging or validation purposes.
        tools: List of tools available for the model to call.
        tool_choice: Controls which tool the model should use ("auto", "none", or specific tool).
        response_format: Specifies the format of the response (for structured outputs).
        test_type: Type of test being performed ("performance", "json_accuracy", "tool_calling").
    """

    model: str
    prompt: Tuple[Union[str, List[Dict[str, str]]], int]  # Support both string and message format
    sampling_params: Optional[Dict[str, Any]] = None
    llm_api: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[Union[str, Dict[str, str]]] = None
    response_format: Optional[Union[ResponseFormat, Dict[str, Any]]] = None
    test_type: Optional[str] = "performance"  # "performance", "json_accuracy", "tool_calling"


class BenchmarkResult(BaseModel):
    """Result of a benchmark test."""
    model: str
    test_type: str
    metrics: Dict[str, Any]
    generated_text: str
    request_config: RequestConfig
    accuracy_score: Optional[float] = None
    validation_errors: Optional[List[str]] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
