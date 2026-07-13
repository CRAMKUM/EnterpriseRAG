"""Main orchestrator agent using Gemini 3.1 Flash for intelligent routing."""

import json
from typing import Dict, Any, Optional
from utils.logger import get_logger
from utils.exceptions import ToolExecutionError
from config.config_loader import get_config_loader
from tools import OpenCVTool, TesseractTool, UnstructuredTool, GemmaTool

logger = get_logger(__name__)

class Orchestrator:
    """
    Main orchestrator that routes user requests to appropriate tools.
    Uses Gemini 3.1 Flash to analyze requests and determine the most
    cost-effective tool for each task.
    """

    def __init__(self):
        """Initialize orchestrator with Gemini and all tools."""
        self.config_loader = get_config_loader()
        self.logger = logger

        # Initialize Gemini for orchestration
        self._init_gemini()

        # Initialize all tools
        self._init_tools()

        self.logger.info("Orchestrator initialized")

    def _init_gemini(self):
        """Initialize Gemini 3.1 Flash for orchestration."""
        try:
            from google import genai
            from google.genai import types

            model_config = self.config_loader.get_model_config()
            graph_config = self.config_loader.get_graph_config()

            project_id = graph_config.get("project_id")
            location = graph_config.get("location", "us-central1")

            # Initialize Google GenAI client with Vertex AI
            self.client = genai.Client(
                vertexai=True,
                project=project_id,
                location=location
            )

            orchestrator_config = model_config.get("orchestrator", {})
            self.model_name = orchestrator_config.get("name", "gemini-1.5-flash")
            self.model_config = orchestrator_config
            self.types = types
            self.logger.info(f"Gemini orchestrator initialized: {self.model_name}")
        except Exception as e:
            self.logger.error(f"Failed to initialize Gemini: {e}")
            raise

    def _init_tools(self):
        """Initialize all available tools."""
        tool_config = self.config_loader.get_tool_config()

        self.tools = {
            "opencv": OpenCVTool(tool_config.get("opencv", {})),
            "tesseract": TesseractTool(tool_config.get("tesseract", {})),
            "unstructured_io": UnstructuredTool(tool_config.get("unstructured_io", {})),
            "gemma": GemmaTool(tool_config.get("gemma", {}))
        }
        self.logger.info(f"Initialized {len(self.tools)} tools")

    def route_request(self, user_request: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyze user request and route to appropriate tool.
        """
        try:
            self.logger.info(f"Routing request: {user_request[:100]}...")

            system_prompts = self.config_loader.get_system_prompts()
            system_instruction = system_prompts.get("orchestrator", "")

            routing_prompt = self._build_routing_prompt(user_request, context)

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=routing_prompt,
                config=self.types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=self.model_config.get("temperature", 0.7),
                    max_output_tokens=1024,
                )
            )

            routing_decision = self._parse_routing_response(response.text)
            self.logger.info(f"Routed to: {routing_decision['tool']}")
            return routing_decision
        except Exception as e:
            self.logger.error(f"Request routing failed: {e}")
            return self._fallback_routing(user_request, context)

    def execute_request(
        self,
        user_request: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Route and execute user request."""
        try:
            routing = self.route_request(user_request, context)
            tool_name = routing["tool"]
            parameters = routing.get("parameters", {})

            if tool_name == "graph_rag":
                return self._execute_graph_rag(user_request, parameters)
            elif tool_name in self.tools:
                tool = self.tools[tool_name]

                # Dynamically filter parameters to only pass those accepted by the tool's execute method
                import inspect
                sig = inspect.signature(tool.execute)
                valid_params = {
                    k: v for k, v in parameters.items()
                    if k in sig.parameters or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
                }

                result = tool.execute(**valid_params)

                return {
                    "tool_used": tool_name,
                    "routing_reasoning": routing.get("reasoning"),
                    "result": result,
                    "cost_tier": tool.cost_tier,
                    "success": True
                }
            else:
                raise ValueError(f"Unknown tool: {tool_name}")
        except Exception as e:
            self.logger.error(f"Request execution failed: {e}")
            raise ToolExecutionError(f"Execution failed: {e}")

    def _build_routing_prompt(self, user_request: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Build prompt for routing decision."""
        context_str = json.dumps(context, indent=2) if context else "None"
        current_doc_path = context.get("current_document_path") if isinstance(context, dict) else None

        doc_path_hint = ""
        if current_doc_path:
            doc_path_hint = f"\n\nCRITICAL DIRECTIVE: An active document is currently uploaded at '{current_doc_path}'. " \
                            f"If the selected tool requires a path (e.g. 'image_path' for 'tesseract'/'opencv'/'gemma', " \
                            f"or 'document_path' for 'unstructured_io'), you MUST supply '{current_doc_path}' as that parameter's exact value."

        return f"""Analyze this user request and route it to the most appropriate tool.

Available tools:
1. **opencv** - Deblur images (FREE)
2. **tesseract** - Extract text via OCR (FREE)
3. **unstructured_io** - Parse documents, extract tables (LOW cost)
4. **gemma** - Interpret images using small VLM (LOW cost)
5. **graph_rag** - Answer complex questions using knowledge graph (HIGH cost)

User Request: {user_request}
Context: {context_str}{doc_path_hint}

Respond in JSON format:
{{
    "tool": "tool_name",
    "reasoning": "why this tool was chosen",
    "parameters": {{"param1": "value1"}}
}}
"""

    def _parse_routing_response(self, response_text: str) -> Dict[str, Any]:
        """Parse Gemini's routing response."""
        try:
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            if start_idx == -1 or end_idx == 0:
                raise ValueError("No JSON found")

            json_str = response_text[start_idx:end_idx]
            routing = json.loads(json_str)
            if "tool" not in routing:
                raise ValueError("Missing 'tool' field")
            return routing
        except Exception as e:
            self.logger.error(f"Failed to parse routing response: {e}")
            raise

    def _fallback_routing(self, user_request: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Fallback keyword-based routing logic."""
        request_lower = user_request.lower()
        current_doc_path = context.get("current_document_path") if isinstance(context, dict) else None

        # Build fallback params
        img_params = {"image_path": current_doc_path} if current_doc_path else {}
        doc_params = {"document_path": current_doc_path} if current_doc_path else {}

        if "deblur" in request_lower or "blur" in request_lower:
            return {"tool": "opencv", "reasoning": "Fallback: blur", "parameters": img_params}
        elif "extract text" in request_lower or "ocr" in request_lower:
            return {"tool": "tesseract", "reasoning": "Fallback: ocr", "parameters": img_params}
        elif "table" in request_lower:
            return {"tool": "unstructured_io", "reasoning": "Fallback: table", "parameters": doc_params}
        elif "image" in request_lower or "picture" in request_lower:
            return {"tool": "gemma", "reasoning": "Fallback: image", "parameters": img_params}
        else:
            return {"tool": "graph_rag", "reasoning": "Fallback: default to Graph RAG", "parameters": {"question": user_request}}

    def _execute_graph_rag(self, query: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Placeholder for Graph RAG execution."""
        self.logger.info("Graph RAG execution requested")
        return {
            "tool_used": "graph_rag",
            "query": query,
            "result": {
                "message": "Graph RAG execution - implemented in graph module",
                "parameters": parameters
            },
            "cost_tier": "high",
            "success": True
        }

    def process_query(self, query: str, context: str = "", mmkg_available: bool = False) -> Dict[str, Any]:
        """Process query end-to-end."""
        try:
            ctx = {"conversation_context": context, "mmkg_available": mmkg_available}
            result = self.execute_request(query, context=ctx)

            tool_calls = []
            if result.get("tool_used"):
                tool_calls.append({
                    "tool": result["tool_used"],
                    "result": str(result.get("result", "")),
                    "reasoning": result.get("routing_reasoning", "")
                })

            response_text = result.get("result", {})
            if isinstance(response_text, dict):
                response_text = response_text.get("message", json.dumps(response_text, indent=2))

            return {
                "response": str(response_text),
                "tool_calls": tool_calls,
                "graph_query": result.get("result", {}).get("graph_query") if isinstance(result.get("result"), dict) else None
            }
        except Exception as e:
            self.logger.error(f"process_query failed: {e}")
            return {
                "response": f"I encountered an error processing your request: {str(e)}",
                "tool_calls": [],
                "graph_query": None
            }

    def get_tool_status(self) -> Dict[str, Any]:
        """Get status of all tools."""
        return {
            tool_name: {
                "name": tool.name,
                "cost_tier": tool.cost_tier,
                "available": True
            }
            for tool_name, tool in self.tools.items()
        }
