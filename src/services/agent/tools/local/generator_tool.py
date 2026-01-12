"""Generator tool for creating responses."""

import logging
from typing import Any

from src.services.agent.tools.base import BaseTool, ToolCategory, ToolMetadata

logger = logging.getLogger(__name__)


class GeneratorTool(BaseTool):
    """Tool for generating text responses."""
    
    def __init__(self, generator_client):
        """Initialize generator tool.
        
        Args:
            generator_client: Generator client instance
        """
        metadata = ToolMetadata(
            name="generator",
            description="Generate natural language responses based on context and query",
            category=ToolCategory.LOCAL,
            capabilities=[
                "text generation",
                "response creation",
                "question answering",
                "summarization",
            ],
            cost_per_call=0.0,  # Local model, no cost
            avg_latency_ms=500.0,
            success_rate=0.90,
            requires_api_key=False,
        )
        super().__init__(metadata)
        self.generator = generator_client
    
    async def execute(self, query: str, **kwargs: Any) -> dict[str, Any]:
        """Execute text generation.
        
        Args:
            query: User query or prompt
            **kwargs: Optional parameters:
                - context: Context documents to include
                - max_length: Maximum response length
                - temperature: Generation temperature
                
        Returns:
            Dictionary with generated response
        """
        try:
            if not self.validate_input(query, **kwargs):
                return {
                    "success": False,
                    "result": None,
                    "error": "Invalid input parameters",
                    "metadata": {},
                }
            
            # Extract parameters
            context = kwargs.get("context", "")
            max_length = kwargs.get("max_length", 512)
            temperature = kwargs.get("temperature", 0.7)
            
            self.logger.info(f"Generating response for query: {query[:100]}...")
            
            # Build prompt with context if provided
            if context:
                if isinstance(context, list):
                    # If context is a list of documents
                    context_str = "\n\n".join([
                        doc.get("content", str(doc)) if isinstance(doc, dict) else str(doc)
                        for doc in context[:3]  # Limit to top 3
                    ])
                else:
                    context_str = str(context)
                
                prompt = f"""Context:
{context_str}

Question: {query}

Answer:"""
            else:
                prompt = query
            
            # Generate response
            if hasattr(self.generator, "generate"):
                # Use generator's generate method
                response = self.generator.generate(
                    prompt=prompt,
                    max_length=max_length,
                    temperature=temperature,
                )
            elif callable(self.generator):
                # Generator is callable
                outputs = self.generator(
                    prompt,
                    max_length=max_length,
                    temperature=temperature,
                    do_sample=True,
                )
                # Extract text from output
                if isinstance(outputs, list) and len(outputs) > 0:
                    response = outputs[0].get("generated_text", prompt)
                    # Remove prompt from response
                    response = response[len(prompt):].strip()
                else:
                    response = str(outputs)
            else:
                raise AttributeError("Generator client is not callable or does not have generate method")
            
            self.logger.info(f"Generated response: {response[:100]}...")
            
            return {
                "success": True,
                "result": {
                    "response": response,
                    "query": query,
                    "has_context": bool(context),
                },
                "error": None,
                "metadata": {
                    "max_length": max_length,
                    "temperature": temperature,
                    "response_length": len(response),
                },
            }
            
        except Exception as e:
            self.logger.error(f"Generation failed: {e}")
            return {
                "success": False,
                "result": None,
                "error": str(e),
                "metadata": {},
            }
