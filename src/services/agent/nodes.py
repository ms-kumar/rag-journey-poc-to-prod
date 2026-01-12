"""Agent nodes for LangGraph workflow."""

import logging
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from src.services.agent.state import AgentState
from src.services.agent.tools.registry import ToolRegistry
from src.services.agent.tools.router import AgentRouter

logger = logging.getLogger(__name__)


class AgentNodes:
    """Agent nodes for LangGraph state machine."""
    
    def __init__(self, registry: ToolRegistry, router: AgentRouter, llm: Any = None):
        """Initialize agent nodes.
        
        Args:
            registry: Tool registry instance
            router: Agent router instance
            llm: Optional LLM for planning/reflection
        """
        self.registry = registry
        self.router = router
        self.llm = llm
        self.logger = logging.getLogger(__name__)
    
    async def plan_node(self, state: AgentState) -> dict:
        """Plan node: Decompose query into subtasks.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with plan
        """
        self.logger.info("=== PLAN NODE ===")
        query = state["query"]
        
        # Simple planning: Check if query needs decomposition
        # For complex queries, we could use an LLM here
        keywords = ["and", "also", "then", "after", "plus"]
        needs_decomposition = any(kw in query.lower() for kw in keywords)
        
        if needs_decomposition and self.llm:
            # Use LLM for planning (optional enhancement)
            try:
                plan_prompt = f"""Break down this complex query into simple subtasks:
Query: {query}

Provide 2-4 simple, actionable subtasks. Format as a numbered list."""
                
                response = self.llm.predict(plan_prompt)
                # Parse numbered list
                plan = [
                    line.strip().split(". ", 1)[1] if ". " in line else line.strip()
                    for line in response.split("\n")
                    if line.strip() and any(c.isdigit() for c in line[:3])
                ]
            except Exception as e:
                self.logger.warning(f"LLM planning failed: {e}, using simple plan")
                plan = [query]
        else:
            # Simple plan: treat entire query as one task
            plan = [query]
        
        self.logger.info(f"Created plan with {len(plan)} tasks: {plan}")
        
        return {
            "plan": plan,
            "current_task": plan[0] if plan else query,
            "iteration_count": state.get("iteration_count", 0) + 1,
            "messages": state.get("messages", []) + [
                HumanMessage(content=query),
                AIMessage(content=f"Plan created: {len(plan)} tasks"),
            ],
        }
    
    async def route_node(self, state: AgentState) -> dict:
        """Route node: Select appropriate tool.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with routing decision
        """
        self.logger.info("=== ROUTE NODE ===")
        current_task = state["current_task"]
        
        # Route to appropriate tool
        decision = await self.router.route(current_task)
        
        self.logger.info(
            f"Routed to '{decision.tool_name}' with confidence {decision.confidence:.2f}"
        )
        
        # Store routing decision
        tool_history = state.get("tool_history", [])
        tool_history.append({
            "task": current_task,
            "tool": decision.tool_name,
            "confidence": decision.confidence,
            "reasoning": decision.reasoning,
            "status": "pending",
        })
        
        return {
            "tool_history": tool_history,
            "confidence": decision.confidence,
        }
    
    async def execute_node(self, state: AgentState) -> dict:
        """Execute node: Execute selected tool.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with execution results
        """
        self.logger.info("=== EXECUTE NODE ===")
        
        # Get last routing decision
        tool_history = state["tool_history"]
        if not tool_history:
            raise ValueError("No tool selected for execution")
        
        last_decision = tool_history[-1]
        tool_name = last_decision["tool"]
        current_task = last_decision["task"]
        
        # Get tool from registry
        tool = self.registry.get_tool(tool_name)
        if not tool:
            self.logger.error(f"Tool '{tool_name}' not found in registry")
            # Update history with error
            tool_history[-1]["status"] = "failed"
            tool_history[-1]["error"] = f"Tool '{tool_name}' not found"
            return {
                "tool_history": tool_history,
                "needs_replanning": True,
            }
        
        # Execute tool
        try:
            self.logger.info(f"Executing tool '{tool_name}'...")
            
            # Pass context from previous results if available
            results = state.get("results", [])
            context = {"previous_results": results} if results else {}
            
            # For certain tools, pass specific parameters
            if tool_name == "reranker" and results:
                # Pass documents from previous retrieval
                last_result = results[-1] if results else {}
                if "result" in last_result and "documents" in last_result["result"]:
                    context["documents"] = last_result["result"]["documents"]
            elif tool_name == "generator" and results:
                # Pass documents as context for generation
                last_result = results[-1] if results else {}
                if "result" in last_result:
                    if "documents" in last_result["result"]:
                        context["context"] = last_result["result"]["documents"]
                    elif "results" in last_result["result"]:
                        # From web search or wikipedia
                        context["context"] = last_result["result"]["results"]
            
            result = await tool.execute(current_task, **context)
            
            # Update history
            tool_history[-1]["status"] = "success" if result["success"] else "failed"
            tool_history[-1]["result"] = result
            
            # Store result
            results = state.get("results", [])
            results.append(result)
            
            self.logger.info(
                f"Tool execution {'succeeded' if result['success'] else 'failed'}"
            )
            
            return {
                "tool_history": tool_history,
                "results": results,
            }
            
        except Exception as e:
            self.logger.error(f"Tool execution failed: {e}")
            tool_history[-1]["status"] = "failed"
            tool_history[-1]["error"] = str(e)
            return {
                "tool_history": tool_history,
                "needs_replanning": True,
            }
    
    async def reflect_node(self, state: AgentState) -> dict:
        """Reflect node: Evaluate results and decide next steps.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with reflection decision
        """
        self.logger.info("=== REFLECT NODE ===")
        
        plan = state["plan"]
        results = state.get("results", [])
        tool_history = state.get("tool_history", [])
        iteration_count = state.get("iteration_count", 0)
        max_iterations = state.get("max_iterations", 5)
        
        # Check if all tasks completed
        completed_tasks = len([h for h in tool_history if h["status"] == "success"])
        
        self.logger.info(
            f"Reflection: {completed_tasks}/{len(plan)} tasks completed, "
            f"iteration {iteration_count}/{max_iterations}"
        )
        
        # Check stopping conditions
        if iteration_count >= max_iterations:
            self.logger.warning("Max iterations reached")
            return {
                "final_answer": self._synthesize_answer(state),
                "needs_replanning": False,
            }
        
        if completed_tasks >= len(plan):
            # All tasks completed
            self.logger.info("All tasks completed successfully")
            return {
                "final_answer": self._synthesize_answer(state),
                "needs_replanning": False,
            }
        
        # Check if last tool failed
        if tool_history and tool_history[-1]["status"] == "failed":
            self.logger.warning("Last tool failed, needs replanning")
            return {
                "needs_replanning": True,
            }
        
        # Move to next task
        next_task_idx = completed_tasks
        if next_task_idx < len(plan):
            next_task = plan[next_task_idx]
            self.logger.info(f"Moving to next task: {next_task}")
            return {
                "current_task": next_task,
                "needs_replanning": False,
            }
        
        # Default: finish
        return {
            "final_answer": self._synthesize_answer(state),
            "needs_replanning": False,
        }
    
    def _synthesize_answer(self, state: AgentState) -> str:
        """Synthesize final answer from results.
        
        Args:
            state: Current agent state
            
        Returns:
            Final answer string
        """
        results = state.get("results", [])
        query = state["query"]
        
        if not results:
            return "I apologize, but I couldn't find any relevant information to answer your question."
        
        # Combine successful results
        answer_parts = []
        for i, result in enumerate(results, 1):
            if not result.get("success"):
                continue
            
            result_data = result.get("result", {})
            
            # Extract content based on result type
            if "documents" in result_data:
                # Retrieval/reranking result
                docs = result_data["documents"][:3]
                content = "\n\n".join([
                    doc.get("content", "") for doc in docs
                ])
                answer_parts.append(content)
            elif "response" in result_data:
                # Generator result
                answer_parts.append(result_data["response"])
            elif "results" in result_data:
                # Web search/Wikipedia result
                search_results = result_data["results"][:3]
                content = "\n\n".join([
                    f"{r.get('title', '')}: {r.get('content', r.get('summary', ''))}"
                    for r in search_results
                ])
                answer_parts.append(content)
            elif "output" in result_data:
                # Code execution result
                answer_parts.append(f"Execution output:\n{result_data['output']}")
        
        if not answer_parts:
            return "I found some information but couldn't format a proper response."
        
        # Combine all parts
        final_answer = "\n\n".join(answer_parts)
        
        # Truncate if too long
        if len(final_answer) > 2000:
            final_answer = final_answer[:2000] + "..."
        
        return final_answer
