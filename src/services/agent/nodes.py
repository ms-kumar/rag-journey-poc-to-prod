"""Agent nodes for LangGraph workflow."""

import logging
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from src.services.agent.benchmarking import TaskBenchmarker
from src.services.agent.planning import QueryPlanner
from src.services.agent.reflection import AnswerCritic, SourceVerifier
from src.services.agent.state import AgentState
from src.services.agent.tools.registry import ToolRegistry
from src.services.agent.tools.router import AgentRouter

logger = logging.getLogger(__name__)


class AgentNodes:
    """Agent nodes for LangGraph state machine."""

    def __init__(
        self,
        registry: ToolRegistry,
        router: AgentRouter,
        llm: Any = None,
        enable_reflection: bool = True,
        enable_planning: bool = True,
        enable_benchmarking: bool = False,
    ):
        """Initialize agent nodes.

        Args:
            registry: Tool registry instance
            router: Agent router instance
            llm: Optional LLM for planning/reflection
            enable_reflection: Enable answer critique and source verification
            enable_planning: Enable advanced query planning
            enable_benchmarking: Enable task benchmarking
        """
        self.registry = registry
        self.router = router
        self.llm = llm
        self.logger = logging.getLogger(__name__)

        # Initialize reflection components
        self.enable_reflection = enable_reflection
        self.answer_critic: AnswerCritic | None = None
        self.source_verifier: SourceVerifier | None = None
        if enable_reflection:
            self.answer_critic = AnswerCritic(llm=llm)
            self.source_verifier = SourceVerifier()

        # Initialize planning component
        self.enable_planning = enable_planning
        self.query_planner: QueryPlanner | None = None
        if enable_planning:
            self.query_planner = QueryPlanner(llm=llm)

        # Initialize benchmarking
        self.enable_benchmarking = enable_benchmarking
        self.benchmarker: TaskBenchmarker | None = None
        if enable_benchmarking:
            self.benchmarker = TaskBenchmarker(storage_path="logs/agent_benchmarks.jsonl")

    async def plan_node(self, state: AgentState) -> dict:
        """Plan node: Decompose query into subtasks.

        Args:
            state: Current agent state

        Returns:
            Updated state with plan
        """
        self.logger.info("=== PLAN NODE ===")
        query = state["query"]

        # Use advanced planner if enabled
        if self.enable_planning and self.query_planner:
            try:
                execution_plan = self.query_planner.create_plan(query)
                plan = [task.description for task in execution_plan.tasks]

                self.logger.info(
                    f"Advanced planning: {len(plan)} tasks, "
                    f"complexity={execution_plan.complexity_level}, "
                    f"strategy={execution_plan.execution_strategy}"
                )
                self.logger.info(f"Plan rationale: {execution_plan.rationale}")

                # Start benchmark if enabled
                if self.enable_benchmarking and self.benchmarker:
                    benchmark = self.benchmarker.start_benchmark(
                        query=query,
                        plan_complexity=execution_plan.complexity_level,
                        num_tasks=len(plan),
                    )
                    state["_benchmark"] = benchmark

            except Exception as e:
                self.logger.warning(f"Advanced planning failed: {e}, using simple plan")
                plan = self._simple_plan(query)
        else:
            # Simple planning
            plan = self._simple_plan(query)

        self.logger.info(f"Created plan with {len(plan)} tasks: {plan}")

        return {
            "plan": plan,
            "current_task": plan[0] if plan else query,
            "iteration_count": state.get("iteration_count", 0) + 1,
            "messages": state.get("messages", [])
            + [
                HumanMessage(content=query),
                AIMessage(content=f"Plan created: {len(plan)} tasks"),
            ],
        }

    def _simple_plan(self, query: str) -> list[str]:
        """Create a simple plan without advanced planning.

        Args:
            query: Query text

        Returns:
            List of task descriptions
        """
        # Simple planning: Check if query needs decomposition
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
                if plan:
                    return plan
            except Exception as e:
                self.logger.warning(f"LLM planning failed: {e}")

        # Default: single task
        return [query]

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
        tool_history.append(
            {
                "task": current_task,
                "tool": decision.tool_name,
                "confidence": decision.confidence,
                "reasoning": decision.reasoning,
                "status": "pending",
            }
        )

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
        import time

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

            # Record in benchmark
            if self.enable_benchmarking and self.benchmarker and "_benchmark" in state:
                self.benchmarker.record_task(
                    benchmark=state["_benchmark"],
                    task_id=f"task_{len(tool_history)}",
                    task_description=current_task,
                    execution_time=0.0,
                    success=False,
                    tool_used=tool_name,
                    error_message=f"Tool '{tool_name}' not found",
                )

            return {
                "tool_history": tool_history,
                "needs_replanning": True,
            }

        # Execute tool
        start_time = time.time()
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
            execution_time = time.time() - start_time

            # Update history
            tool_history[-1]["status"] = "success" if result["success"] else "failed"
            tool_history[-1]["result"] = result
            tool_history[-1]["execution_time"] = execution_time

            # Store result
            results = state.get("results", [])
            results.append(result)

            # Record in benchmark
            if self.enable_benchmarking and self.benchmarker and "_benchmark" in state:
                self.benchmarker.record_task(
                    benchmark=state["_benchmark"],
                    task_id=f"task_{len(tool_history)}",
                    task_description=current_task,
                    execution_time=execution_time,
                    success=result["success"],
                    tool_used=tool_name,
                    tool_confidence=last_decision.get("confidence", 0.0),
                )

            self.logger.info(
                f"Tool execution {'succeeded' if result['success'] else 'failed'} "
                f"in {execution_time:.2f}s"
            )

            return {
                "tool_history": tool_history,
                "results": results,
            }

        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Tool execution failed: {e}")
            tool_history[-1]["status"] = "failed"
            tool_history[-1]["error"] = str(e)
            tool_history[-1]["execution_time"] = execution_time

            # Record in benchmark
            if self.enable_benchmarking and self.benchmarker and "_benchmark" in state:
                self.benchmarker.record_task(
                    benchmark=state["_benchmark"],
                    task_id=f"task_{len(tool_history)}",
                    task_description=current_task,
                    execution_time=execution_time,
                    success=False,
                    tool_used=tool_name,
                    error_message=str(e),
                )

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
            final_answer = self._synthesize_answer(state)

            # Critique answer if reflection enabled
            critique_result = {}
            if self.enable_reflection and self.answer_critic:
                critique_result = self._critique_and_verify(
                    answer=final_answer,
                    query=state["query"],
                    state=state,
                )

            # Complete benchmark
            if self.enable_benchmarking and self.benchmarker and "_benchmark" in state:
                self.benchmarker.complete_benchmark(
                    benchmark=state["_benchmark"],
                    final_quality_score=critique_result.get("quality_score", 0.0),
                )

            return {
                "final_answer": final_answer,
                "needs_replanning": False,
                **critique_result,
            }

        if completed_tasks >= len(plan):
            # All tasks completed - perform reflection
            self.logger.info("All tasks completed successfully")
            final_answer = self._synthesize_answer(state)

            # Critique answer if reflection enabled
            critique_result = {}
            if self.enable_reflection and self.answer_critic:
                critique_result = self._critique_and_verify(
                    answer=final_answer,
                    query=state["query"],
                    state=state,
                )

                # Check if answer needs improvement
                if critique_result.get("needs_revision", False):
                    self.logger.warning(
                        "Answer critique suggests revision needed: "
                        f"{critique_result.get('issues', [])}"
                    )

                    # If we haven't exceeded max iterations, consider replanning
                    if iteration_count < max_iterations - 1:
                        return {
                            "needs_replanning": True,
                            "current_task": f"Improve answer quality: {state['query']}",
                            **critique_result,
                        }

            # Complete benchmark
            if self.enable_benchmarking and self.benchmarker and "_benchmark" in state:
                self.benchmarker.complete_benchmark(
                    benchmark=state["_benchmark"],
                    final_quality_score=critique_result.get("quality_score", 0.0),
                )

            return {
                "final_answer": final_answer,
                "needs_replanning": False,
                **critique_result,
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
        final_answer = self._synthesize_answer(state)
        return {
            "final_answer": final_answer,
            "needs_replanning": False,
        }

    def _critique_and_verify(self, answer: str, query: str, state: AgentState) -> dict[str, Any]:
        """Critique answer and verify sources.

        Args:
            answer: The generated answer
            query: Original query
            state: Agent state

        Returns:
            Dictionary with critique results
        """
        results = state.get("results", [])
        tool_history = state.get("tool_history", [])

        # Extract sources from results
        sources = []
        for result in results:
            if result.get("success"):
                result_data = result.get("result", {})
                if "documents" in result_data:
                    sources.extend(result_data["documents"])

        # Return empty result if reflection components not available
        if not self.answer_critic or not self.source_verifier:
            return {}

        # Critique answer
        critique = self.answer_critic.critique_answer(
            answer=answer,
            query=query,
            sources=sources,
            tool_history=tool_history,
        )

        # Verify sources
        verification = self.source_verifier.verify_sources(
            answer=answer,
            sources=sources,
            tool_history=tool_history,
        )

        self.logger.info(
            f"Answer critique: quality={critique.overall_score:.2f}, "
            f"needs_revision={critique.needs_revision}"
        )
        self.logger.info(
            f"Source verification: {verification.sources_verified}/"
            f"{verification.sources_found} verified, "
            f"hallucination_risk={verification.hallucination_risk:.2f}"
        )

        return {
            "quality_score": critique.overall_score,
            "completeness_score": critique.completeness_score,
            "accuracy_score": critique.accuracy_score,
            "relevance_score": critique.relevance_score,
            "clarity_score": critique.clarity_score,
            "source_quality_score": critique.source_quality_score,
            "needs_revision": critique.needs_revision,
            "issues": critique.issues,
            "suggestions": critique.suggestions,
            "missing_aspects": critique.missing_aspects,
            "sources_verified": verification.sources_verified,
            "hallucination_risk": verification.hallucination_risk,
            "questionable_claims": verification.questionable_claims,
        }

    def _synthesize_answer(self, state: AgentState) -> str:
        """Synthesize final answer from results.

        Args:
            state: Current agent state

        Returns:
            Final answer string
        """
        results = state.get("results", [])

        if not results:
            return "I apologize, but I couldn't find any information to answer your query."

        # Combine successful results
        answer_parts = []
        for _i, result in enumerate(results, 1):
            if not result.get("success"):
                continue

            result_data = result.get("result", {})

            # Extract content based on result type
            if "documents" in result_data:
                # Retrieval/reranking result
                docs = result_data["documents"][:3]
                content = "\n\n".join([doc.get("content", "") for doc in docs])
                answer_parts.append(content)
            elif "response" in result_data:
                # Generator result
                answer_parts.append(result_data["response"])
            elif "results" in result_data:
                # Web search/Wikipedia result
                search_results = result_data["results"][:3]
                content = "\n\n".join(
                    [
                        f"{r.get('title', '')}: {r.get('content', r.get('summary', ''))}"
                        for r in search_results
                    ]
                )
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
