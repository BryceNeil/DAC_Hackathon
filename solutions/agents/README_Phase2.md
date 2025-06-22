# Phase 2: LangGraph Agent Architecture

## Overview

Phase 2 implements a sophisticated multi-agent system using LangGraph for the ASU Spec2Tapeout flow. This implementation follows LangGraph best practices including:

- **Plan-and-Execute Pattern**: For orchestrating the EDA flow
- **ReAct Agents**: For tool-calling and iterative problem solving
- **StateGraph**: For managing complex agent interactions
- **Human-in-the-Loop**: For critical design reviews

## Architecture

### Core Components

1. **Planning Agent** (`planning_agent.py`)
   - Creates execution plans from YAML specifications
   - Uses structured output with Pydantic models
   - Manages complexity assessment and tool selection

2. **RTL Generation Agent** (`rtl_generator.py`)
   - Uses ReAct pattern with tools
   - Validates syntax with Verilator
   - Generates testbenches
   - Compiles with Icarus Verilog

3. **Tapeout Graph** (`tapeout_graph.py`)
   - Main StateGraph orchestration
   - Conditional routing between agents
   - Error handling and recovery
   - Checkpointing for state persistence

4. **Human Validation** (`human_validation.py`)
   - Configurable review criteria
   - Structured review requests
   - Support for accept/edit/reject/skip actions

## Usage

### Basic Usage

```python
from agents import TapeoutGraph

# Initialize the graph
graph = TapeoutGraph(llm_model="gpt-4o")

# Run the flow
result = await graph.run(
    yaml_path="problems/example.yaml",
    thread_id="run_001"
)
```

### Command Line Interface

```bash
# Basic run
python run_langgraph_agent.py problems/example.yaml

# With human review
python run_langgraph_agent.py problems/example.yaml --human-review

# With custom model
python run_langgraph_agent.py problems/example.yaml --model gpt-4o-mini
```

## State Management

The system uses a typed state structure (`TapeoutState`) that includes:

- **Input/Output**: YAML path, problem specification, final response
- **Planning**: Execution plan with step tracking
- **Artifacts**: RTL code, SDC constraints, verification results
- **Metrics**: Timing, area, power, utilization
- **Error Tracking**: Accumulated errors for debugging

## Agent Communication

Agents communicate through state updates:

```python
# Agent returns state updates
return {
    "rtl_code": generated_rtl,
    "past_steps": [("rtl_generation", "Generated RTL successfully")],
    "errors": []  # Any errors encountered
}
```

## Tool Integration

### RTL Generation Tools

```python
@tool
def validate_rtl_syntax(rtl_code: str) -> str:
    """Validate RTL syntax using Verilator"""
    # Implementation details...

@tool
def compile_rtl(rtl_code: str, testbench_code: str) -> dict:
    """Compile RTL with Icarus Verilog"""
    # Implementation details...
```

### Human Interrupts

The system supports interrupts for human review:

```python
# Check if review needed
if human_validator.needs_rtl_review(state):
    review_request = human_validator.create_review_request(
        ReviewType.RTL_REVIEW, 
        state
    )
    # Trigger interrupt for human review
```

## Error Handling

The graph includes sophisticated error handling:

1. **Agent-level**: Each agent handles its own errors
2. **Graph-level**: Error handler node for recovery
3. **Retry Logic**: Automatic retry with backoff
4. **Graceful Degradation**: Falls back to simpler approaches

## Checkpointing

State is automatically checkpointed using LangGraph's MemorySaver:

```python
# Compile with checkpointing
app = graph.compile(checkpointer=MemorySaver())

# Resume from checkpoint
config = {"configurable": {"thread_id": "previous_run"}}
result = await app.ainvoke(state, config)
```

## Configuration

### Environment Variables

```bash
export OPENAI_API_KEY="your-api-key"
export LANGCHAIN_TRACING_V2="true"  # Optional: Enable tracing
export LANGCHAIN_PROJECT="asu-tapeout"  # Optional: Project name
```

### Human Validation Criteria

```python
criteria = ValidationCriteria(
    complexity_threshold="medium",
    timing_margin_threshold=0.1,  # ns
    area_threshold=5000.0,        # um²
    error_count_threshold=2
)
```

## Monitoring

The system provides detailed progress logging:

```
[planner] Created execution plan with 6 steps
[spec_analyzer] Analyzed specification successfully
[rtl_generator] Generated and validated RTL
[verification_agent] Verification passed
[constraint_generator] Generated SDC constraints
[physical_designer] Completed physical design
[validator] Final validation passed
```

## Extension Points

### Adding New Agents

1. Create agent class with async methods
2. Add to `TapeoutGraph.agents` dictionary
3. Add node and routing logic
4. Update state schema if needed

### Custom Tools

```python
@tool
def custom_analysis_tool(data: dict) -> str:
    """Custom tool description"""
    # Implementation
    return result
```

### Custom Review Criteria

Extend `ValidationCriteria` with new fields:

```python
class CustomCriteria(ValidationCriteria):
    custom_threshold: float = Field(default=1.0)
```

## Best Practices

1. **Async First**: All agent methods should be async
2. **State Updates**: Return only changed fields
3. **Error Messages**: Be descriptive for debugging
4. **Tool Descriptions**: Clear descriptions for LLM understanding
5. **Checkpointing**: Use thread IDs for resumability

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies installed
2. **API Key**: Check OPENAI_API_KEY is set
3. **Tool Failures**: Verify external tools (Verilator, Icarus) installed
4. **State Errors**: Check state schema matches updates

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Performance Optimization

1. **Parallel Tool Calls**: Tools run in parallel when possible
2. **Caching**: LLM responses cached by LangChain
3. **Streaming**: Results streamed for responsiveness
4. **Minimal State**: Only store necessary data

## Future Enhancements

- [ ] Multi-model support (Claude, Llama, etc.)
- [ ] Advanced replanning strategies
- [ ] Distributed execution
- [ ] Web UI for human reviews
- [ ] Integration with more EDA tools 