# ASU Spec2Tapeout Hackathon Solution - Grading Guide

## 📋 Solution Overview

This solution implements an automated RTL-to-GDSII flow using a multi-agent system powered by LangGraph. The system takes hardware specifications in YAML format and produces synthesizable RTL, timing constraints (SDC), and physical design layouts (ODB).

## 🏗️ Architecture Diagram

```ascii
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ASU Spec2Tapeout LangGraph Agent                     │
│                                                                             │
│  ┌──────────────┐                                                          │
│  │ YAML Spec    │──┐                                                       │
│  │ (p1.yaml)    │  │                                                       │
│  └──────────────┘  │                                                       │
│                    ▼                                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                      LangGraph Orchestrator                          │  │
│  │  ┌────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────────┐ │  │
│  │  │ Process    │  │   Planner   │  │    Spec     │  │   Error    │ │  │
│  │  │   Input    │──│   Agent     │──│  Analyzer   │──│  Handler   │ │  │
│  │  └────────────┘  └─────────────┘  └─────────────┘  └────────────┘ │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                    │                                                       │
│                    ▼                                                       │
│  ╔═══════════════════════════════════════════════════════════════════╗    │
│  ║                    Specialized Execution Agents                    ║    │
│  ╠═══════════════════════════════════════════════════════════════════╣    │
│  ║                                                                   ║    │
│  ║  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐    ║    │
│  ║  │     RTL      │     │ Verification │     │  Constraint  │    ║    │
│  ║  │  Generator   │────▶│    Agent     │────▶│  Generator   │    ║    │
│  ║  └──────────────┘     └──────────────┘     └──────────────┘    ║    │
│  ║         │                     │                     │            ║    │
│  ║         ▼                     ▼                     ▼            ║    │
│  ║    [module.v]          [test results]         [6_final.sdc]     ║    │
│  ║                                                     │            ║    │
│  ║  ┌──────────────────────────────────────────────────┘           ║    │
│  ║  │                                                               ║    │
│  ║  ▼                                                               ║    │
│  ║  ┌──────────────┐     ┌──────────────┐                         ║    │
│  ║  │  Physical    │────▶│  Validator   │                         ║    │
│  ║  │  Designer    │     │    Agent     │                         ║    │
│  ║  └──────────────┘     └──────────────┘                         ║    │
│  ║         │                     │                                  ║    │
│  ║         ▼                     ▼                                  ║    │
│  ║    [6_final.odb]       [validation]                            ║    │
│  ║                                                                   ║    │
│  ╚═══════════════════════════════════════════════════════════════════╝    │
│                                                                             │
│  Output Files:                                                              │
│  ├── p1.v          (Synthesizable RTL)                                     │
│  ├── 6_final.sdc   (Timing Constraints)                                    │
│  └── 6_final.odb   (Physical Layout Database)                              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

                        Execution Modes Available:
                 ┌────────────┬─────────────┬────────────────┐
                 │ Autonomous │ Human-Loop  │ Human-Approval │
                 └────────────┴─────────────┴────────────────┘
```

## 🔧 How It Works

### 1. **Input Processing**
   - Reads YAML specification files containing hardware module descriptions
   - Extracts module parameters, interface signals, and functional requirements

### 2. **Planning Phase**
   - Creates an execution plan using the Planner Agent
   - Determines which specialized agents to invoke and in what order
   - Adapts strategy based on module complexity

### 3. **RTL Generation**
   - RTL Generator Agent creates Verilog code from specifications
   - Supports various design patterns (FSMs, arithmetic units, DSP blocks)
   - Uses templates and LLM-guided synthesis

### 4. **Verification**
   - Verification Agent runs functional tests on generated RTL
   - Uses Verilator for simulation
   - Validates against specification requirements

### 5. **Constraint Generation**
   - Creates timing constraints (SDC format)
   - Handles clock definitions, I/O delays, and timing exceptions
   - Ensures synthesizability

### 6. **Physical Design**
   - Physical Designer Agent runs OpenROAD flow
   - Performs synthesis, placement, and routing
   - Generates final layout database (ODB)

### 7. **Validation**
   - Final validation of all outputs
   - Checks for completeness and correctness
   - Error handling and recovery mechanisms

## 🚀 Running Instructions

### Prerequisites

1. **Docker Environment**
   ```bash
   docker run -it --rm \
     -v ~/iclad_hackathon:/workspace/iclad_hackathon \
     iclad_hackathon:latest bash
   ```

2. **Navigate to Solutions Directory**
   ```bash
   cd /workspace/iclad_hackathon/ICLAD-Hackathon-2025/problem-categories/ASU-Spec2Tapeout-ICLAD25-Hackathon/solutions
   ```

3. **Install Dependencies**
   ```bash
   python3 -m pip install -r requirements.txt
   ```

### Running the Agent

#### Basic Autonomous Mode (Single Problem)
```bash
python3 your_agent_langgraph.py --problem ../problems/visible/p1.yaml --output_dir ./visible/p1/
```

#### Process All Visible Problems
```bash
python3 your_agent_langgraph.py --problem_dir ../problems/visible/ --output_base ./visible/
```

#### Human-in-the-Loop Mode
```bash
python3 your_agent_langgraph.py --mode human_in_loop --problem ../problems/visible/p1.yaml --output_dir ./visible/p1/
```

#### Quiet Mode (Minimal Output)
```bash
python3 your_agent_langgraph.py --quiet --problem_dir ../problems/visible/ --output_base ./visible/
```

#### With Custom API Key
```bash
python3 your_agent_langgraph.py --llm_key YOUR_OPENAI_KEY --problem ../problems/visible/p1.yaml --output_dir ./visible/p1/
```

#### Using Claude Model
```bash
python3 your_agent_langgraph.py --model claude-3-5-sonnet-20241022 --llm_key YOUR_ANTHROPIC_KEY --problem ../problems/visible/p1.yaml --output_dir ./visible/p1/
```

### Available Options

| Option | Description | Default |
|--------|-------------|---------|
| `--problem` | Single YAML problem file to solve | - |
| `--problem_dir` | Directory containing problem YAML files | - |
| `--output_dir` | Output directory for single problem | - |
| `--output_base` | Base output directory for multiple problems | - |
| `--llm_key` | API key (OpenAI or Anthropic) | From .env |
| `--model` | LLM model to use | claude-sonnet-4-20250514 |
| `--mode` | Execution mode: autonomous, human_in_loop, human_approval | autonomous |
| `--quiet` | Suppress verbose output | False |

## 📊 Expected Outputs

For each problem, the solution generates:

1. **RTL File** (`problemName.v`)
   - Synthesizable Verilog code
   - Implements specified functionality
   - Follows coding standards

2. **SDC File** (`6_final.sdc`)
   - Timing constraints
   - Clock definitions
   - I/O delays and exceptions

3. **ODB File** (`6_final.odb`)
   - OpenROAD Database format
   - Contains placed and routed design
   - Physical layout information

## 🎯 Key Features

- **Multi-Agent Architecture**: Specialized agents for each design phase
- **LangGraph Orchestration**: Robust state management and error recovery
- **Template Library**: Pre-built RTL templates for common patterns
- **Adaptive Planning**: Dynamic execution based on problem complexity
- **Human-in-the-Loop**: Optional human intervention points
- **Progress Tracking**: Real-time status updates during execution

## 📈 Performance Metrics

The solution tracks:
- Execution time per agent
- Success/failure rates
- Area, power, and timing metrics from physical design
- Verification coverage

## 🐛 Troubleshooting

1. **API Key Issues**: Ensure `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` is set in `.env` file
2. **Docker Permissions**: Run with appropriate volume mount permissions
3. **Memory Issues**: For large designs, increase Docker memory allocation
4. **Tool Failures**: Check OpenROAD installation and environment setup

## 📚 Additional Resources

- Problem specifications: `../problems/visible/`
- RTL templates: `templates/rtl_templates/`
- Example solutions: `examples/`
- Implementation details: `Implementation_Plan.md` 