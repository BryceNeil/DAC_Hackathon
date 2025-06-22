# ASU Tapeout Agent Setup Instructions

## Quick Start

### 1. Environment Setup (LLM Category - Recommended)

Start the Docker container with all dependencies:
```bash
docker run -it --rm \
  -v ~/iclad_hackathon:/workspace/iclad_hackathon \
  iclad_hackathon:latest bash
```

Navigate to the project:
```bash
cd /workspace/iclad_hackathon/ICLAD-Hackathon-2025/problem-categories/ASU-Spec2Tapeout-ICLAD25-Hackathon
```

### 2. Install Python Dependencies

```bash
cd solutions
pip install -r requirements.txt
```

### 3. Configure LLM API (Choose one)

**Option A: OpenAI**
```bash
export OPENAI_API_KEY="your-api-key-here"
```

**Option B: Anthropic Claude**
```bash
export ANTHROPIC_API_KEY="your-api-key-here"
```

### 4. Test the Agent

Run on a single problem:
```bash
python3 your_agent.py --problem ../problems/visible/p1.yaml --output_dir ./visible/p1/
```

Run on all problems:
```bash
python3 your_agent.py --problem_dir ../problems/visible/ --output_base ./visible/
```

### 5. Verify Your Solutions

Test RTL functionality:
```bash
cd ../evaluation
python3 evaluate_verilog.py --verilog ../solutions/visible/p1/seq_detector_0011.v --problem 1 --tb ./visible/p1_tb.v
```

Test layout metrics:
```bash
python3 evaluate_openroad.py --odb ../solutions/visible/p1/6_final.odb --sdc ../solutions/visible/p1/6_final.sdc --flow_root /path/to/OpenROAD-flow-scripts --problem 1
```

## Next Steps

1. **Implement LLM Integration**: Edit `your_agent.py` to add actual LLM API calls
2. **Add OpenROAD Flow**: Integrate with OpenROAD-flow-scripts for physical design
3. **Iterative Testing**: Use the evaluation scripts to refine your solutions

## Directory Structure After Setup

```
solutions/
├── your_agent.py          # Main agent script
├── requirements.txt       # Python dependencies  
├── setup_instructions.md  # This file
└── visible/              # Generated solutions
    ├── p1/
    │   ├── seq_detector_0011.v
    │   ├── 6_final.sdc
    │   └── 6_final.odb
    ├── p5/
    └── ...
``` 