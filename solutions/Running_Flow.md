Opening Docker

```bash
docker run -it --rm \
  -v ~/iclad_hackathon:/workspace/iclad_hackathon \
  iclad_hackathon:latest bash
```


Go to proper Folder
```bash
cd /workspace/iclad_hackathon/ICLAD-Hackathon-2025/problem-categories/ASU-Spec2Tapeout-ICLAD25-Hackathon/solutions
```


Install Requirements

```bash
cd /workspace/iclad_hackathon/ICLAD-Hackathon-2025/problem-categories/ASU-Spec2Tapeout-ICLAD25-Hackathon/solutions && python3 -m pip install -r requirements.txt
```


Running the agent (LangGraph implementation)


The main entry point is `your_agent_langgraph.py` which provides the complete RTL-to-GDSII flow using LangGraph.

Basic autonomous mode:
```bash
python3 your_agent_langgraph.py --problem ../problems/visible/p1.yaml --output_dir ./visible/p1/
```

Run all visible problems in autonomous mode:
```bash
python3 your_agent_langgraph.py --problem_dir ../problems/visible/ --output_base ./visible/
```

With human-in-the-loop:
```bash
python3 your_agent_langgraph.py --mode human_in_loop --problem ../problems/visible/p1.yaml --output_dir ./visible/p1/
```

Quiet mode (minimal output):
```bash
python3 your_agent_langgraph.py --quiet --problem_dir ../problems/visible/ --output_base ./visible/
```

With custom API key:
```bash
python3 your_agent_langgraph.py --llm_key YOUR_OPENAI_KEY --problem ../problems/visible/p1.yaml --output_dir ./visible/p1/
```