#!/bin/bash

# ASU Tapeout Agent Docker Runner
# This script runs the ASU tapeout agent inside the Docker container
# where all EDA tools (Yosys, OpenROAD, etc.) are available

set -e

echo "Starting ASU Tapeout Agent in Docker container..."
echo "This will run the agent with access to EDA tools."

# Check if we're already inside a container
if [ -f /.dockerenv ]; then
    echo "Already inside Docker container. Running agent directly..."
    cd /workspace/iclad_hackathon/ICLAD-Hackathon-2025/problem-categories/ASU-Spec2Tapeout-ICLAD25-Hackathon/solutions
    python3 your_agent_langgraph.py "$@"
else
    echo "Running agent inside Docker container..."
    
    # Check if Docker image exists
    if ! docker image inspect iclad_hackathon:latest >/dev/null 2>&1; then
        echo "Error: Docker image 'iclad_hackathon:latest' not found!"
        echo "Please build or pull the ICLAD hackathon Docker image first."
        exit 1
    fi
    
    # Run the container with the agent
    docker run -it --rm \
        -v ~/iclad_hackathon:/workspace/iclad_hackathon \
        iclad_hackathon:latest \
        bash -c "cd /workspace/iclad_hackathon/ICLAD-Hackathon-2025/problem-categories/ASU-Spec2Tapeout-ICLAD25-Hackathon/solutions && python3 your_agent_langgraph.py $*"
fi

echo "Agent execution completed." 