# 🧠 Thought Kernel

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: Unlicense](https://img.shields.io/badge/license-Unlicense-blue.svg)](http://unlicense.org/)

> **Inspired by Stanford's CRFM research** on [AI-generated kernels](https://crfm.stanford.edu/2025/05/28/fast-kernels.html), Thought Kernel is an advanced optimization framework that leverages AI to systematically explore and optimize complex systems through branching exploration and iterative refinement.

> Vibe coded by [Jason Head](https://github.com/jasonmhead) through Grok AI

## 🌟 Features

- **Multi-LLM Support**: Works with OpenAI, Ollama, and OpenRouter for flexible model choices
- **Local & Cloud Options**: Run models locally with Ollama or use cloud-based solutions
- **Systematic Optimization**: Break down complex systems into manageable tasks
- **Branching Exploration**: Generate and evaluate multiple solution variants in parallel
- **Iterative Refinement**: Continuously improve solutions through multiple iterations
- **Performance-Driven**: Focus on maximizing your specified performance metrics
- **Constraint-Aware**: Respects system constraints and requirements
- **Human-in-the-Loop**: Seamlessly integrates human feedback when needed

## 🔄 Optimization Process

Thought Kernel follows a systematic, six-step optimization process to break down complex systems and find optimal solutions:

1. **System Decomposition**  
   - Breaks down your system into core tasks and components
   - Identifies critical paths and dependencies
   - Validates task coverage against system constraints

2. **Scenario Creation**  
   - Generates diverse test cases and scenarios
   - Covers both normal and edge cases
   - Builds a representative dataset for evaluation

3. **Strategy Formulation**  
   - Proposes multiple optimization hypotheses per task
   - Considers diverse approaches (algorithmic, architectural, etc.)
   - Balances exploration and exploitation

4. **Solution Exploration**  
   - Creates solution variants for each hypothesis
   - Evaluates trade-offs between different approaches
   - Maintains multiple solution branches in parallel

5. **Testing & Selection**  
   - Executes automated tests using configured tools
   - Measures performance against defined metrics
   - Selects top-performing solutions

6. **Iteration & Refinement**  
   - Analyzes results and identifies improvement areas
   - Refines solutions based on performance data
   - Iterates until convergence or stopping criteria met

This process is fully automated but allows for human intervention when needed, and all steps are tracked and can be reviewed for transparency.

## 📂 Repository Structure

```
thought_kernel/
├── README.md                     # Project overview, setup, and usage instructions
├── SYSTEM_README.md              # Detailed system architecture and configuration
├── requirements.txt              # Python package dependencies
├── config.json                   # Configuration file for the optimizer
├── LICENSE                       # License information (Unlicense)
├── thought_kernel.py             # Main optimization logic and entry point
├── tools.py                      # Utility functions and helper methods
├── Systematic_Optimization_prompt.txt  # initial prompt for systematic optimization
└── generating_prompts.txt        # Additional prompt templates
```

### Key Files:
- **README.md**: Project overview, quick start guide, and basic usage
- **SYSTEM_README.md**: In-depth system documentation and architecture
- **requirements.txt**: Python package dependencies (install with `pip install -r requirements.txt`)
- **config.json**: Main configuration file for model providers and optimization parameters
- **thought_kernel.py**: Main script containing the optimization logic
- **tools.py**: Helper functions and utilities
- **Systematic_Optimization_prompt.txt**: Core prompt used for systematic optimization
- **generating_prompts.txt**: Additional prompt templates for various use cases

## 📚 Documentation

For detailed system architecture, components, and advanced configuration, see [SYSTEM_README.md](SYSTEM_README.md).

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- Required Python packages (install via `pip install -r requirements.txt`)

### Installation
```bash
# Clone the repository
git clone https://github.com/jasonmhead/thought_kernel
cd thought_kernel

# Install dependencies
pip install -r requirements.txt
```

## 🛠️ Usage

1. **Define Your System**:
   - Describe your system (S)
   - Specify performance metric (P) to optimize
   - List any constraints (C)

2. **Run the Optimizer**:
   ```bash
   python thought_kernel.py --system "Your system description" --metric "Performance metric" --constraints "Your constraints"
   ```

3. **Review Results**:
   - The system will output optimized solutions
   - Performance metrics for each solution variant
   - Suggested system design improvements

## 🔍 How It Works

Thought Kernel follows a six-step optimization process:

1. **Problem Decomposition** - Breaks down the system into core tasks
2. **Scenario Generation** - Creates representative test cases
3. **Strategy Formulation** - Proposes diverse optimization approaches
4. **Parallel Exploration** - Tests multiple solution variants
5. **Performance Evaluation** - Ranks solutions by effectiveness
6. **Iterative Refinement** - Improves solutions through multiple cycles

## ⚙️ Configuration

Create a `config.yaml` file in your project root with the following structure:

```yaml
# AI Model Configuration
openai:
  api_key: your_openai_key
  model: gpt-4-turbo
  temperature: 0.7
  max_tokens: 4096

# Ollama Configuration (local models)
ollama:
  base_url: http://localhost:11434
  model: llama3  # or any Ollama model
  temperature: 0.7

# OpenRouter Configuration
openrouter:
  api_key: your_openrouter_key
  model: anthropic/claude-3-opus  # or any OpenRouter model
  temperature: 0.7

# Default provider (openai, ollama, or openrouter)
provider: openai

# Logging
logging:
  level: INFO
  file: thought_kernel.log
  max_size_mb: 10
  backup_count: 3

# Parallel Processing
parallel:
  max_workers: 4  # Set based on your CPU cores
  timeout_seconds: 300

# Caching
cache:
  enabled: true
  ttl_hours: 24
  path: .cache
```

## 🚀 Quick Start Guide

1. **Install the package**:
   ```bash
   pip install -e .
   ```

2. **Set up your environment**:
   ```bash
   cp config.example.yaml config.yaml
   # Edit config.yaml with your API keys and preferences
   ```

3. **Run the optimizer**:
   ```bash
   # Basic usage
   python -m thought_kernel optimize --system "Your system description" \
     --metric "Performance metric" \
     --constraints "Your constraints"
   
   # With custom config
   python -m thought_kernel optimize --config custom_config.yaml
   
   # For interactive mode
   python -m thought_kernel interactive
   ```

4. **Monitor progress**:
   - Check the log file specified in your config
   - View real-time updates in the console
   - Monitor system resource usage

## 🛠️ Advanced Usage

### Command Line Arguments

```bash
python -m thought_kernel optimize \
  --system "System description" \
  --metric "Performance metric" \
  --constraints "System constraints" \
  --output results/ \n  --max-iterations 5 \
  --num-variants 10 \
  --temperature 0.7 \
  --verbose
```

### Environment Variables

```bash
export THOUGHT_KERNEL_OPENAI_API_KEY=your_api_key
export THOUGHT_KERNEL_LOG_LEVEL=DEBUG
python -m thought_kernel optimize ...
```

### Programmatic Usage

```python
from thought_kernel import ThoughtKernel, Config

# Initialize with custom config
config = Config.from_yaml("config.yaml")
tk = ThoughtKernel(config)

# Run optimization
result = tk.optimize(
    system_description="Your system description",
    performance_metric="Your performance metric",
    constraints="Your constraints"
)

# Access results
print(f"Best solution: {result.best_solution}")
print(f"Performance: {result.performance_metrics}")
```

## 📊 Monitoring and Debugging

- Check log files in the specified directory
- Use `--verbose` flag for detailed output
- Monitor system resources with tools like `htop` or `glances`
- For long-running optimizations, consider using `screen` or `tmux`

## 🤖 Example 1: Robotic Control System Optimization

### Optimizing an Industrial Robotic Arm Control System

```python
# Define the robotic system to optimize
system_description = """
6-DOF industrial robotic arm with the following components:
- High-precision servo motors with encoders
- Force-torque sensor at end-effector
- Vision system for object detection
- Real-time control system (1kHz update rate)

Current challenges:
- End-effector positioning error of ±0.5mm (required: ±0.1mm)
- Vibration during high-speed movements
- Delays in sensor feedback loop (current: 5ms, target: <1ms)
- Inconsistent performance under varying payloads (0.5-5kg)
"""

# Define the key performance metrics to optimize
performance_metric = """
- Positional accuracy (target: ±0.1mm)
- Settling time after movement (target: <100ms)
- Vibration amplitude (target: <0.05mm)
- Control loop latency (target: <1ms)
- Energy efficiency (Joules/cycle)
- Repeatability (target: ±0.02mm)
"""

# Define system constraints
constraints = """
- Must maintain safety standards (ISO 10218-1, ISO 13849)
- Maximum update rate: 2kHz (hardware limit)
- Must handle payload variations (0.5-5kg)
- No mechanical modifications allowed
- Must maintain backward compatibility with existing programs
- Implementation must be completed within 3 months
- Maximum additional compute budget: $50,000
"""

# Run the optimization
optimized_system = optimize(
    system_description=system_description,
    performance_metric=performance_metric,
    constraints=constraints
)

# Example output might include:
# - Optimized PID/MPC control parameters
# - Adaptive feedforward control implementation
# - Vibration damping algorithms
# - Sensor fusion for improved state estimation
# - Real-time trajectory optimization
# - Predictive maintenance schedule
```

### Expected Outcomes:
- 80% improvement in positional accuracy (±0.1mm)
- 60% reduction in settling time
- 70% reduction in vibration amplitude
- Control loop latency reduced to 0.8ms
- 15% improvement in energy efficiency
- Full compliance with ISO 10218-1 and ISO 13849

## 🏭 Example 2: Manufacturing System Optimization

### Optimizing a CNC Machining Production Line

```python
# Define the manufacturing system to optimize
system_description = """
CNC machining production line with the following components:
- 5-axis CNC machines (x4)
- Automated material handling system
- Quality inspection stations
- Maintenance team

Current issues:
- Machine utilization averages only 65%
- High variability in part quality
- Frequent unplanned downtime
- Inefficient material flow between stations
"""

# Define the key performance metrics to optimize
performance_metric = """
- Overall Equipment Effectiveness (OEE)
- First-pass yield rate
- Mean time between failures (MTBF)
- Production throughput (parts/hour)
"""

# Define system constraints
constraints = """
- Maximum budget for changes: $750,000
- Must maintain current workforce levels
- No reduction in production capacity during transition
- Must maintain or improve part quality
- Implementation must be completed within 4 months
- Must comply with ISO 9001 and AS9100 standards
"""

# Run the optimization
optimized_system = optimize(
    system_description=system_description,
    performance_metric=performance_metric,
    constraints=constraints
)

# Example output might include:
# - Predictive maintenance schedule for CNC machines
# - Optimized tool change procedures
# - Improved material flow layout
# - Staff cross-training program
# - Real-time monitoring dashboard
```

### Expected Outcomes:
- 20-30% increase in OEE
- 40% reduction in unplanned downtime
- 15% improvement in first-pass yield
- 25% increase in production throughput
- ROI within 14 months

## 🤝 Contributing

Contribute! Please feel free to submit a Pull Request.

## 📄 License

This is free and unencumbered software released into the public domain. For more information, see the [LICENSE](LICENSE) file or visit [unlicense.org](https://unlicense.org).

## 🌟 Acknowledgments

- Inspired by Stanford CRFM's research on [AI-generated kernels](https://crfm.stanford.edu/2025/05/28/fast-kernels.html)
- Created by [Jason Head](https://github.com/jasonmhead)
- Connect: [GitHub](https://github.com/jasonmhead) | [X/Twitter](https://x.com/jasonmhead)
