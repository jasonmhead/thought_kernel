import psycopg2
import json
import ollama
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tabulate import tabulate
import sys
from typing import Dict, List, Any
import datetime
import threading
import importlib

class ConfigManager:
    """Manages configuration from config.json."""
    def __init__(self, config_path: str = "config.json"):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        self.validate_config()

    def validate_config(self):
        """Validate configuration structure."""
        required = ["database", "llm", "optimization", "tools"]
        for key in required:
            if key not in self.config:
                raise ValueError(f"Missing {key} in config.json")
        for tool in self.config["tools"]:
            if not all(k in tool for k in ["name", "description", "type", "details"]):
                raise ValueError(f"Invalid tool format: {tool}")
            if tool["type"] not in ["python_function", "api_call"]:
                raise ValueError(f"Unsupported tool type: {tool['type']}")

    def get_db_config(self) -> Dict:
        return self.config["database"]

    def get_llm_config(self) -> Dict:
        return self.config["llm"]

    def get_optimization_config(self) -> Dict:
        return self.config["optimization"]

    def get_tools(self) -> List[Dict]:
        return self.config["tools"]

class OptimizationSystem:
    def __init__(self, config_path: str = "config.json"):
        self.config = ConfigManager(config_path)
        self.conn = psycopg2.connect(**self.config.get_db_config())
        self.conn.set_session(autocommit=True)
        self.create_tables()
        self.state = {
            "system_description": {},
            "task_set": [],
            "dataset": [],
            "hypotheses": [],
            "solution_branches": {},
            "best_solutions": [],
            "suggested_design": {},
            "performance_results": [],
            "iteration_count": 0,
            "error_log": []
        }
        self.current_step = 1
        self.lock = threading.Lock()

    def create_tables(self):
        """Initialize PostgreSQL tables."""
        with self.conn.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS state (
                    key TEXT PRIMARY KEY,
                    value JSONB
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS error_log (
                    id SERIAL PRIMARY KEY,
                    step INTEGER,
                    error TEXT,
                    resolution TEXT,
                    timestamp TIMESTAMP
                )
            """)

    def save_state(self):
        """Save state to PostgreSQL."""
        with self.lock:
            with self.conn.cursor() as cursor:
                for key, value in self.state.items():
                    cursor.execute(
                        "INSERT INTO state (key, value) VALUES (%s, %s) ON CONFLICT (key) DO UPDATE SET value = %s",
                        (key, json.dumps(value), json.dumps(value))
                    )

    def load_state(self):
        """Load state from PostgreSQL."""
        with self.lock:
            with self.conn.cursor() as cursor:
                cursor.execute("SELECT key, value FROM state")
                rows = cursor.fetchall()
                for key, value in rows:
                    self.state[key] = json.loads(value)
        return self.state

    def log_error(self, step: int, error: str, resolution: str = None):
        """Log errors to PostgreSQL."""
        with self.lock:
            with self.conn.cursor() as cursor:
                timestamp = datetime.datetime.now().isoformat()
                cursor.execute(
                    "INSERT INTO error_log (step, error, resolution, timestamp)自主

                    "timestamp": timestamp
                )
                self.state["error_log"].append({"step": step, "error": error, "resolution": resolution, "timestamp": timestamp})
                self.save_state()

    def call_llm(self, prompt: str, model: str = None) -> str:
        """Call LLM (Ollama or OpenRouter)."""
        llm_config = self.config.get_llm_config()
        model = model or llm_config["primary_model"]
        if llm_config["use_openrouter"]:
            headers = {"Authorization": f"Bearer {llm_config['openrouter_api_key']}", "Content-Type": "application/json"}
            payload = {"model": model, "messages": [{"role": "user", "content": prompt}]}
            response = requests.post("https://openrouter.ai/api/v1/chat/completions", json=payload, headers=headers)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        else:
            response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
            return response["message"]["content"]

    def resolve_error(self, step: int, error: str) -> str:
        """Attempt LLM resolution, escalate to human."""
        llm_config = self.config.get_llm_config()
        prompt = f"Error in Step {step}: {error}. Resolve by reasoning or adjusting the approach. Provide a solution or explain why it's unresolvable."
        for attempt in range(2):
            response = self.call_llm(prompt, model=llm_config["validation_model"])
            if "unresolvable" not in response.lower():
                self.log_error(step, error, resolution=response)
                return response
            prompt += f"\nAttempt {attempt+1} failed. Try a different approach."
        self.log_error(step, error, resolution=None)
        print(f"Error: {error}. Please provide guidance to resolve.")
        resolution = input("Your input: ")
        self.log_error(step, error, resolution=resolution)
        return resolution

    def execute_tool(self, tool: Dict, variant: Dict, dataset: List) -> Dict:
        """Execute a tool for evaluation."""
        try:
            if tool["type"] == "python_function":
                module = importlib.import_module(tool["details"]["module"])
                func = getattr(module, tool["details"]["function"])
                return func(variant, dataset)
            elif tool["type"] == "api_call":
                body = tool["details"]["body_template"].replace("{{variant}}", json.dumps(variant)).replace("{{dataset}}", json.dumps(dataset))
                response = requests.request(
                    method=tool["details"]["method"],
                    url=tool["details"]["endpoint"],
                    headers=tool["details"].get("headers", {}),
                    data=body
                )
                response.raise_for_status()
                return response.json()
        except Exception as e:
            resolution = self.resolve_error(5, f"Tool {tool['name']} failed: {str(e)}")
            return json.loads(resolution) if resolution else {"metric": "unknown", "value": 0}

    def step_1_break_down(self, system_input: Dict[str, Any]):
        """Step 1: Break Down the Problem."""
        prompt = f"""
        System Description: {json.dumps(system_input)}
        Decompose the system S into tasks T = {{T_1, T_2, ...}}, where each T_i is a subsystem critical to P.
        For each T_i, provide:
        - Name and description
        - Role in S
        - Impact on P
        Validate that tasks cover S and align with constraints C.
        Return a JSON list of tasks.
        """
        try:
            response = self.call_llm(prompt)
            tasks = json.loads(response)
            self.state["system_description"] = system_input
            self.state["task_set"] = tasks
            self.save_state()
            print(f"Step 1 Summary:\n{tabulate([[t['name'], t['description']] for t in tasks], headers=['Task', 'Description'], tablefmt='grid')}")
        except Exception as e:
            resolution = self.resolve_error(1, str(e))
            tasks = json.loads(resolution) if resolution else []
            self.state["task_set"] = tasks
            self.save_state()
        self.current_step = 2

    def step_2_create_scenarios(self):
        """Step 2: Create Scenarios (Parallel)."""
        def generate_dataset(task: Dict) -> Dict:
            prompt = f"""
            Task: {json.dumps(task)}
            Constraints: {json.dumps(self.state['system_description'].get('constraints', {}))}
            Generate a synthetic dataset D_i = {{(I_i, O_i)}} with 10–50 pairs (normal and edge cases).
            Describe generation method and relevance.
            Return JSON: {{task_name, pairs, method}}.
            """
            try:
                response = self.call_llm(prompt)
                return json.loads(response)
            except Exception as e:
                resolution = self.resolve_error(2, f"Failed to generate dataset for {task['name']}: {str(e)}")
                return json.loads(resolution) if resolutionრ

                resolution = json.loads(resolution) if resolution else {"task_name": task["name"], "pairs": [], "method": "failed"}

        with ThreadPoolExecutor(max_workers=self.config.get_optimization /

System: I notice your message was cut off. It seems you were in the middle of providing code for `step_2_create_scenarios`. Would you like me to complete that function or address any specific part of the integration of LLM tool use, configuration logic, or parallel threading? Since you mentioned the original paper and a robot arm control system, I can also tailor the response to focus on that context if desired.

To proceed efficiently, I’ll assume you want the completion of the `step_2_create_scenarios` function and the remaining steps, ensuring the script integrates **LLM tool use** for evaluation, uses a **configurable number of tools** from `config.json`, and maintains **parallel threading** and **PostgreSQL** for state management. I’ll complete the script, focusing on the robot arm optimization example, and include tool usage in **Step 5** for evaluating latency or other metrics relevant to a robot arm control system. I’ll also ensure the configuration logic is fully separated in `config.json`.

Below is the continuation of the Python script, starting from the incomplete `step_2_create_scenarios` function, with explanations of how tool use, threading, and configuration are integrated.

---

### Continuation of Python Script (`optimize.py`)

```python
System) -> Dict:
            with ThreadPoolExecutor(max_workers=self.config.get_optimization_config()['max_workers']) as executor:
                future_to_task = {executor.submit(generate_dataset, task): task for task in self.state["task_set"]}
                dataset = []
                for future in as_completed(future_to_task):
                    dataset.append(future.result())
            self.state["dataset"] = dataset
            self.save_state()
            print(f"Step 2 Summary:\n{tabulate([[d['task_name'], len(d['pairs']), d['method']] for d in dataset], headers=['Task', 'Pairs', 'Method'], tablefmt='grid')}")
            self.current_step = 3

    def step_3_reason_strategies(self):
        """Step 3: Reason Strategies (Parallel)."""
        def generate_hypotheses(task: Dict) -> Dict:
            prompt = f"""
            Task: {json.dumps(task)}
            Dataset: {json.dumps(next((d for d in self.state['dataset'] if d['task_name'] == task['name']), {}))}
            Constraints: {json.dumps(self.state['system_description'].get('constraints', {}))}
            Propose 3–5 diverse optimization strategies H_i to improve P (e.g., minimize latency for robot arm control).
            Strategies should be feasible and varied (e.g., caching, parallelization, algorithmic changes).
            Return JSON: {{task_name, strategies: []}}.
            """
            try:
                response = self.call_llm(prompt, model=self.config.get_llm_config()['primary_model'])
                return json.loads(response)
            except Exception as e:
                resolution = self.resolve_error(3, f"Failed to generate hypotheses for {task['name']}: {str(e)}")
                return json.loads(resolution) if resolution else {"task_name": task["name"], "strategies": []}

        with ThreadPoolExecutor(max_workers=self.config.get_optimization_config()['max_workers']) as executor:
            future_to_task = {executor.submit(generate_hypotheses, task): task for task in self.state["task_set"]}
            hypotheses = []
            for future in as_completed(future_to_task):
                hypotheses.append(future.result())
        self.state["hypotheses"] = hypotheses
        self.save_state()
        print(f"Step 3 Summary:\n{tabulate([[h['task_name'], len(h['strategies'])] for h in hypotheses], headers=['Task', 'Strategies'], tablefmt='grid')}")
        self.current_step = 4

    def step_4_explore_solutions(self):
        """Step 4: Explore Solutions (Parallel)."""
        def generate_branch(task: Dict, hypothesis: str, branch_id: str) -> Dict:
            prompt = f"""
            Task: {json.dumps(task)}
            Hypothesis: {hypothesis}
            Constraints: {json.dumps(self.state['system_description'].get('constraints', {}))}
            Create a branch B_ij with 3–5 solution variants V_ijk (e.g., different algorithms or configurations for robot arm control).
            Describe each V_ijk, its implementation of H_ij, and expected impact on P.
            Return JSON: {{task_name, branch_id, hypothesis, variants: []}}.
            """
            try:
                response = self.call_llm(prompt, model=self.config.get_llm_config()['primary_model'])
                return json.loads(response)
            except Exception as e:
                resolution = self.resolve_error(4, f"Failed to generate branch {branch_id} for {task['name']}: {str(e)}")
                return json.loads(resolution) if resolution else {"task_name": task["name"], "branch_id": branch_id, "hypothesis": hypothesis, "variants": []}

        branches = {}
        with ThreadPoolExecutor(max_workers=self.config.get_optimization_config()['max_workers']) as executor:
            future_to_branch = {}
            for task in self.state["task_set"]:
                task_name = task["name"]
                branches[task_name] = {}
                for h_idx, h in enumerate(next((h["strategies"] for h in self.state["hypotheses"] if h["task_name"] == task_name), []), 1):
                    branch_id = f"b_{h_idx}"
                    future = executor.submit(generate_branch, task, h, branch_id)
                    future_to_branch[future] = (task_name, branch_id)
            for future in as_completed(future_to_branch):
                task_name, branch_id = future_to_branch[future]
                branches[task_name][branch_id] = future.result()
        self.state["solution_branches"] = branches
        self.save_state()
        table = [[task, bid, len(branch["variants"])] for task, branches in branches.items() for bid, branch in branches.items()]
        print(f"Step 4 Summary:\n{tabulate(table, headers=['Task', 'Branch', 'Variants'], tablefmt='grid')}")
        self.current_step = 5

    def step_5_test_select(self):
        """Step 5: Test and Select Best Solutions (Parallel with Tool Use)."""
        def evaluate_variant(task_name: str, branch_id: str, variant: Dict) -> Dict:
            tools = self.config.get_tools()
            prompt = f"""
            Variant: {json.dumps(variant)}
            Task: {task_name}
            Dataset: {json.dumps(next((d for d in self.state['dataset'] if d['task_name'] == task_name), {}))}
            Metric: {self.state['system_description'].get('metric', 'unknown')}
            Constraints: {json.dumps(self.state['system_description'].get('constraints', {}))}
            Available Tools: {json.dumps([{"name": t["name"], "description": t["description"]} for t in tools])}
            Select the most appropriate tool to evaluate performance (e.g., latency for robot arm control).
            Describe the evaluation process and assume correctness if outputs match expected.
            Return JSON: {{task_name, branch_id, variant_id, tool_used, performance: {{metric, value}}}}.
            """
            try:
                response = self.call_llm(prompt, model=self.config.get_llm_config()['validation_model'])
                eval_plan = json.loads(response)
                tool_name = eval_plan["tool_used"]
                tool = next((t for t in tools if t["name"] == tool_name), None)
                if not tool:
                    raise ValueError(f"Tool {tool_name} not found")
                result = self.execute_tool(tool, variant, next((d for d in self.state['dataset'] if d['task_name'] == task_name), []))
                return {
                    "task_name": task_name,
                    "branch_id": branch_id,
                    "variant_id": variant.get("id", "unknown"),
                    "tool_used": tool_name,
                    "performance": result
                }
            except Exception as e:
                resolution = self.resolve_error(5, f"Evaluation failed for {task_name}: {str(e)}")
                return json.loads(resolution) if resolution else {
                    "task_name": task_name,
                    "branch_id": branch_id,
                    "variant_id": variant.get("id", "unknown"),
                    "tool_used": "none",
                    "performance": {"metric": "unknown", "value": 0}
                }

        performance_results = []
        with ThreadPoolExecutor(max_workers=self.config.get_optimization_config()['max_workers']) as executor:
            future_to_variant = {}
            for task_name, branches in self.state["solution_branches"].items():
                for branch_id, branch in branches.items():
                    for v_idx, variant in enumerate(branch["variants"], 1):
                        variant["id"] = f"v_{v_idx}"  # Ensure variant has an ID
                        future = executor.submit(evaluate_variant, task_name, branch_id, variant)
                        future_to_variant[future] = (task_name, branch_id, variant["id"])
            for future in as_completed(future_to_variant):
                performance_results.append(future.result())

        # Select best solutions
        best_solutions = []
        for task_name in {r["task_name"] for r in performance_results}:
            task_results = sorted(
                [r for r in performance_results if r["task_name"] == task_name],
                key=lambda x: x["performance"]["value"],
                reverse=self.state["system_description"].get("metric", "").startswith("minimize")
            )[:2]  # Top 2 per task
            best_solutions.extend(task_results)

        # Propose suggested system design
        prompt = f"""
        Best Solutions: {json.dumps(best_solutions)}
        Task Set: {json.dumps(self.state['task_set'])}
        Constraints: {json.dumps(self.state['system_description'].get('constraints', {}))}
        Available Tools: {json.dumps([{"name": t["name"], "description": t["description"]} for t in self.config.get_tools()])}
        Propose a suggested system design S' combining best solutions for a robot arm control system:
        - Components: Tasks with selected variants
        - Interactions: Data flow between tasks
        - Evaluation plan: Select a tool and describe its use
        Return JSON: {{components, interactions, evaluation: {{tool, method}}}}.
        """
        try:
            response = self.call_llm(prompt, model=self.config.get_llm_config()['validation_model'])
            suggested_design = json.loads(response)
            # Evaluate suggested design if tool specified
            if suggested_design["evaluation"]["tool"]:
                tool = next((t for t in self.config.get_tools() if t["name"] == suggested_design["evaluation"]["tool"]), None)
                if tool:
                    design_result = self.execute_tool(tool, suggested_design["components"], self.state["dataset"])
                    suggested_design["evaluation"]["result"] = design_result
        except Exception as e:
            resolution = self.resolve_error(5, f"Failed to propose or evaluate design: {str(e)}")
            suggested_design = json.loads(resolution) if resolution else {"components": [], "interactions": [], "evaluation": {"tool": "none", "method": "unknown"}}

        self.state["best_solutions"] = best_solutions
        self.state["suggested_design"] = suggested_design
        self.state["performance_results"] = performance_results
        self.save_state()
        print(f"Step 5 Summary:\nBest Solutions: {len(best_solutions)}\nSuggested Design: {json.dumps(suggested_design, indent=2)}")
        self.current_step = 6

    def step_6_iterate_refine(self):
        """Step 6: Iterate and Refine (Sequential)."""
        prompt = f"""
        Best Solutions: {json.dumps(self.state['best_solutions'])}
        Suggested Design: {json.dumps(self.state['suggested_design'])}
        Performance Results: {json.dumps(self.state['performance_results'])}
        Task Set: {json.dumps(self.state['task_set'])}
        Constraints: {json.dumps(self.state['system_description'].get('constraints', {}))}
        Available Tools: {json.dumps([{"name": t["name"], "description": t["description"]} for t in self.config.get_tools()])}
        1. Analyze results for strengths/weaknesses.
        2. Update task_set with best solutions (re-integrate subsystems).
        3. Validate compatibility in suggested_design.
        4. If performance meets target (ask user if unclear) or iteration_count >= {self.config.get_optimization_config()['max_iterations']}, return final design.
        5. Else, propose new hypotheses H' and prune low-performing branches.
        Return JSON: {{new_hypotheses: [], updated_task_set: [], final_design: {{}} or null}}.
        """
        self.state["iteration_count"] += 1
        try:
            response = self.call_llm(prompt, model=self.config.get_llm_config()['primary_model'])
            result = json.loads(response)
            self.state["task_set"] = result["updated_task_set"]
            if result["final_design"]:
                # Evaluate final design
                tool_name = result["final_design"].get("evaluation", {}).get("tool", "")
                if tool_name:
                    tool = next((t for t in self.config.get_tools() if t["name"] == tool_name), None)
                    if tool:
                        design_result = self.execute_tool(tool, result["final_design"]["components"], self.state["dataset"])
                        result["final_design"]["evaluation"]["result"] = design_result
                self.state["suggested_design"] = result["final_design"]
                self.save_state()
                print(f"Step 6 Summary: Final Design\n{json.dumps(result['final_design'], indent=2)}")
                self.current_step = 0
            else:
                self.state["hypotheses"] = result["new_hypotheses"]
                # Prune branches
                for task_name, branches in self.state["solution_branches"].items():
                    for bid in list(branches.keys()):
                        if not any(v["variant_id"] in [b["variant_id"] for b in self.state["best_solutions"]] for v in branches[bid]["variants"]):
                            del branches[bid]
                self.save_state()
                print(f"Step 6 Summary: Iteration {self.state['iteration_count']}, new hypotheses generated.")
                self.current_step = 4
        except Exception as e:
            resolution = self.resolve_error(6, f"Failed to refine or finalize: {str(e)}")
            result = json.loads(resolution) if resolution else {"new_hypotheses": [], "updated_task_set": self.state["task_set"], "final_design": None}
            self.state.update({"hypotheses": result["new_hypotheses"], "task_set": result["updated_task_set"]})
            if result["final_design"]:
                self.state["suggested_design"] = result["final_design"]
                self.current_step = 0
            else:
                self.current_step = 4
            self.save_state()

    def run(self):
        """Main loop."""
        system_input = input("Describe system S, metric P, constraints C (JSON or text): ")
        try:
            system_input = json.loads(system_input)
        except json.JSONDecodeError:
            system_input = {"description": system_input, "metric": "unknown", "constraints": {}}
        
        while self.current_step != 0:
            if self.current_step == 1:
                self.step_1_break_down(system_input)
            elif self.current_step == 2:
                self.step_2_create_scenarios()
            elif self.current_step == 3:
                self.step_3_reason_strategies()
            elif self.current_step == 4:
                self.step_4_explore_solutions()
            elif self.current_step == 5:
                self.step_5_test_select()
            elif self.current_step == 6:
                self.step_6_iterate_refine()
        self.conn.close()

if __name__ == "__main__":
    opt = OptimizationSystem()
    opt.run()