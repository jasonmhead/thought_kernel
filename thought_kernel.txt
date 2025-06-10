import sqlite3
import json
import ollama
import requests
from tabulate import tabulate
import sys
from typing import Dict, List, Any
import datetime

# Configuration
USE_OPENROUTER = False  # Set to True to use OpenRouter instead of Ollama
OPENROUTER_API_KEY = "your_openrouter_api_key_here"  # Required if USE_OPENROUTER=True
PRIMARY_MODEL = "llama3.1"  # Primary LLM for creative tasks (Ollama model name)
VALIDATION_MODEL = "llama3.1"  # Secondary LLM for validation and error resolution
MAX_ITERATIONS = 3  # Maximum iterations for Step 6
DB_PATH = "optimization_state.db"  # SQLite database path

class OptimizationSystem:
    def __init__(self):
        self.conn = sqlite3.connect(DB_PATH)
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

    def create_tables(self):
        """Initialize SQLite tables for state persistence."""
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS state (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS error_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                step INTEGER,
                error TEXT,
                resolution TEXT,
                timestamp TEXT
            )
        """)
        self.conn.commit()

    def save_state(self):
        """Save state dictionary to SQLite."""
        cursor = self.conn.cursor()
        state_json = {k: json.dumps(v) for k, v in self.state.items()}
        for key, value in state_json.items():
            cursor.execute("INSERT OR REPLACE INTO state (key, value) VALUES (?, ?)", (key, value))
        self.conn.commit()

    def load_state(self):
        """Load state dictionary from SQLite."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT key, value FROM state")
        rows = cursor.fetchall()
        for key, value in rows:
            self.state[key] = json.loads(value)
        return self.state

    def log_error(self, step: int, error: str, resolution: str = None):
        """Log errors and resolutions to SQLite."""
        cursor = self.conn.cursor()
        timestamp = datetime.datetime.now().isoformat()
        cursor.execute(
            "INSERT INTO error_log (step, error, resolution, timestamp) VALUES (?, ?, ?, ?)",
            (step, error, resolution, timestamp)
        )
        self.state["error_log"].append({"step": step, "error": error, "resolution": resolution, "timestamp": timestamp})
        self.save_state()
        self.conn.commit()

    def call_llm(self, prompt: str, model: str = PRIMARY_MODEL) -> str:
        """Call LLM (Ollama or OpenRouter)."""
        if USE_OPENROUTER:
            headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
            payload = {"model": model, "messages": [{"role": "user", "content": prompt}]}
            response = requests.post("https://openrouter.ai/api/v1/chat/completions", json=payload, headers=headers)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        else:
            response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
            return response["message"]["content"]

    def resolve_error(self, step: int, error: str) -> str:
        """Attempt LLM resolution, escalate to human if needed."""
        prompt = f"Error in Step {step}: {error}. Attempt to resolve by reasoning or adjusting the approach. Provide a solution or explain why it's unresolvable."
        for attempt in range(2):
            response = self.call_llm(prompt, model=VALIDATION_MODEL)
            if "unresolvable" not in response.lower():
                self.log_error(step, error, resolution=response)
                return response
            prompt += f"\nAttempt {attempt+1} failed. Try a different approach."
        self.log_error(step, error, resolution=None)
        print(f"Error: {error}. Please provide guidance to resolve.")
        resolution = input("Your input: ")
        self.log_error(step, error, resolution=resolution)
        return resolution

    def step_1_break_down(self, system_input: Dict[str, Any]):
        """Step 1: Break Down the Problem."""
        prompt = f"""
        System Description: {json.dumps(system_input)}
        Decompose the system S into tasks T = {{T_1, T_2, ...}}, where each T_i is a subsystem or operation critical to performance metric P.
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
        """Step 2: Create Representative Scenarios."""
        prompt = f"""
        Task Set: {json.dumps(self.state['task_set'])}
        Constraints: {json.dumps(self.state['system_description'].get('constraints', {}))}
        For each task T_i, generate a synthetic dataset D_i = {{(I_i, O_i)}} with 10–50 input-output pairs (normal and edge cases).
        Describe generation method and relevance.
        Return a JSON list of datasets, each with task_name, pairs, and method.
        """
        try:
            response = self.call_llm(prompt)
            dataset = json.loads(response)
            self.state["dataset"] = dataset
            self.save_state()
            print(f"Step 2 Summary:\n{tabulate([[d['task_name'], len(d['pairs']), d['method']] for d in dataset], headers=['Task', 'Pairs', 'Method'], tablefmt='grid')}")
        except Exception as e:
            resolution = self.resolve_error(2, str(e))
            dataset = json.loads(resolution) if resolution else []
            self.state["dataset"] = dataset
            self.save_state()
        self.current_step = 3

    def step_3_reason_strategies(self):
        """Step 3: Reason About Optimization Strategies."""
        prompt = f"""
        Task Set: {json.dumps(self.state['task_set'])}
        Dataset: {json.dumps(self.state['dataset'])}
        Constraints: {json.dumps(self.state['system_description'].get('constraints', {}))}
        For each task T_i, propose 3–5 diverse optimization strategies H_i in natural language to improve P.
        Ensure strategies are feasible and varied (e.g., algorithmic, structural).
        Return a JSON list of hypotheses, each with task_name and strategies.
        """
        try:
            response = self.call_llm(prompt)
            hypotheses = json.loads(response)
            self.state["hypotheses"] = hypotheses
            self.save_state()
            print(f"Step 3 Summary:\n{tabulate([[h['task_name'], len(h['strategies'])] for h in hypotheses], headers=['Task', 'Strategies'], tablefmt='grid')}")
        except Exception as e:
            resolution = self.resolve_error(3, str(e))
            hypotheses = json.loads(resolution) if resolution else []
            self.state["hypotheses"] = hypotheses
            self.save_state()
        self.current_step = 4

    def step_4_explore_solutions(self):
        """Step 4: Explore Multiple Solutions (Branching)."""
        prompt = f"""
        Hypotheses: {json.dumps(self.state['hypotheses'])}
        Task Set: {json.dumps(self.state['task_set'])}
        Constraints: {json.dumps(self.state['system_description'].get('constraints', {}))}
        For each hypothesis H_ij, create a branch B_ij with 3–5 solution variants V_ijk (e.g., algorithms, configurations).
        Describe each V_ijk and its implementation of H_ij.
        Return a JSON object with branches: {{task_name: {{branch_id: {{hypothesis, variants}}}}}}.
        """
        try:
            response = self.call_llm(prompt, model=PRIMARY_MODEL)
            branches = json.loads(response)
            self.state["solution_branches"] = branches
            self.save_state()
            table = []
            for task, branches in branches.items():
                for bid, branch in branches.items():
                    table.append([task, bid, len(branch['variants'])])
            print(f"Step 4 Summary:\n{tabulate(table, headers=['Task', 'Branch', 'Variants'], tablefmt='grid')}")
        except Exception as e:
            resolution = self.resolve_error(4, str(e))
            branches = json.loads(resolution) if resolution else {}
            self.state["solution_branches"] = branches
            self.save_state()
        self.current_step = 5

    def step_5_test_select(self):
        """Step 5: Test and Select Best Solutions."""
        prompt = f"""
        Solution Branches: {json.dumps(self.state['solution_branches'])}
        Dataset: {json.dumps(self.state['dataset'])}
        Performance Metric: {self.state['system_description'].get('metric', 'unknown')}
        Constraints: {json.dumps(self.state['system_description'].get('constraints', {}))}
        For each variant V_ijk, describe how to evaluate performance on dataset D (e.g., simulation, metrics).
        Assume correctness if outputs match expected. Select top 2–3 variants per task.
        Propose a suggested system design S' combining best variants, including:
        - Components (tasks with selected variants)
        - Interactions (data flow)
        - Evaluation plan
        Return a JSON object: {{best_solutions: [], suggested_design: {{components, interactions, evaluation}}, performance_results: []}}.
        """
        try:
            response = self.call_llm(prompt, model=VALIDATION_MODEL)
            result = json.loads(response)
            self.state["best_solutions"] = result["best_solutions"]
            self.state["suggested_design"] = result["suggested_design"]
            self.state["performance_results"] = result["performance_results"]
            self.save_state()
            print(f"Step 5 Summary:\nBest Solutions: {len(result['best_solutions'])}\nSuggested Design: {json.dumps(result['suggested_design'], indent=2)}")
        except Exception as e:
            resolution = self.resolve_error(5, str(e))
            result = json.loads(resolution) if resolution else {"best_solutions": [], "suggested_design": {}, "performance_results": []}
            self.state.update(result)
            self.save_state()
        self.current_step = 6

    def step_6_iterate_refine(self):
        """Step 6: Iterate and Refine."""
        prompt = f"""
        Best Solutions: {json.dumps(self.state['best_solutions'])}
        Suggested Design: {json.dumps(self.state['suggested_design'])}
        Performance Results: {json.dumps(self.state['performance_results'])}
        Task Set: {json.dumps(self.state['task_set'])}
        Constraints: {json.dumps(self.state['system_description'].get('constraints', {}))}
        1. Analyze results to identify strengths/weaknesses.
        2. Update task_set with best solutions (re-integrate subsystems).
        3. Validate compatibility in suggested_design.
        4. If performance meets target (ask user if unclear) or iteration_count >= {MAX_ITERATIONS}, return final suggested_design.
        5. Else, propose new hypotheses H' and prune low-performing branches.
        Return a JSON object: {{new_hypotheses: [], updated_task_set: [], final_design: {{}} or null}}.
        """
        self.state["iteration_count"] += 1
        try:
            response = self.call_llm(prompt, model=PRIMARY_MODEL)
            result = json.loads(response)
            self.state["task_set"] = result["updated_task_set"]
            if result["final_design"]:
                self.state["suggested_design"] = result["final_design"]
                self.save_state()
                print(f"Step 6 Summary: Final Design\n{json.dumps(result['final_design'], indent=2)}")
                self.current_step = 0  # Stop
            else:
                self.state["hypotheses"] = result["new_hypotheses"]
                # Prune branches below median performance
                for task, branches in self.state["solution_branches"].items():
                    for bid in list(branches.keys()):
                        if not any(v in self.state["best_solutions"] for v in branches[bid]["variants"]):
                            del branches[bid]
                self.save_state()
                print(f"Step 6 Summary: Iteration {self.state['iteration_count']}, new hypotheses generated.")
                self.current_step = 4
        except Exception as e:
            resolution = self.resolve_error(6, str(e))
            result = json.loads(resolution) if resolution else {"new_hypotheses": [], "updated_task_set": self.state["task_set"], "final_design": None}
            self.state.update({"hypotheses": result["new_hypotheses"], "task_set": result["updated_task_set"]})
            if result["final_design"]:
                self.state["suggested_design"] = result["final_design"]
                self.current_step = 0
            else:
                self.current_step = 4
            self.save_state()

    def run(self):
        """Main loop to execute steps."""
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