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

# Configuration
USE_OPENROUTER = False  # Set to True for OpenRouter
OPENROUTER_API_KEY = "your_openrouter_api_key_here"
PRIMARY_MODEL = "llama3.1"  # For creative tasks
VALIDATION_MODEL = "llama3.1"  # For validation/error resolution
MAX_ITERATIONS = 3
DB_CONFIG = {
    "dbname": "optimization_db",
    "user": "postgres",
    "password": "your_password",
    "host": "localhost",
    "port": "5432"
}
MAX_WORKERS = 4  # Max threads for parallel execution

class OptimizationSystem:
    def __init__(self):
        self.conn = psycopg2.connect(**DB_CONFIG)
        self.conn.set_session(autocommit=True)  # Ensure thread safety
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
        self.lock = threading.Lock()  # For thread-safe DB access

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
        """Save state to PostgreSQL with thread safety."""
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
                    "INSERT INTO error_log (step, error, resolution, timestamp) VALUES (%s, %s, %s, %s)",
                    (step, error, resolution, timestamp)
                )
                self.state["error_log"].append({"step": step, "error": error, "resolution": resolution, "timestamp": timestamp})
                self.save_state()

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
        prompt = f"Error in Step {step}: {error}. Resolve by reasoning or adjusting the approach. Provide a solution or explain why it's unresolvable."
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
        """Step 1: Break Down the Problem (Sequential)."""
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
                return json.loads(resolution) if resolution else {"task_name": task["name"], "pairs": [], "method": "failed"}

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
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
            Propose 3–5 diverse optimization strategies H_i to improve P.
            Return JSON: {{task_name, strategies: []}}.
            """
            try:
                response = self.call_llm(prompt, model=PRIMARY_MODEL)
                return json.loads(response)
            except Exception as e:
                resolution = self.resolve_error(3, f"Failed to generate hypotheses for {task['name']}: {str(e)}")
                return json.loads(resolution) if resolution else {"task_name": task["name"], "strategies": []}

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
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
            Create a branch B_ij with 3–5 solution variants V_ijk.
            Describe each V_ijk and its implementation.
            Return JSON: {{task_name, branch_id, hypothesis, variants: []}}.
            """
            try:
                response = self.call_llm(prompt, model=PRIMARY_MODEL)
                return json.loads(response)
            except Exception as e:
                resolution = self.resolve_error(4, f"Failed to generate branch {branch_id} for {task['name']}: {str(e)}")
                return json.loads(resolution) if resolution else {"task_name": task["name"], "branch_id": branch_id, "hypothesis": hypothesis, "variants": []}

        branches = {}
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_branch = {}
            for task in self.state["task_set"]:
                task_name = task["name"]
                branches[task_name] = {}
                for h in next((h["strategies"] for h in self.state["hypotheses"] if h["task_name"] == task_name), []):
                    branch_id = f"b_{len(branches[task_name]) + 1}"
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
        """Step 5: Test and Select (Parallel)."""
        def evaluate_variant(task_name: str, branch_id: str, variant: Dict) -> Dict:
            prompt = f"""
            Variant: {json.dumps(variant)}
            Task: {task_name}
            Dataset: {json.dumps(next((d for d in self.state['dataset'] if d['task_name'] == task_name), {}))}
            Metric: {self.state['system_description'].get('metric', 'unknown')}
            Constraints: {json.dumps(self.state['system_description'].get('constraints', {}))}
            Describe how to evaluate performance and assume correctness if outputs match.
            Return JSON: {{task_name, branch_id, variant_id, performance: {{metric, value}}}}.
            """
            try:
                response = self.call_llm(prompt, model=VALIDATION_MODEL)
                return json.loads(response)
            except Exception as e:
                resolution = self.resolve_error(5, f"Failed to evaluate variant for {task_name}: {str(e)}")
                return json.loads(resolution) if resolution else {"task_name": task_name, "branch_id": branch_id, "variant_id": variant.get("id", "unknown"), "performance": {"metric": "unknown", "value": 0}}

        # Parallel evaluation of variants
        performance_results = []
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_variant = {}
            for task_name, branches in self.state["solution_branches"].items():
                for branch_id, branch in branches.items():
                    for variant in branch["variants"]:
                        future = executor.submit(evaluate_variant, task_name, branch_id, variant)
                        future_to_variant[future] = (task_name, branch_id, variant)
            for future in as_completed(future_to_variant):
                performance_results.append(future.result())

        # Select best solutions
        best_solutions = []
        for task_name in {r["task_name"] for r in performance_results}:
            task_results = sorted(
                [r for r in performance_results if r["task_name"] == task_name],
                key=lambda x: x["performance"]["value"],
                reverse=True
            )[:2]  # Top 2 per task
            best_solutions.extend(task_results)

        # Propose suggested design
        prompt = f"""
        Best Solutions: {json.dumps(best_solutions)}
        Task Set: {json.dumps(self.state['task_set'])}
        Constraints: {json.dumps(self.state['system_description'].get('constraints', {}))}
        Propose a suggested system design S' combining best solutions:
        - Components: Tasks with selected variants
        - Interactions: Data flow
        - Evaluation plan: Metrics or simulation
        Return JSON: {{components, interactions, evaluation}}.
        """
        try:
            response = self.call_llm(prompt, model=VALIDATION_MODEL)
            suggested_design = json.loads(response)
        except Exception as e:
            resolution = self.resolve_error(5, f"Failed to propose design: {str(e)}")
            suggested_design = json.loads(resolution) if resolution else {"components": [], "interactions": [], "evaluation": "unknown"}

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
        1. Analyze results for strengths/weaknesses.
        2. Update task_set with best solutions (re-integrate).
        3. Validate compatibility in suggested_design.
        4. If performance meets target (ask user if unclear) or iteration_count >= {MAX_ITERATIONS}, return final design.
        5. Else, propose new hypotheses H' and prune low-performing branches.
        Return JSON: {{new_hypotheses: [], updated_task_set: [], final_design: {{}} or null}}.
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