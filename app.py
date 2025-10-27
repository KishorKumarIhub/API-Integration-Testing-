import os
import logging
import json
import yaml
from fastapi import FastAPI, UploadFile, HTTPException, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
from typing import List, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
import pdfplumber
import tempfile
import subprocess
import docker
from dotenv import load_dotenv
from datetime import datetime
import random
import string
import google.generativeai as genai

app = FastAPI(title="API Integration Testing Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# --------------------------------------------
# Logging Setup
# --------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("app")

# --------------------------------------------
# Load Environment Variables
# --------------------------------------------
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")

if not groq_api_key:
    logger.warning("GROQ_API_KEY not set, will attempt to use Gemini fallback")
if not gemini_api_key:
    logger.warning("GEMINI_API_KEY not set, fallback will not be available")

# --------------------------------------------
# Initialize LLMs
# --------------------------------------------
llm_groq = ChatGroq(model="llama-3.3-70b-versatile", api_key=groq_api_key) if groq_api_key else None

def initialize_gemini():
    if not gemini_api_key:
        return None
    try:
        genai.configure(api_key=gemini_api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')
        logger.info("Initialized Gemini LLM")
        return model
    except Exception as e:
        logger.error(f"Failed to initialize Gemini LLM: {str(e)}")
        return None

llm_gemini = initialize_gemini()

# --------------------------------------------
# File Storage Configuration
# --------------------------------------------
STORAGE_DIR = "./storage"
SWAGGER_FILE = os.path.join(STORAGE_DIR, "swagger.json")
FRD_FILE = os.path.join(STORAGE_DIR, "frd.txt")
TEST_FILE = os.path.join(STORAGE_DIR, "test_script.py")
REPORT_FILE = os.path.join(STORAGE_DIR, "custom_report.json")

os.makedirs(STORAGE_DIR, exist_ok=True)

# --------------------------------------------
# JSON Encoder for datetime support
# --------------------------------------------
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

# --------------------------------------------
# Pydantic Models
# --------------------------------------------
class ScenarioInput(BaseModel):
    scenario_name: str
    flow_names: List[str]

class RunScenarioRequest(BaseModel):
    base_url: str
    scenario_details: ScenarioInput

class WorkflowState(BaseModel):
    swagger_doc: Dict[str, Any] = {}
    frd_text: str = ""
    base_url: str = ""
    scenarios: List[ScenarioInput] = []
    test_code: str = ""
    test_results: Dict[str, Any] = {}

# --------------------------------------------
# Helper Functions
# --------------------------------------------
def cleanup_storage(delete_all: bool = False):
    files_to_delete = [TEST_FILE] if not delete_all else [SWAGGER_FILE, FRD_FILE, TEST_FILE]
    for file_path in files_to_delete:
        if os.path.exists(file_path):
            try:
                os.unlink(file_path)
                logger.info(f"Deleted existing file: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to delete {file_path}: {str(e)}")

def save_to_storage(content, file_path, is_binary=False):
    mode = 'wb' if is_binary else 'w'
    encoding = None if is_binary else 'utf-8'
    with open(file_path, mode, encoding=encoding) as f:
        f.write(content)
    logger.info(f"Saved content to {file_path}")

def load_from_storage(file_path, is_binary=False):
    if not os.path.exists(file_path):
        raise ValueError(f"File not found: {file_path}")
    mode = 'rb' if is_binary else 'r'
    encoding = None if is_binary else 'utf-8'
    with open(file_path, mode, encoding=encoding) as f:
        return f.read()

# --------------------------------------------
# Gemini LLM Wrapper
# --------------------------------------------
def invoke_gemini(prompt_text: str, inputs: Dict[str, Any]) -> str:
    """Invoke Gemini LLM with the given prompt text and inputs."""
    if not llm_gemini:
        raise ValueError("Gemini LLM not initialized. Please set GEMINI_API_KEY.")
    
    try:
        prompt = prompt_text.format(**inputs)
        response = llm_gemini.generate_content(prompt)
        return response.text
    except Exception as e:
        logger.error(f"Gemini LLM invocation failed: {str(e)}")
        raise

# --------------------------------------------
# LangGraph Node Functions
# --------------------------------------------
def parse_swagger(state: WorkflowState) -> WorkflowState:
    logger.info("Loading stored Swagger document")
    try:
        swagger_content = load_from_storage(SWAGGER_FILE)
        state.swagger_doc = yaml.safe_load(swagger_content) if SWAGGER_FILE.endswith(('.yaml', '.yml')) else json.loads(swagger_content)
    except Exception as e:
        logger.error(f"Failed to load Swagger: {str(e)}")
        raise
    return state

def parse_frd(state: WorkflowState) -> WorkflowState:
    logger.info("Loading stored FRD text")
    try:
        state.frd_text = load_from_storage(FRD_FILE, is_binary=False)
    except Exception as e:
        logger.error(f"Failed to load FRD: {str(e)}")
        raise
    return state

def generate_scenarios_node(state: WorkflowState) -> WorkflowState:
    logger.info("Generating scenarios using LLM")
    try:
        prompt = ChatPromptTemplate.from_template("""
        You are an expert QA engineer specializing in API integration testing. Your task is to deeply analyze the provided Swagger (OpenAPI) specification and FRD (Functional Requirements Document) text to identify all endpoints, their dependencies, payloads, parameters, responses, and role-based access controls.

        **DETAILED ANALYSIS INSTRUCTIONS:**
        1. **Parse Swagger Thoroughly:**
           - List all paths (endpoints) with methods (GET, POST, PUT, DELETE, etc.).
           - For each endpoint: Extract required/optional parameters (path, query, header, cookie), request body schema (payload fields, types, constraints like required fields, enums), response schemas (status codes, body fields).
           - Identify authentication/authorization: Look for security schemes (e.g., API keys, JWT, OAuth), required headers (e.g., Authorization: Bearer {{token}}), roles/permissions in descriptions or security scopes.
           - Detect dynamic features: e.g., entities or actions restricted to certain user types, or role-based actions and permissions.

        2. **Parse FRD Text Thoroughly:**
           - Extract business flows, user stories, and feature-specific logic.
           - Identify dependencies between endpoints (e.g., a resource must exist before another can be created or referenced).
           - Note role-based conditions, preconditions, and sequencing of actions.
           - Detect order-sensitive or multi-user operations (e.g., creation → approval → update → retrieval).

        3. **Infer Dependencies and Flows:**
           - Construct logical flow sequences that mirror realistic user or system journeys.
           - Ensure each action follows its necessary prerequisites (registration before login, creation before update/delete, authentication before protected actions, etc.).
           - Maintain consistent context across actions — for example, use valid sessions or tokens throughout related sequences without breaking the flow or mixing unrelated user contexts.
           - If multiple users or roles exist, design flows to switch context only when logically required, keeping actions grouped coherently for each user or system entity.
           - Avoid illogical transitions (e.g., performing dependent actions under the wrong session, skipping re-authentication when switching users, or using resources that haven’t been created yet).
           - Capture error and negative flows where applicable (e.g., unauthorized access, invalid data).

        4. **Generate Scenarios:**
           - Create 3–5 comprehensive integration scenarios that represent end-to-end journeys.
           - Each scenario should have:
             - A meaningful "scenario_name"
             - An ordered "flow_names" array — a sequence of endpoint interactions with short explanatory notes describing what each step accomplishes and any key conditions.
           - Ensure each scenario forms a coherent, executable story that can be reproduced in a real testing pipeline.
           - Maintain dependency correctness (e.g., authentication → creation → dependent action → verification).
           - Preserve logical user/session continuity throughout.

        Swagger (OpenAPI spec):
        {swagger}

        FRD text:
        {frd}

        **OUTPUT FORMAT: Strictly Return ONLY valid JSON array, no extra text/markdown.**
        [
          {{
            "scenario_name": "Descriptive scenario name",
            "flow_names": [
              "POST /endpoint (Short explanation of what this step does and its purpose and what shouldn't need to do )",
              "GET /endpoint (Short explanation of what this retrieves or verifies and what shouldn't need to do)"
            ]
          }},
          ...
        ]
        """)

        inputs = {
            "swagger": json.dumps(state.swagger_doc, indent=2, cls=CustomJSONEncoder),
            "frd": state.frd_text
        }

        # Extract the prompt template string for Gemini
        prompt_text = prompt.messages[0].prompt.template

        # Try Groq first
        text_output = None
        if llm_groq:
            try:
                chain = prompt | llm_groq
                response = chain.invoke(inputs)
                text_output = response.content.strip()
                logger.info("Successfully generated scenarios with Groq LLM")
            except Exception as e:
                logger.warning(f"Groq LLM failed: {str(e)}, falling back to Gemini")

        # Fallback to Gemini if Groq fails or is not available
        if not text_output and llm_gemini:
            logger.info("Attempting scenario generation with Gemini LLM")
            text_output = invoke_gemini(prompt_text, inputs)

        if not text_output:
            raise ValueError("Both Groq and Gemini LLM failed to generate scenarios")

        logger.info(f"🌐 LLM Response Text Output: {text_output}")

        # Robust JSON extraction
        try:
            scenarios_data = json.loads(text_output)
        except json.JSONDecodeError:
            start_idx = text_output.find('[')
            end_idx = text_output.rfind(']') + 1
            if start_idx != -1 and end_idx != 0:
                json_str = text_output[start_idx:end_idx]
                try:
                    scenarios_data = json.loads(json_str)
                except json.JSONDecodeError as e:
                    logger.error(f"JSON parsing error after cleanup: {str(e)}")
                    logger.debug(f"Raw response content: {text_output}")
                    raise ValueError(f"Failed to parse cleaned LLM response as JSON: {str(e)}")
            else:
                logger.error(f"No valid JSON array found in response")
                logger.debug(f"Raw response content: {text_output}")
                raise ValueError("No valid JSON array found in LLM response")

        if not isinstance(scenarios_data, list):
            raise ValueError("LLM response is not a valid JSON list of scenarios")

        state.scenarios = [ScenarioInput(**s) for s in scenarios_data]
        logger.info(f"Generated {len(state.scenarios)} scenarios")
        return state

    except Exception as e:
        logger.error(f"Error generating scenarios: {str(e)}")
        raise ValueError(f"Scenario generation failed: {str(e)}")


def generate_pytest_code(state: WorkflowState) -> WorkflowState:
    logger.info("Generating pytest code via LLM")
    prompt = ChatPromptTemplate.from_template("""
You are an expert Python pytest developer for API integration testing. Generate comprehensive, runnable pytest code based on the analyzed Swagger, FRD, and provided scenario. Focus on dependency chaining, role-based auth, dynamic IDs, and flexible validation.

**DETAILED ANALYSIS RECAP (Use this to inform code):**
- Endpoints: From Swagger – paths, methods, params (path/query/header), payloads (request body schemas), responses (status codes, fields).
- Dependencies: Chain fixtures logically (e.g., register → login → create product (extract ID) → create order (use product_id)).
- Roles: If detected (e.g., admin for projects), include role assignment in flows; use headers/tokens; test role-specific successes/failures (e.g., 403 for unauthorized).
- Dynamic: Extract IDs from responses (e.g., product_id = response.json()["id"]); no hardcoding.
- FRD: Validate against requirements (e.g., expected fields, business rules like "order requires valid product").

**CRITICAL CODE REQUIREMENTS:**
1. **Imports & Setup:** import requests, pytest, json, random, string; session = requests.Session(); base_url.
2. **Fixtures (session scope, chained):**
   - user_credentials: Random email/username/password dict.
   - registered_user (if register exists): POST /auth/register with creds; return {{"response": r, "credentials": creds, "id": r.json().get("id") if 201}}.
   - auth_token: POST /auth/login with creds from registered_user or hardcoded test creds; add "Authorization": f"Bearer {{token}}" to session.headers; return {{"token": token, "response": r}}.
   - Role fixtures if needed: e.g., assign_role (POST /users/{{id}}/roles, depends on registered_user/auth_token); return role data.
   - created_product: POST /products with random payload (use schema: e.g., {{"name": random_string, "price": random_int}}); depends on auth_token; extract id; return {{"id": id, "response": r}}.
   - created_order: POST /orders with payload including {{"product_id": created_product["id"]}}; depends on created_product/auth_token; return {{"id": id, "response": r}}.
   - For projects/dynamic: e.g., created_project (POST /projects), assigned_role (update with role), then role-specific actions.
   - Handle failures: If status != expected (e.g., 201), log error, set id=None, but continue (for skips).
3. **Tests (Validate fixtures, no re-execution):**
   - def test_<flow>(relevant_fixtures): e.g., test_user_registration(registered_user): assert registered_user["response"].status_code == 201; validate json fields per schema/FRD.
   - For dependents: if prereq_id is None: append skip to report, pytest.skip("Prereq failed"); else: perform call with dynamic ID, validate.
   - Role tests: e.g., test_admin_update_project(auth_token, created_project): if admin_role, expect 200; else test 403.
   - Inject all needed fixtures as params (e.g., auth_token for headers).
4. **Reporting (Comprehensive, real-time):**
   - Globals: TEST_REPORT = [], FIXTURE_DETAILS = [], TOTAL_TESTS = PASSED_TESTS = FAILED_TESTS = 0, SCENARIO_FAILED = False.
   - In fixtures: After exec, append to FIXTURE_DETAILS {{"name": "registered_user", "status": "passed/failed", "data": sanitized_resp, "used_by": [test_names]}}; save_test_report().
   - In tests: At start, TOTAL_TESTS +=1; append to TEST_REPORT {{"test_name", "endpoint": flow_name, "payload": payload or None, "response": resp.json() or err, "status_code": or None, "expected_status": 201, "passed": True/False, "failure_reason": detailed msg or "Skipped: prereq failed"}}; if skip/fail: FAILED_TESTS +=1, SCENARIO_FAILED=True; ALWAYS call save_test_report() at end.
   - save_test_report(): Dump to "custom_report.json": {{"scenario_summary": scenario_name, "total_tests", "passed_tests": PASSED_TESTS, "failed_tests": FAILED_TESTS, "test_details": TEST_REPORT, "fixture_details": FIXTURE_DETAILS, "overall_status": "passed" if not SCENARIO_FAILED else "failed"}}; use json.dump(indent=4).
   - Session autouse fixture: yield; save_test_report() # Final backup.
5. **General:** Random data (no faker); handle 409 duplicates (randomize); flexible asserts (check key fields exist/types); session for auth persistence; timeout=10s.
6. Strictly follow the flow names and the description for each end point 

Scenario: {scenario_name}
Flows: {flow_names} (respect order/deps)
Base URL: {base_url}
Swagger: {swagger}
FRD: {frd}

**OUTPUT: ONLY runnable Python code, no markdown/extras/comments beyond essentials.**
""")

    scenario = state.scenarios[0] if state.scenarios else ScenarioInput(scenario_name="", flow_names=[])

    inputs = {
        "scenario_name": scenario.scenario_name,
        "flow_names": scenario.flow_names,
        "base_url": state.base_url,
        "swagger": json.dumps(state.swagger_doc, indent=2, cls=CustomJSONEncoder),
        "frd": state.frd_text[:3000]
    }

    # Extract the prompt template string for Gemini
    prompt_text = prompt.messages[0].prompt.template

    # Try Groq first
    code = None
    if llm_groq:
        try:
            chain = prompt | llm_groq
            response = chain.invoke(inputs)
            code = response.content.strip()
            logger.info("Successfully generated pytest code with Groq LLM")
        except Exception as e:
            logger.warning(f"Groq LLM failed for pytest code generation: {str(e)}, falling back to Gemini")

    # Fallback to Gemini if Groq fails or is not available
    if not code and llm_gemini:
        logger.info("Attempting pytest code generation with Gemini LLM")
        code = invoke_gemini(prompt_text, inputs)

    if not code:
        raise ValueError("Both Groq and Gemini LLM failed to generate pytest code")

    if code.startswith("```"):
        code = code.strip("`").replace("python", "").strip()

    state.test_code = code

    save_to_storage(code, TEST_FILE)
    logger.info("✅ Pytest code generated and saved")

    return state

# --------------------------------------------
# Execute Tests (No Changes Needed)
# --------------------------------------------
def execute_tests(state: WorkflowState) -> WorkflowState:
    logger.info("Executing generated pytest code...")
    
    try:
        client = None
        connection_methods = [
            lambda: docker.from_env(),
            lambda: docker.DockerClient(base_url='tcp://localhost:2375'),
            lambda: docker.DockerClient(base_url='npipe:////./pipe/docker_engine'),
            lambda: docker.DockerClient(base_url='unix://var/run/docker.sock')
        ]
        
        for i, method in enumerate(connection_methods):
            try:
                client = method()
                client.ping()
                logger.info(f"Successfully connected to Docker using method {i+1}")
                break
            except Exception as e:
                logger.warning(f"Docker connection method {i+1} failed: {str(e)}")
                continue
        
        if client is None:
            raise Exception("Docker not available, falling back to direct execution")
        
        logger.info("Using Docker for test execution...")
        return _execute_tests_docker(client, state)
        
    except ImportError:
        logger.warning("Docker module not available, falling back to direct pytest execution...")
        return _execute_tests_direct(state)
    except Exception as docker_error:
        logger.warning(f"Docker execution failed: {str(docker_error)}")
        logger.info("Falling back to direct pytest execution...")
        return _execute_tests_direct(state)

def _execute_tests_direct(state: WorkflowState) -> WorkflowState:
    """Execute pytest directly without Docker"""
    try:
        logger.info("Executing pytest directly...")
        
        # Change to the storage directory where the test file is located
        original_cwd = os.getcwd()
        os.chdir(STORAGE_DIR)
        
        try:
            # Run pytest with JSON report
            result = subprocess.run([
                'python', '-m', 'pytest', 
                'test_script.py', 
                '--json-report', 
                '--json-report-file=report.json',
                '-v'
            ], capture_output=True, text=True, timeout=300)
            
            # Load custom report
            custom_report = None
            if os.path.exists(REPORT_FILE):
                try:
                    with open(REPORT_FILE, 'r') as f:
                        custom_report = json.load(f)
                        logger.info(f"✅ Loaded custom report with {len(custom_report.get('test_details', []))} test details")
                except Exception as e:
                    logger.warning(f"Could not load custom report: {str(e)}")
            else:
                logger.warning(f"Custom report file not found at {REPORT_FILE}")
            
            # Load JSON report if it exists
            json_report = None
            report_file = os.path.join(STORAGE_DIR, 'report.json')
            if os.path.exists(report_file):
                try:
                    with open(report_file, 'r') as f:
                        json_report = json.load(f)
                except Exception as e:
                    logger.warning(f"Could not load JSON report: {str(e)}")
            
            state.test_results = {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "json_report": json_report,
                "custom_report": custom_report,
                "exit_code": result.returncode
            }
            
            logger.info(
                "Test execution results: success=%s, exit_code=%s, stdout=\n%s",
                state.test_results["success"],
                state.test_results["exit_code"],
                state.test_results["stdout"]
            )
            logger.info("✅ Test execution completed")
            
        finally:
            # Always restore the original working directory
            os.chdir(original_cwd)
            
    except subprocess.TimeoutExpired:
        logger.error("Test execution timed out after 5 minutes")
        state.test_results = {"success": False, "error": "Test execution timed out"}
    except Exception as e:
        logger.error(f"Error executing tests directly: {str(e)}")
        state.test_results = {"success": False, "error": str(e)}
        logger.info(
            "Test execution results: success=%s, error=%s",
            state.test_results["success"],
            state.test_results["error"]
        )
    
    return state

def _execute_tests_docker(client, state: WorkflowState) -> WorkflowState:
    try:
        # Build Docker image
        logger.info("Building Docker image for pytest execution...")
        image, build_logs = client.images.build(
            path=".",
            tag="pytest-runner",
            rm=True
        )
        logger.info("Docker image built successfully")
        
        # Run container in the storage directory
        logger.info("Running pytest in Docker container...")
        container = client.containers.run(
            image,
            detach=True,
            volumes={
                os.path.abspath(STORAGE_DIR): {'bind': '/app/storage', 'mode': 'rw'}
            },
            environment={
                'PYTHONPATH': '/app'
            },
            working_dir='/app/storage'  # Set working directory to storage
        )
        
        # Wait for container to complete and get logs
        result = container.wait()
        logs = container.logs().decode('utf-8')
        
        # Get custom report if it exists
        custom_report = None
        try:
            # Copy the custom report from container
            report_data, _ = container.get_archive('/app/storage/custom_report.json')
            if report_data:
                import tarfile
                import io
                
                # Extract the JSON report from tar archive
                tar = tarfile.open(fileobj=io.BytesIO(b''.join(report_data)))
                json_file = tar.extractfile('custom_report.json')
                if json_file:
                    custom_report = json.loads(json_file.read().decode('utf-8'))
                    logger.info(f"✅ Loaded custom report from Docker with {len(custom_report.get('test_details', []))} test details")
        except Exception as e:
            logger.warning(f"Could not retrieve custom report: {str(e)}")
        
        # Get JSON report if it exists
        json_report = None
        try:
            # Copy the JSON report from container
            report_data, _ = container.get_archive('/app/storage/report.json')
            if report_data:
                import tarfile
                import io
                
                # Extract the JSON report from tar archive
                tar = tarfile.open(fileobj=io.BytesIO(b''.join(report_data)))
                json_file = tar.extractfile('report.json')
                if json_file:
                    json_report = json.loads(json_file.read().decode('utf-8'))
        except Exception as e:
            logger.warning(f"Could not retrieve JSON report: {str(e)}")
        
        # Clean up container
        container.remove()
        
        state.test_results = {
            "success": result['StatusCode'] == 0,
            "stdout": logs,
            "stderr": "",
            "json_report": json_report,
            "custom_report": custom_report,
            "exit_code": result['StatusCode']
        }
        
        logger.info(
            "Test execution results: success=%s, exit_code=%s, stdout=\n%s",
            state.test_results["success"],
            state.test_results["exit_code"],
            state.test_results["stdout"]
        )
        logger.info("✅ Test execution completed")
        
    except docker.errors.DockerException as e:
        logger.error(f"Docker error executing tests: {str(e)}")
        state.test_results = {"success": False, "error": f"Docker error: {str(e)}"}
        logger.info(
            "Test execution results: success=%s, error=%s",
            state.test_results["success"],
            state.test_results["error"]
        )
    except Exception as e:
        logger.error(f"Error executing tests: {str(e)}")
        state.test_results = {"success": False, "error": str(e)}
        logger.info(
            "Test execution results: success=%s, error=%s",
            state.test_results["success"],
            state.test_results["error"]
        )

    return state

# --------------------------------------------
# LangGraph Workflow Setup
# --------------------------------------------
# Workflow for generating scenarios (full workflow)
scenario_workflow = StateGraph(WorkflowState)
scenario_workflow.add_node("parse_swagger", parse_swagger)
scenario_workflow.add_node("parse_frd", parse_frd)
scenario_workflow.add_node("generate_scenarios", generate_scenarios_node)

scenario_workflow.add_edge("parse_swagger", "parse_frd")
scenario_workflow.add_edge("parse_frd", "generate_scenarios")
scenario_workflow.add_edge("generate_scenarios", END)

scenario_workflow.set_entry_point("parse_swagger")
scenario_graph = scenario_workflow.compile()

# Workflow for running scenarios (skip scenario generation)
run_workflow = StateGraph(WorkflowState)
run_workflow.add_node("parse_swagger", parse_swagger)
run_workflow.add_node("parse_frd", parse_frd)
run_workflow.add_node("generate_pytest_code", generate_pytest_code)
run_workflow.add_node("execute_tests", execute_tests)

run_workflow.add_edge("parse_swagger", "parse_frd")
run_workflow.add_edge("parse_frd", "generate_pytest_code")
run_workflow.add_edge("generate_pytest_code", "execute_tests")
run_workflow.add_edge("execute_tests", END)

run_workflow.set_entry_point("parse_swagger")
run_graph = run_workflow.compile()

# --------------------------------------------
# FastAPI Endpoints
# --------------------------------------------
@app.options("/{path:path}")
async def options_handler(path: str):
    """Handle CORS preflight requests"""
    return Response(
        status_code=200,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Allow-Credentials": "true"
        }
    )

@app.post("/generate-scenarios")
async def generate_scenarios(
    swagger_doc: UploadFile = File(...),
    frd_file: UploadFile = File(...)
):
    logger.info("📥 Received request to generate scenarios")
    cleanup_storage(delete_all=True)  # Delete all files

    try:
        # Validate file types
        if swagger_doc.content_type not in ["application/json", "application/yaml", "text/yaml"]:
            logger.error(f"Invalid Swagger file type: {swagger_doc.content_type}")
            raise HTTPException(status_code=400, detail={"error": "Swagger file must be JSON or YAML"})
        
        if frd_file.content_type != "application/pdf":
            logger.error(f"Invalid FRD file type: {frd_file.content_type}")
            raise HTTPException(status_code=400, detail={"error": "FRD file must be a PDF"})

        # Save Swagger file
        swagger_bytes = await swagger_doc.read()
        if not swagger_bytes:
            logger.error("Swagger file is empty")
            raise HTTPException(status_code=400, detail={"error": "Swagger file is empty"})
        save_to_storage(swagger_bytes, SWAGGER_FILE, is_binary=True)
        logger.info(f"Swagger file saved, size: {len(swagger_bytes)} bytes")

        # Save FRD file temporarily to disk
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            frd_bytes = await frd_file.read()
            if not frd_bytes:
                logger.error("FRD file is empty")
                raise HTTPException(status_code=400, detail={"error": "FRD file is empty"})
            temp_pdf.write(frd_bytes)
            temp_pdf_path = temp_pdf.name
            logger.info(f"FRD file saved temporarily to {temp_pdf_path}, size: {len(frd_bytes)} bytes")

        # Process PDF with pdfplumber
        try:
            with pdfplumber.open(temp_pdf_path) as pdf:
                frd_text = "\n".join(page.extract_text() or "" for page in pdf.pages)
            if not frd_text.strip():
                logger.warning("Extracted FRD text is empty")
                raise HTTPException(status_code=400, detail={"error": "No text could be extracted from the PDF"})
            save_to_storage(frd_text, FRD_FILE)
            logger.info(f"FRD text extracted and saved, length: {len(frd_text)} characters")
        except Exception as e:
            logger.error(f"Failed to process PDF: {str(e)}")
            raise HTTPException(status_code=400, detail={"error": f"Invalid PDF file: {str(e)}"})
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_pdf_path)
                logger.info(f"Deleted temporary PDF file: {temp_pdf_path}")
            except Exception as e:
                logger.warning(f"Failed to delete temporary PDF file: {str(e)}")

        # Run the scenario generation workflow
        state = WorkflowState()
        result_dict = scenario_graph.invoke(state)
        result = WorkflowState(**result_dict)

        body = {
            "status": "success",
            "scenarios": [{"scenario_name": s.scenario_name, "flow_names": s.flow_names} for s in result.scenarios]
        }
        return Response(content=json.dumps(body, cls=CustomJSONEncoder), media_type="application/json")

    except Exception as e:
        logger.error(f"Error in /generate-scenarios: {str(e)}")
        raise HTTPException(status_code=400, detail={"error": str(e)})

@app.post("/run-scenario")
async def run_scenario(request_data: RunScenarioRequest):
    logger.info("📥 Received request to run scenario")
    cleanup_storage(delete_all=False)  # Delete only test script and report

    try:
        # Verify that Swagger and FRD files exist
        if not os.path.exists(SWAGGER_FILE):
            raise ValueError("Swagger file not found. Please call /generate-scenarios first.")
        if not os.path.exists(FRD_FILE):
            raise ValueError("FRD file not found. Please call /generate-scenarios first.")

        # Initialize state with provided scenario details
        state = WorkflowState(
            base_url=request_data.base_url,
            scenarios=[request_data.scenario_details]
        )

        # Use the run workflow which skips scenario generation
        result_dict = run_graph.invoke(state)
        result = WorkflowState(**result_dict)

        # Load and return the custom_report.json content directly
        if os.path.exists(REPORT_FILE):
            try:
                with open(REPORT_FILE, 'r') as f:
                    custom_report = json.load(f)
                    logger.info(f"✅ Returning custom report with {len(custom_report.get('test_details', []))} test details")
                    return Response(content=json.dumps(custom_report, cls=CustomJSONEncoder), media_type="application/json")
            except Exception as e:
                logger.error(f"Error loading custom report: {str(e)}")
                raise HTTPException(status_code=500, detail={"error": f"Failed to load custom report: {str(e)}"})
        else:
            logger.error(f"Custom report file not found at {REPORT_FILE}")
            raise HTTPException(status_code=404, detail={"error": "Custom report file not found"})

    except Exception as e:
        logger.error(f"Error in /run-scenario: {str(e)}")
        raise HTTPException(status_code=400, detail={"error": str(e)})

# --------------------------------------------
# Run Server
# --------------------------------------------
if __name__ == "__main__":
    import uvicorn
    logger.info("🚀 Starting FastAPI server on http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
