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
    with open(file_path, mode) as f:
        f.write(content)
    logger.info(f"Saved content to {file_path}")

def load_from_storage(file_path, is_binary=False):
    if not os.path.exists(file_path):
        raise ValueError(f"File not found: {file_path}")
    mode = 'rb' if is_binary else 'r'
    with open(file_path, mode) as f:
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
        You are an expert QA engineer. Given this Swagger JSON and FRD text,
        generate integration test scenarios. Return **only valid JSON** with scenario name and flow names.
        Do not include markdown, code fences, or any additional text outside the JSON structure.

        Ensure flow names are in a logical, sequential order based on dependencies. For example:
        - User-related flows: "POST /auth/signup" must come before "POST /auth/login".
        - Resource creation (e.g., "POST /products") requires authentication, so "POST /auth/login" must precede it.
        - Retrieval or modification (e.g., "GET /products/:id", "PUT /products/:id") follows creation.

        Analyze the Swagger and FRD to determine dependencies and order flows accordingly.
        Importantly: If there is a login endpoint, there must be a register endpoint that comes before it.

        Swagger (OpenAPI spec):
        {swagger}

        FRD text:
        {frd}

        Format:
        [
          {{
            "scenario_name": "string",
            "flow_names": ["string"]
          }}
        ]
        """)

        inputs = {
            "swagger": json.dumps(state.swagger_doc, indent=2, cls=CustomJSONEncoder),
            "frd": state.frd_text[:3000]
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
    Generate pytest code for the following API testing scenario using the requests library.
    Validate status codes and JSON structure based on provided Swagger and FRD.
    Ensure test steps follow the sequential order of flow names as provided, respecting dependencies
    (e.g., user registration before login, login before resource creation).

    CRITICAL REQUIREMENTS:
    1. Use **session-scoped pytest fixtures** (scope="session") for shared data (e.g., user credentials, auth token, created resource IDs) to ensure a single user registration and login is performed once per test session.
    2. Handle duplicate registration gracefully (use random emails or expect 409 status).
    3. Use flexible response validation instead of hardcoded assertions.
    4. Generate clean, runnable pytest code without any markdown formatting.
    5. Use proper pytest fixtures for shared state management.
    6. Handle authentication token sharing between tests using a session-scoped `auth_token` fixture.
    7. Use random data generation to avoid conflicts (emails, usernames, etc.) using Python's `random` and `string` modules.
    8. Validate response structure flexibly - check for expected fields without strict assertions.
    9. Handle both success and expected error cases appropriately.
    10. **If both registration and login endpoints are available:**
        - STRICTLY DO NOT access or rely on the password from the registration API response.
        - Create a **session-scoped** fixture `user_credentials` to generate and store random email, username, and password using `random` and `string`.
        - Create a **session-scoped** fixture `registered_user` that uses the stored credentials to make the signup request, then returns a dictionary with both the API response data AND the original stored credentials.
        - Create a **session-scoped** fixture `auth_token` that uses the stored email and password from `registered_user` to make the login request and returns the token.
        - Ensure all dependent tests (e.g., resource creation, retrieval, update, delete) use the `auth_token` fixture for authentication.
    11. Do NOT use the `faker` library. Use Python's `random` and `string` modules.
    12. **COMPREHENSIVE TEST REPORTING - CRITICAL REQUIREMENT**:
        - Create global variables:
            - TEST_REPORT = []  # store test details
            - SCENARIO_FAILED = False  # track overall scenario status
            - FIXTURE_DETAILS = []  # store fixture usage details
            - TOTAL_TESTS = 0
            - PASSED_TESTS = 0
            - FAILED_TESTS = 0
        - For each test function:
            a. Increment TOTAL_TESTS at the start.
            b. **ALWAYS append to TEST_REPORT and FIXTURE_DETAILS, even if the test skips or fails due to dependencies** - do not place these inside conditional `if` blocks.
            c. If a prerequisite fixture fails (e.g., no credentials or token), use `pytest.skip(f"Skipped: {reason}")` to explicitly skip the test, log the skip reason in `failure_reason`, set `passed=False`, increment `FAILED_TESTS`, and set `SCENARIO_FAILED=True`.
            d. For successful cases, set `passed=True` and increment `PASSED_TESTS`.
            e. Append a detailed dictionary to TEST_REPORT with:
                - test_name
                - endpoint
                - payload (or None if not applicable)
                - response (or error message if skipped/failed)
                - status_code (or None if skipped)
                - expected_status_code
                - passed
                - failure_reason (e.g., API error message or skip reason)
                - expected_response
            f. For each fixture used in the test, append a dictionary to FIXTURE_DETAILS with:
                - fixture_name
                - used_by_tests (list of test names using it)
                - data (sanitized if sensitive)
            g. **CRITICAL: For the registration test (e.g., test_register_user), make it DEPEND ON the `registered_user` fixture** - do NOT perform a separate signup request in the test itself to avoid duplicates. Instead, validate the fixture's response (status, structure) and log it. If the fixture failed, skip/log the test.
            h. Ensure dependent tests (login, create, etc.) check fixtures first: if missing (e.g., no token), skip and log without making API calls.
        - Create a function save_test_report() that:
            a. Creates a comprehensive report with:
                - scenario_summary
                - total_tests
                - passed_tests
                - failed_tests
                - test_details
                - fixture_details
                - overall_status ("passed" if SCENARIO_FAILED is False else "failed")
            b. Saves this report to "custom_report.json".
        - Create a session-scoped pytest fixture with autouse=True to ensure the report is saved after all tests complete:
            @pytest.fixture(scope="session", autouse=True)
            def session_cleanup():
                yield
                save_test_report()

    Scenario Name: {scenario_name}
    Flow Names: {flow_names}
    Base URL: {base_url}
    Swagger: {swagger}
    FRD: {frd}

    Return only runnable Python code without any markdown or additional text.
""")


    scenario = state.scenarios[0] if state.scenarios else state.test_results.get("scenario_details")

    inputs = {
        "scenario_name": scenario.scenario_name,
        "flow_names": scenario.flow_names,
        "base_url": state.base_url,
        "swagger": json.dumps(state.swagger_doc, indent=2, cls=CustomJSONEncoder),
        "frd": state.frd_text[:3000],
        "reason": "prerequisite fixture failed"
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
    logger.info("Generated pytest code:\n%s", code)

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