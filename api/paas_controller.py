from pathlib import Path
from flask import Flask, request, jsonify, render_template
import os
import sys
import datetime
import random
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, Callable, List, Union
from flask_cors import CORS # Import CORS
import subprocess
# Add parent directory to path to allow imports from other modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import components from their respective directories
from ai.lstm_predictor import LSTMPredictor
from ai.q_learning import QLearningAgent
from ai.dependency_mediator import DependencyMediator
from serverless.cronjob_generator import CronJobGenerator
from sdn.throttler import Throttler
from scheduler.solar_scheduler import SolarScheduler
from serverless.deployment_generator import DeploymentGenerator

app = Flask(__name__, template_folder='templates')
CORS(app) # Enable CORS for all routes and origins

# --- Global Instances of Components ---
lstm_predictor: Union[LSTMPredictor, None] = None
q_agent: Union[QLearningAgent, None] = None
dependency_mediator: Union[DependencyMediator, None] = None
cronjob_generator: Union[CronJobGenerator, None] = None
throttler: Union[Throttler, None] = None
solar_scheduler: Union[SolarScheduler, None] = None
deployment_generator: Union[DeploymentGenerator, None] = None
deployment_generator = DeploymentGenerator()


def initialize_components():
    """
    Initializes all core components of AutoPaaS-X.
    This function should be called once when the application starts.
    """
    global lstm_predictor, q_agent, dependency_mediator, \
           cronjob_generator, throttler, solar_scheduler

    print("Initializing AutoPaaS-X components...")

    # Ensure data directory exists for LSTM and cronjob manifests
    os.makedirs("../data", exist_ok=True)
    os.makedirs("../generated_k8s_manifests", exist_ok=True) # For generated YAMLs
    historical_data_path = "../data/historical_deployments.csv"

    # AI Components
    try:
        lstm_predictor = LSTMPredictor(epochs=20, batch_size=8)
        lstm_predictor.train(data_path=historical_data_path)
        print("LSTMPredictor initialized and trained.")
    except Exception as e:
        print(f"Error initializing LSTMPredictor: {e}")
        lstm_predictor = None

    NUM_LOAD_STATES = 3
    NUM_SCALE_ACTIONS = 3
    q_agent = QLearningAgent(num_states=NUM_LOAD_STATES, num_actions=NUM_SCALE_ACTIONS)
    print("Training Q-Learning Agent for initial state...")
    for episode in range(500):
        current_state = random.randint(0, NUM_LOAD_STATES - 1)
        action = q_agent.choose_action(current_state)
        next_state = current_state
        reward = 5.0 if current_state == 1 and action == 1 else -1.0
        q_agent.learn(current_state, action, reward, next_state)
        q_agent.decay_exploration_rate(episode)
    print("QLearningAgent initialized and trained for initial state.")

    dependency_mediator = DependencyMediator()
    if lstm_predictor:
        dependency_mediator.register_component("lstm_predictor", lstm_predictor)
    dependency_mediator.register_component("q_agent", q_agent)
    print("DependencyMediator initialized.")

    # Serverless Component (CronJob Generator)
    cronjob_generator = CronJobGenerator(template_path="../manifests/cronjob.yaml")
    print("CronJobGenerator initialized.")

    # SDN Component
    throttler = Throttler()
    print("Throttler initialized.")

    #Deployment Component

    # Scheduler Component
    def custom_solar_predictor_for_scheduler(dt: datetime.datetime) -> float:
        hour = dt.hour
        if 8 <= hour <= 17:
            return 80.0 + (5 * np.sin(np.pi * (hour - 8) / 9))
        return 10.0
    solar_scheduler = SolarScheduler(solar_prediction_func=custom_solar_predictor_for_scheduler)
    dependency_mediator.register_component("solar_scheduler", solar_scheduler)
    print("SolarScheduler initialized.")

    print("All AutoPaaS-X components initialized.")

# Call initialization once when the app starts
with app.app_context():
    initialize_components()

# --- Helper Functions for API endpoints ---

def transform_predicted_resources_to_state(cpu_cores: float, ram_gb: float) -> int:
    """
    Transforms predicted CPU and RAM into an integer state for the Q-learning agent.
    This mapping defines the "resource utilization" states.
    """
    if cpu_cores < 1.5 and ram_gb < 3:
        return 0 # Underutilized
    elif (1.5 <= cpu_cores < 3.5) and (3 <= ram_gb < 7):
        return 1 # Optimal
    else:
        return 2 # Overutilized

# --- API Endpoints ---

@app.route('/')
def serve_dashboard():
    """
    Serves the main dashboard HTML file.
    """
    return render_template('index.html')

@app.route('/health') # Moved health check to a dedicated endpoint
def health_check():
    """
    Health check endpoint for the API.
    """
    return jsonify({"status": "AutoPaaS-X API is running!", "version": "1.0"})

@app.route('/deploy_application', methods=['POST'])
def deploy_application():
    data = request.get_json()
    required_fields = ["app_name", "os", "architecture", "base_image_size_gb"]
    if not all(field in data for field in required_fields):
        return jsonify({"error": f"Missing required fields. Required: {required_fields}"}), 400

    app_name = data['app_name']
    os_type = data['os']
    architecture = data['architecture']
    base_image_size_gb = data['base_image_size_gb']

    image = "python:3.9-slim"  # lightweight image for dummy container
    initial_replicas = 1
    port = 80
    labels = {"app-tier": "web", "managed-by": "autopaa-x"}
    annotations = {"autopaa-x.com/predicted-by": "lstm-qlearning"}

    if not lstm_predictor or not q_agent or not dependency_mediator or not cronjob_generator or not deployment_generator:
        return jsonify({"error": "Core components not initialized."}), 500

    try:
        predicted_cpu, predicted_ram, scaling_action_id, scaling_action_name = \
            dependency_mediator.mediate_prediction_to_scaling_decision(
                app_characteristics={
                    "os": os_type,
                    "architecture": architecture,
                    "base_image_size_gb": base_image_size_gb
                },
                lstm_predictor_name="lstm_predictor",
                q_agent_name="q_agent",
                state_transform_func=transform_predicted_resources_to_state
            )

        if predicted_cpu is None:
            return jsonify({"error": "Prediction failed."}), 500

        final_replicas = initial_replicas
        if scaling_action_name == "scale_up":
            final_replicas = max(initial_replicas + 1, 1)
        elif scaling_action_name == "scale_down":
            final_replicas = max(initial_replicas - 1, 1)

        # Generate Deployment YAML
        deployment_manifest = deployment_generator.generate_deployment(
            name=f"{app_name}-deployment",
            image=image,
            cpu_cores=predicted_cpu,
            ram_gb=predicted_ram,
            replicas=final_replicas,
            port=port,
            labels=labels,
            annotations=annotations,
            command=[
                "/bin/sh",
                "-c",
                f"fallocate -l {base_image_size_gb}G /tmp/dummy.img && python3 -m http.server 80"
            ]
        )
        deployment_path = deployment_generator.save_deployment_to_file(deployment_manifest, app_name)
        print(deployment_path)
        # Generate Warmup CronJob YAML
        warmup_url = f"http://{app_name}-service.default.svc.cluster.local/health"


        warmup_cronjob_manifest = cronjob_generator.generate_cronjob(
            name=f"{app_name}-warmup-cronjob",
            schedule="0 * * * *",
            image="curlimages/curl:latest",
            command=["/bin/sh", "-c", f"curl -s -o /dev/null -w '%{{http_code}}' {warmup_url}"],
            labels={"app-tier": "warmup", "managed-by": "autopaa-x"},
            annotations={"autopaa-x.com/warmup-for": app_name}
        )
        warmup_path = cronjob_generator.save_cronjob_to_file(warmup_cronjob_manifest, "../generated_k8s_manifests")
        # Normalize file paths
        print(warmup_path)
        deployment_path = Path(deployment_path).resolve().as_posix()
        # print(deployment_path)
        warmup_path = Path(warmup_path).resolve().as_posix()
        print(warmup_path)

        # Apply both YAMLs via kubectl
        deployment_status = subprocess.run(["kubectl", "apply", "-f", deployment_path], capture_output=True, text=True)
        warmup_status = subprocess.run(["kubectl", "apply", "-f", warmup_path], capture_output=True, text=True)

        return jsonify({
            "message": f"App '{app_name}' deployed successfully with warmup job.",
            "predicted_resources": {
                "cpu_cores": predicted_cpu,
                "ram_gb": predicted_ram
            },
            "scaling_action": {
                "id": scaling_action_id,
                "name": scaling_action_name,
                "final_replicas": final_replicas
            },
            "deployment_output": deployment_status.stdout,
            "deployment_error": deployment_status.stderr,
            "warmup_output": warmup_status.stdout,
            "warmup_error": warmup_status.stderr,
            "info": {
                "deployment_yaml": deployment_path,
                "warmup_yaml": warmup_path,
                "note": "You can inspect and edit YAMLs in 'generated_k8s_manifests/'"
            }
        }), 200

    except Exception as e:
        print(f"Error during deployment: {e}")
        return jsonify({"error": f"Failed to deploy: {str(e)}"}), 500

@app.route('/monitor_traffic_and_qos', methods=['POST'])
def monitor_traffic_and_qos():
    """
    Endpoint for Phase 2: SDN-Based Tenant Isolation.
    Receives real-time network traffic data per tenant,
    uses the Throttler to decide and apply QoS rules.
    """
    data = request.get_json()
    required_fields = ["tenant_id", "bandwidth_mbps", "latency_ms"]
    if not all(field in data for field in required_fields):
        return jsonify({"error": f"Missing required fields. Required: {required_fields}"}), 400

    tenant_id = data['tenant_id']
    bandwidth_mbps = data['bandwidth_mbps']
    latency_ms = data['latency_ms']

    if not throttler:
        return jsonify({"error": "SDN Throttler component not initialized."}), 500

    try:
        throttling_decision = {"action": "no_action", "reason": "Traffic within acceptable limits."}
        rate_limit = None

        MAX_BANDWIDTH_THRESHOLD = 1000 # Mbps
        LATENCY_THRESHOLD_MS = 100 # ms
        DEFAULT_THROTTLE_RATE = 500 # Mbps

        if bandwidth_mbps > MAX_BANDWIDTH_THRESHOLD or latency_ms > LATENCY_THRESHOLD_MS:
            throttling_decision["action"] = "throttle"
            throttling_decision["rate_limit_mbps"] = DEFAULT_THROTTLE_RATE
            rate_limit = DEFAULT_THROTTLE_RATE
            throttling_decision["reason"] = "Exceeded bandwidth or latency thresholds."
            if bandwidth_mbps > MAX_BANDWIDTH_THRESHOLD and latency_ms > LATENCY_THRESHOLD_MS:
                 throttling_decision["reason"] = "Exceeded both bandwidth and latency thresholds."
                 throttling_decision["rate_limit_mbps"] = DEFAULT_THROTTLE_RATE * 0.8
                 rate_limit = DEFAULT_THROTTLE_RATE * 0.8


        response_message = f"Traffic for tenant '{tenant_id}' monitored. Decision: {throttling_decision['action']}. Reason: {throttling_decision['reason']}"

        if throttling_decision["action"] == "throttle":
            throttler.apply_rate_limit(tenant_id, rate_limit)
            response_message += f" Applied rate limit: {rate_limit} Mbps."
        elif throttling_decision["action"] == "no_action":
            current_rule = throttler.get_flow_status(tenant_id)
            if current_rule and current_rule.get("rate_limit_mbps") is not None:
                throttler.remove_rate_limit(tenant_id)
                response_message += " Previous throttling rule removed as traffic is now acceptable."

        return jsonify({"message": response_message, "throttling_decision": throttling_decision}), 200

    except Exception as e:
        print(f"Error during traffic monitoring and QoS enforcement: {e}")
        return jsonify({"error": f"Failed to process traffic and QoS: {str(e)}"}), 500

@app.route('/get_throttling_rules', methods=['GET'])
def get_throttling_rules():
    """
    Endpoint to get all active SDN throttling rules.
    """
    if not throttler:
        return jsonify({"error": "SDN Throttler component not initialized."}), 500
    return jsonify(throttler.get_all_throttling_rules()), 200

@app.route('/remove_throttling_rule', methods=['POST'])
def remove_throttling_rule():
    """
    Endpoint to manually remove a specific SDN throttling rule.
    Expects a JSON body with 'tenant_id'.
    """
    data = request.get_json()
    if not data or 'tenant_id' not in data:
        return jsonify({"error": "Missing 'tenant_id' in request body."}), 400

    tenant_id = data['tenant_id']
    if not throttler:
        return jsonify({"error": "SDN Throttler component not initialized."}), 500

    try:
        success = throttler.remove_rate_limit(tenant_id)
        if success:
            return jsonify({"message": f"Throttling rule for tenant '{tenant_id}' removed."}), 200
        else:
            return jsonify({"message": f"No throttling rule found for tenant '{tenant_id}' to remove."}), 404
    except Exception as e:
        return jsonify({"error": f"Failed to remove throttling rule: {str(e)}"}), 500

if __name__ == '__main__':
    os.makedirs("../data", exist_ok=True)
    os.makedirs("../generated_k8s_manifests", exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000)
