import time
from typing import Dict, Any, Callable
import random

class DependencyMediator:
    """
    The DependencyMediator class is responsible for managing the flow of data
    and interactions between different components of the AutoPaaS-X system,
    especially between AI models (LSTM, Q-learning) and other modules
    like the scheduler or API.

    It acts as a central hub to ensure that outputs from one component
    are correctly fed as inputs to another, and to handle any necessary
    data transformations or asynchronous operations.
    """

    def __init__(self):
        """
        Initializes the DependencyMediator.
        Stores references to registered components and their methods.
        """
        self._components: Dict[str, Any] = {}
        self._data_store: Dict[str, Any] = {} # A simple in-memory data store for shared data
        print("DependencyMediator initialized.")

    def register_component(self, name: str, component_instance: Any):
        """
        Registers a component with the mediator.

        Args:
            name (str): A unique name for the component (e.g., "lstm_predictor", "q_agent").
            component_instance (Any): An instance of the component class.
        """
        if name in self._components:
            print(f"Warning: Component '{name}' is already registered. Overwriting.")
        self._components[name] = component_instance
        print(f"Component '{name}' registered successfully.")

    def get_component(self, name: str):
        """
        Retrieves a registered component instance.

        Args:
            name (str): The name of the component.

        Returns:
            Any: The component instance, or None if not found.
        """
        return self._components.get(name)

    def store_data(self, key: str, data: Any):
        """
        Stores data in the mediator's internal data store.

        Args:
            key (str): The key to store the data under.
            data (Any): The data to be stored.
        """
        self._data_store[key] = data
        # print(f"Data stored under key: '{key}'")

    def retrieve_data(self, key: str):
        """
        Retrieves data from the mediator's internal data store.

        Args:
            key (str): The key for the data.

        Returns:
            Any: The retrieved data, or None if the key does not exist.
        """
        return self._data_store.get(key)

    def mediate_prediction_to_scaling_decision(self,
                                               app_characteristics: Dict[str, Any],
                                               lstm_predictor_name: str,
                                               q_agent_name: str,
                                               state_transform_func: Callable[[float, float], int]):
        """
        Mediates the process from application characteristics to a VM scaling decision.
        1. Calls LSTM predictor to estimate CPU/RAM needs.
        2. Transforms these predictions into a state for the Q-learning agent.
        3. Calls Q-learning agent to get a scaling action.

        Args:
            app_characteristics (Dict[str, Any]): Dictionary containing OS, architecture,
                                                  base_image_size_gb.
            lstm_predictor_name (str): The name of the LSTM predictor component.
            q_agent_name (str): The name of the Q-learning agent component.
            state_transform_func (Callable): A function to transform predicted CPU/RAM
                                             into an integer state for the Q-agent.
                                             Signature: `(predicted_cpu: float, predicted_ram: float) -> int`.

        Returns:
            tuple: (predicted_cpu_cores, predicted_ram_gb, scaling_action_id, scaling_action_name)
                   or (None, None, None, None) if mediation fails.
        """
        lstm_predictor = self.get_component(lstm_predictor_name)
        q_agent = self.get_component(q_agent_name)

        if not lstm_predictor or not hasattr(lstm_predictor, 'predict'):
            print(f"Error: LSTM Predictor '{lstm_predictor_name}' not registered or missing 'predict' method.")
            return None, None, None, None
        if not q_agent or not hasattr(q_agent, 'choose_action'):
            print(f"Error: Q-Learning Agent '{q_agent_name}' not registered or missing 'choose_action' method.")
            return None, None, None, None

        os_type = app_characteristics.get('os')
        architecture = app_characteristics.get('architecture')
        base_image_size_gb = app_characteristics.get('base_image_size_gb')

        if not all([os_type, architecture, base_image_size_gb is not None]):
            print("Error: Missing required application characteristics for prediction.")
            return None, None, None, None

        try:
            # 1. Predict CPU/RAM needs
            predicted_cpu, predicted_ram = lstm_predictor.predict(
                os_type, architecture, base_image_size_gb
            )
            print(f"Mediator: Predicted CPU: {predicted_cpu}, RAM: {predicted_ram}")
            self.store_data("predicted_resources", {"cpu": predicted_cpu, "ram": predicted_ram})

            # 2. Transform prediction into a state for Q-agent
            current_state = state_transform_func(predicted_cpu, predicted_ram)
            print(f"Mediator: Transformed predicted resources to state: {current_state} ({q_agent.get_state_name(current_state)})")
            self.store_data("current_system_state", current_state)

            # 3. Get scaling action from Q-learning agent
            scaling_action_id = q_agent.choose_action(current_state)
            scaling_action_name = q_agent.get_action_name(scaling_action_id)
            print(f"Mediator: Q-Learning Agent chose action: {scaling_action_name} ({scaling_action_id})")
            self.store_data("scaling_action", {"id": scaling_action_id, "name": scaling_action_name})

            return predicted_cpu, predicted_ram, scaling_action_id, scaling_action_name

        except Exception as e:
            print(f"Error during prediction to scaling mediation: {e}")
            return None, None, None, None

# Example Usage (for testing purposes)
if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    import os
    from lstm_predictor import LSTMPredictor # Import from same directory
    from q_learning import QLearningAgent # Import from same directory

    # Ensure data directory exists and dummy data is present for LSTM
    # When running this script directly from 'AutoPaaS-X/ai', 'data' is at '../data'
    os.makedirs("../data", exist_ok=True)
    dummy_data_path = "../data/historical_deployments.csv"
    if not os.path.exists(dummy_data_path):
        dummy_data = """os,architecture,base_image_size_gb,cpu_cores_needed,ram_gb_needed
Linux,64-bit,1.2,1,2
Linux,64-bit,1.5,2,4
Windows,64-bit,3.0,2,6
Linux,32-bit,0.8,0,0,1,0,1,1
Linux,64-bit,2.0,3,8
Windows,64-bit,2.8,2,5
Linux,64-bit,1.0,1,2
Linux,64-bit,1.8,3,7
Windows,32-bit,2.5,1,4
Linux,64-bit,1.3,2,3
Linux,64-bit,1.7,3,6
Windows,64-bit,3.2,4,10
"""
        with open(dummy_data_path, "w") as f:
            f.write(dummy_data)
        print(f"Created dummy data at {dummy_data_path}")

    # Initialize components
    lstm_predictor_instance = LSTMPredictor(epochs=10, batch_size=4) # Smaller epochs for faster demo
    lstm_predictor_instance.train(data_path=dummy_data_path)

    q_agent_instance = QLearningAgent(num_states=3, num_actions=3)
    # Train the Q-agent with a few episodes for demonstration
    print("\nTraining Q-Learning Agent for demo...")
    for episode in range(100): # Reduced episodes for quick demo
        current_state = random.randint(0, 2)
        action = q_agent_instance.choose_action(current_state)
        # Simulate next state and reward (simplified)
        next_state = current_state # For simplicity, assume holding state for demo training
        reward = 5.0 if current_state == 1 and action == 1 else -1.0 # Reward for optimal state and hold
        q_agent_instance.learn(current_state, action, reward, next_state)
        q_agent_instance.decay_exploration_rate(episode)
    print("Q-Learning Agent training complete for demo.")


    mediator = DependencyMediator()
    mediator.register_component("my_lstm_predictor", lstm_predictor_instance)
    mediator.register_component("my_q_agent", q_agent_instance)

    # Define the state transformation function
    def transform_resources_to_state(cpu_cores: float, ram_gb: float) -> int:
        """
        Example transformation:
        - If CPU < 1.5 and RAM < 3: Underutilized (State 0)
        - If 1.5 <= CPU < 3.5 and 3 <= RAM < 7: Optimal (State 1)
        - If CPU >= 3.5 or RAM >= 7: Overutilized (State 2)
        """
        if cpu_cores < 1.5 and ram_gb < 3:
            return 0 # Underutilized
        elif (1.5 <= cpu_cores < 3.5) and (3 <= ram_gb < 7):
            return 1 # Optimal
        else:
            return 2 # Overutilized

    print("\n--- Mediating a deployment request ---")
    app_req_1 = {
        "os": "Linux",
        "architecture": "64-bit",
        "base_image_size_gb": 1.8
    }
    cpu_1, ram_1, action_id_1, action_name_1 = mediator.mediate_prediction_to_scaling_decision(
        app_req_1, "my_lstm_predictor", "my_q_agent", transform_resources_to_state
    )
    if cpu_1 is not None:
        print(f"\nResult for App 1: Predicted CPU={cpu_1}, RAM={ram_1}, Scaling Action='{action_name_1}'")

    print("\n--- Mediating another deployment request ---")
    app_req_2 = {
        "os": "Windows",
        "architecture": "64-bit",
        "base_image_size_gb": 3.5
    }
    cpu_2, ram_2, action_id_2, action_name_2 = mediator.mediate_prediction_to_scaling_decision(
        app_req_2, "my_lstm_predictor", "my_q_agent", transform_resources_to_state
    )
    if cpu_2 is not None:
        print(f"\nResult for App 2: Predicted CPU={cpu_2}, RAM={ram_2}, Scaling Action='{action_name_2}'")
