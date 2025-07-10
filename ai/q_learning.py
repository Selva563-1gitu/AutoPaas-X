import numpy as np
import random
from typing import Dict, Any, Union # Ensure Union is imported for type hints

class QLearningAgent:
    """
    A Q-learning agent for deciding VM scaling actions (scale_up/hold/scale_down).
    This agent learns optimal actions based on the current system state (e.g., load,
    resource utilization, predicted demand) to optimize resource allocation.

    Actions:
        0: scale_down
        1: hold
        2: scale_up

    States:
        The state space needs to be defined based on relevant system metrics.
        For this example, let's consider states based on resource utilization levels:
        0: Underutilized (e.g., CPU < 20%, RAM < 30%)
        1: Optimal (e.g., CPU 20-70%, RAM 30-80%)
        2: Overutilized (e.g., CPU > 70%, RAM > 80%)
    """

    def __init__(self, num_states=3, num_actions=3, learning_rate=0.1, discount_factor=0.9,
                 exploration_rate=1.0, min_exploration_rate=0.01, exploration_decay_rate=0.001):
        """
        Initializes the Q-learning agent.

        Args:
            num_states (int): The total number of possible states in the environment.
            num_actions (int): The total number of possible actions the agent can take.
            learning_rate (float): Alpha (α) - The extent to which our Q-values are updated.
            discount_factor (float): Gamma (γ) - The importance of future rewards.
            exploration_rate (float): Epsilon (ε) - The initial rate of exploration (taking random actions).
            min_exploration_rate (float): The minimum exploration rate.
            exploration_decay_rate (float): The rate at which the exploration rate decays.
        """
        self.num_states = num_states
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.min_exploration_rate = min_exploration_rate
        self.exploration_decay_rate = exploration_decay_rate

        # Initialize Q-table with zeros
        self.q_table = np.zeros((num_states, num_actions))
        self.action_map = {0: "scale_down", 1: "hold", 2: "scale_up"}
        self.state_map = {0: "Underutilized", 1: "Optimal", 2: "Overutilized"}
        print(f"QLearningAgent initialized with {num_states} states and {num_actions} actions.")

    def choose_action(self, state: int) -> int:
        """
        Chooses an action based on the epsilon-greedy policy.

        Args:
            state (int): The current state of the system (0, 1, or 2).

        Returns:
            int: The chosen action (0: scale_down, 1: hold, 2: scale_up).
        """
        if not (0 <= state < self.num_states):
            print(f"Warning: Invalid state '{state}' provided. Choosing random action.")
            return random.randint(0, self.num_actions - 1)

        # Exploration-exploitation trade-off
        if random.uniform(0, 1) < self.exploration_rate:
            # Explore: choose a random action
            action = random.randint(0, self.num_actions - 1)
        else:
            # Exploit: choose the action with the highest Q-value for the current state
            action = int(np.argmax(self.q_table[state, :])) # Explicitly cast to int
        return action

    def learn(self, state: int, action: int, reward: float, next_state: int):
        """
        Updates the Q-value for a given state-action pair using the Q-learning formula.

        Q(s,a) = Q(s,a) + α * [R(s,a) + γ * max(Q(s',a')) - Q(s,a)]

        Args:
            state (int): The current state.
            action (int): The action taken in the current state.
            reward (float): The reward received after taking the action.
            next_state (int): The state reached after taking the action.
        """
        if not (0 <= state < self.num_states and 0 <= action < self.num_actions and 0 <= next_state < self.num_states):
            print(f"Error: Invalid state ({state}), action ({action}), or next_state ({next_state}) for learning. Skipping.")
            return

        current_q = self.q_table[state, action]
        max_future_q = np.max(self.q_table[next_state, :])
        
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_future_q - current_q)
        self.q_table[state, action] = new_q

    def decay_exploration_rate(self, episode: int):
        """
        Decays the exploration rate over time.
        """
        self.exploration_rate = self.min_exploration_rate + \
                                (self.exploration_rate - self.min_exploration_rate) * \
                                np.exp(-self.exploration_decay_rate * episode)

    def get_q_table(self) -> np.ndarray:
        """
        Returns the current Q-table.
        """
        return self.q_table

    def get_action_name(self, action_id: int) -> str:
        """Returns the human-readable name for an action ID."""
        return self.action_map.get(action_id, "unknown_action")

    def get_state_name(self, state_id: int) -> str:
        """Returns the human-readable name for a state ID."""
        return self.state_map.get(state_id, "unknown_state")

# Example Usage (for testing purposes)
if __name__ == "__main__":
    # Define a simple environment for demonstration of resource allocation
    # States: 0 (Underutilized), 1 (Optimal), 2 (Overutilized)
    # Actions: 0 (scale_down), 1 (hold), 2 (scale_up)

    NUM_STATES = 3
    NUM_ACTIONS = 3

    agent = QLearningAgent(num_states=NUM_STATES, num_actions=NUM_ACTIONS)

    # Simulate training over several episodes
    num_episodes = 2000
    max_steps_per_episode = 10

    # Define a simple reward function for resource allocation
    # Goal: Keep the system in the "Optimal" state (State 1)
    def get_reward(current_state: int, action: int, next_state: int) -> float:
        reward = 0.0
        # Reward for reaching/staying in Optimal state
        if next_state == 1:
            reward += 10.0
        # Penalties for being in Underutilized or Overutilized states
        elif next_state == 0: # Underutilized
            reward -= 5.0
            if action == 0: # scale_down in underutilized is good
                reward += 2.0
            elif action == 2: # scale_up in underutilized is bad
                reward -= 5.0
        elif next_state == 2: # Overutilized
            reward -= 5.0
            if action == 2: # scale_up in overutilized is bad
                reward -= 5.0
            elif action == 0: # scale_down in overutilized is good
                reward += 2.0

        # Small penalty for any action to encourage efficiency
        reward -= 0.1
        return reward

    print(f"\nStarting Q-learning training for {num_episodes} episodes...")
    for episode in range(num_episodes):
        current_state = random.randint(0, NUM_STATES - 1) # Start from a random state
        done = False
        steps = 0

        while not done and steps < max_steps_per_episode:
            action = agent.choose_action(current_state)

            # Simulate environment transition based on action
            # This is a simplified model; a real environment would be more complex
            next_state = current_state
            if action == 0: # scale_down
                next_state = max(0, current_state - 1) # Move towards underutilized
            elif action == 2: # scale_up
                next_state = min(NUM_STATES - 1, current_state + 1) # Move towards overutilized
            # If action is 1 (hold), state might fluctuate or stay same; for simplicity, let's say it stays same or moves slightly towards optimal
            elif action == 1: # hold
                if current_state == 0: # If underutilized, holding might move to optimal
                    next_state = 1
                elif current_state == 2: # If overutilized, holding might move to optimal
                    next_state = 1
                else: # If optimal, holding keeps it optimal
                    next_state = 1


            reward = get_reward(current_state, action, next_state)

            # Agent learns from the experience
            agent.learn(current_state, action, reward, next_state)

            current_state = next_state
            steps += 1

            if steps == max_steps_per_episode:
                done = True

        agent.decay_exploration_rate(episode)

        if (episode + 1) % 500 == 0:
            print(f"Episode {episode + 1}/{num_episodes}, Exploration Rate: {agent.exploration_rate:.4f}")

    print("\nQ-learning training complete.")
    print("\nLearned Q-table:")
    # Print Q-table with state and action names for better readability
    print("State \\ Action | scale_down (0) | hold (1)     | scale_up (2)")
    print("----------------------------------------------------------------")
    for s in range(NUM_STATES):
        row_values = [f"{agent.q_table[s, a]:<14.4f}" for a in range(NUM_ACTIONS)]
        print(f"{agent.get_state_name(s):<14} | {' | '.join(row_values)}")

    print("\nTesting the trained agent (exploitation phase):")
    test_states = [0, 1, 2, 0, 2]
    for i, state in enumerate(test_states):
        action_id = agent.choose_action(state) # Agent now mostly exploits
        action_name = agent.get_action_name(action_id)
        print(f"Test {i+1}: In state '{agent.get_state_name(state)}' ({state}), agent chooses action: '{action_name}' ({action_id})")
