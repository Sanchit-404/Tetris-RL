import numpy as np
from keras.models import load_model

def load_trained_model(model_path):
    """
    Load the trained DQN model from the file.
    """
    return load_model(model_path)

def extract_policy(model, state_space):
    """
    Extract the policy from the trained model.

    Args:
        model: The trained DQN model.
        state_space: A list or array of all possible states to evaluate.
    
    Returns:
        policy: A dictionary mapping each state to its predicted Q-value.
    """
    policy = {}
    for state in state_space:
        state = np.reshape(state, [1, model.input_shape[1]])  # Reshape for model input
        q_value = model.predict(state, verbose=0)[0][0]  # Predict the Q-value
        policy[tuple(state.flatten())] = q_value
    return policy

# Example usage
if __name__ == "__main__":
    # Path to your trained model
    model_path = "sample.keras"
    model = load_trained_model(model_path)

    # Define the state space (example: states in a defined range)
    state_space = [np.linspace(-1, 1, model.input_shape[1]) for _ in range(10)]
    state_space = np.array(np.meshgrid(*state_space)).T.reshape(-1, model.input_shape[1])  # Create a grid of states

    # Extract the policy
    extracted_policy = extract_policy(model, state_space)

    # Print the policy
    for state, q_value in extracted_policy.items():
        print(f"State: {state}, Predicted Q-value: {q_value}")
