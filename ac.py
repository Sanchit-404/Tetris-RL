import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from keras.models import Sequential, load_model
from keras.layers import Dense
from collections import deque
import numpy as np
import random
class ACAgent:
    '''Actor-Critic Agent'''

    def __init__(self, state_size, action_size=4, mem_size=10000, discount=0.95,
                 epsilon=1, epsilon_min=0, epsilon_stop_episode=0,
                 n_neurons=[32, 32], activations=['relu', 'relu', 'linear'],
                 loss='mse', optimizer='adam', replay_start_size=None, modelFile=None):

        # Similar checks and initializations as the DQNAgent
        if len(activations) != len(n_neurons) + 1:
            raise ValueError("n_neurons and activations do not match, "
                             f"expected a n_neurons list of length {len(activations) - 1}")

        if replay_start_size is not None and replay_start_size > mem_size:
            raise ValueError("replay_start_size must be <= mem_size")

        if mem_size <= 0:
            raise ValueError("mem_size must be > 0")

        self.state_size = state_size
        self.action_size = action_size
        self.mem_size = mem_size
        self.memory = deque(maxlen=mem_size)
        self.discount = discount
        if epsilon_stop_episode > 0:
            self.epsilon = epsilon
            self.epsilon_min = epsilon_min
            self.epsilon_decay = (self.epsilon - self.epsilon_min) / (epsilon_stop_episode)
        else: # no random exploration
            self.epsilon = 0
        self.n_neurons = n_neurons
        self.activations = activations
        self.loss = loss
        self.optimizer = optimizer
        if not replay_start_size:
            replay_start_size = mem_size / 2
        self.replay_start_size = replay_start_size

        # Load an existing model or create a new model
        if modelFile is not None:
            self.actor, self.critic = self._load_models(modelFile)
        else:
            self.actor, self.critic = self._build_models()

    def _build_models(self):
        '''Builds the Actor and Critic models'''

        # Actor model (policy network)
        actor = Sequential()
        actor.add(Dense(self.n_neurons[0], input_dim=self.state_size, activation=self.activations[0]))

        for i in range(1, len(self.n_neurons)):
            actor.add(Dense(self.n_neurons[i], activation=self.activations[i]))

        actor.add(Dense(self.action_size, activation='softmax'))  # Action probabilities (categorical)

        # Critic model (value function network)
        critic = Sequential()
        critic.add(Dense(self.n_neurons[0], input_dim=self.state_size, activation=self.activations[0]))

        for i in range(1, len(self.n_neurons)):
            critic.add(Dense(self.n_neurons[i], activation=self.activations[i]))

        critic.add(Dense(1, activation='linear'))  # State value (scalar)

        actor.compile(optimizer=self.optimizer)
        critic.compile(loss=self.loss, optimizer=self.optimizer)

        return actor, critic

    def _load_models(self, modelFile):
        '''Load pre-trained Actor and Critic models'''
        actor = load_model(modelFile + "_actor")
        critic = load_model(modelFile + "_critic")
        return actor, critic

    def add_to_memory(self, current_state, action, reward, next_state, done):
        '''Adds a transition to the replay buffer'''
        self.memory.append((current_state, action, reward, next_state, done))

    def act(self, state):
        '''Returns the action based on policy'''
        state = np.reshape(state, [1, self.state_size])
        action_probs = self.actor.predict(state)
        action = np.random.choice(self.action_size, p=action_probs[0])  # Sample action based on policy
        return action

    def train(self, batch_size=32, epochs=3):
        '''Trains the agent'''
        if batch_size > self.mem_size:
            print('WARNING: batch size is bigger than mem_size. The agent will not be trained.')

        n = len(self.memory)

        if n >= self.replay_start_size and n >= batch_size:

            batch = random.sample(self.memory, batch_size)

            x_states = []
            y_critic = []
            y_actor = []

            for state, action, reward, next_state, done in batch:

                # Predict value of next state using critic network
                next_value = self.critic.predict(np.reshape(next_state, [1, self.state_size]))[0]

                # Calculate target for the critic network
                target = reward + (self.discount * next_value * (1 - done))

                # Calculate advantage (difference between target and predicted value)
                advantage = target - self.critic.predict(np.reshape(state, [1, self.state_size]))[0]

                # Update critic
                x_states.append(state)
                y_critic.append(target)

                # Update actor
                action_probs = self.actor.predict(np.reshape(state, [1, self.state_size]))
                action_prob = action_probs[0][action]
                y_actor.append(advantage * np.log(action_prob))

            # Train critic model
            self.critic.fit(np.array(x_states), np.array(y_critic), batch_size=batch_size, epochs=epochs, verbose=0)

            # Train actor model
            self.actor.fit(np.array(x_states), np.array(y_actor), batch_size=batch_size, epochs=epochs, verbose=0)

            # Update exploration rate
            if self.epsilon > self.epsilon_min:
                self.epsilon -= self.epsilon_decay

    def save_model(self, name):
        '''Saves the current models.'''
        self.actor.save(name + "_actor")
        self.critic.save(name + "_critic")

