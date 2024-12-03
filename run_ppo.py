from stable_baselines3 import PPO
import gym
from tetris import Tetris
from datetime import datetime
import wandb
from tqdm import tqdm

# Wrap the Tetris environment for compatibility with PPO
class TetrisGymWrapper(gym.Env):
    def __init__(self):
        super(TetrisGymWrapper, self).__init__()

        # Initialize the Tetris environment
        self.tetris = Tetris()

        # Define the action space
        self.action_space = gym.spaces.Discrete(self.tetris.get_action_space_size())

        # Define the observation space
        self.observation_space = gym.spaces.Box(
            low=0,
            high=2,
            shape=(self.tetris.get_state_size(),),
            dtype=int
        )

    def reset(self):
        # Reset the Tetris game
        state = self.tetris.reset()
        return self._get_observation()

    def step(self, action):
        # Perform an action in the environment
        reward, done = self.tetris.play(action)
        observation = self._get_observation()
        return observation, reward, done, {}

    def render(self, mode='human'):
        self.tetris.render()

    def close(self):
        self.tetris.close()

    def _get_observation(self):
        # Return the current state of the game
        return self.tetris.get_state()


# Run PPO with Tetris
def ppo():
    env = TetrisGymWrapper()  # Wrap Tetris in Gym wrapper
    episodes = 3000  # Total number of episodes
    log_dir = f'logs/tetris-ppo-{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    wandb.init(project="tetris-ppo", entity="sanchit278", dir=log_dir)

    # Instantiate the PPO model
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)

    scores = []
    best_score = 0

    for episode in tqdm(range(episodes)):
        # Reset the environment and start the episode
        obs = env.reset()
        done = False
        steps = 0

        # Optionally render the game every few episodes
        if episode % 50 == 0:
            render = True
        else:
            render = False

        # Game loop
        while not done:
            action, _ = model.predict(obs, deterministic=False)  # Predict action
            obs, reward, done, _ = env.step(action)

            # Render if needed
            if render:
                env.render()

            steps += 1

        # Append the score
        scores.append(env.tetris.get_game_score())

        # Train the PPO model every episode
        model.learn(total_timesteps=1000)  # Train on a fixed number of timesteps

        # Log statistics
        if episode % 50 == 0:
            avg_score = sum(scores[-50:]) / 50
            min_score = min(scores[-50:])
            max_score = max(scores[-50:])
            wandb.log({
                "episode": episode,
                "average_score": avg_score,
                "min_score": min_score,
                "max_score": max_score,
            })

        # Save the best model
        if env.tetris.get_game_score() > best_score:
            best_score = env.tetris.get_game_score()
            model.save("best_ppo_model")

    wandb.finish()


if __name__ == "__main__":
    ppo()
