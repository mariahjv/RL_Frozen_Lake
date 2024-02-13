# pip install gymnasium
import gymnasium as gym

environment = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True)

# Resets the environment to an initial state
environment.reset()
# Renders the environments to help visualise what the agent see
environment.render()

environment.close()