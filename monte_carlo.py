import pickle
import gym
import numpy as np
import matplotlib.pyplot as plt

# S = start, F = frozen, H = Hole, G = Goal

def main(num_episodes):
    # Initialize FrozenLake environment with the default 4x4 map
    environment = gym.make('FrozenLake-v1', map_name="4x4", render_mode="f", is_slippery=False)


if __name__ == '__main__':
    main(10000)
