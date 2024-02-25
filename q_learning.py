import gym
import numpy as np
import matplotlib.pyplot as plt

# S = start, F = frozen, H = Hole, G = Goal

def main(num_episodes):
    # Initialize FrozenLake environment with the default 4x4 map
    environment = gym.make('FrozenLake-v1', map_name="4x4", render_mode="False", is_slippery="False")
    # initialize q-learning table(4x4)
    Q = np.zeros((environment.observation_space.n, environment.action_space.n)) 
    # discount factor
    gamma = 0.9 
    # learning rate   
    alpha = 0.1 
    # exploration rate  
    epsilon = 0.5  
    random_num = np.random.default_rng()
    episode_rewards = np.zeros(num_episodes)

    print("Start environmant")
    for i in range(num_episodes):
        # Resets the environment to an initial state; returns tuple
        state = environment.reset()[0]

        terminated = False
        truncated = False

        while not terminated and not truncated:
            # Renders the environment to help visualize what the agent sees
            environment.render()

            if random_num.random() < epsilon:
                action = environment.action_space.sample()
            else:
                action = np.argmax(Q[state, :])

            next_state, reward, terminated, truncated, info = environment.step(action)

            # Q-learning formula to update
            Q[state, action] = Q[state, action] + alpha * (reward + gamma * (np.max(Q[next_state, :]) - Q[state, action]))
            state = next_state

        if reward == 1:
            episode_rewards[i] = 1

        # print(f"Action: {action}, Observation: {observation}, Reward: {reward}, Truncated: {truncated}, Info: {info}")

        if terminated or truncated:
            state = environment.reset()[0]
        
    environment.close()


if __name__ == '__main__':
    main(1500)

# step through manually
# while True:
#     #  0 - left, 1 - down, 2 - right, 3 - up
#     action = int(input("Enter action (0-3): "))
#     step_result = environment.step(action)
#     observation, action, reward, done, info = step_result
#     print(f"Action: {action}, Observation: {observation}, Reward: {reward}, Done: {done}, Info: {info}")
#     environment.render()
#     if done:
#         break