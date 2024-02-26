import pickle
import gym
import numpy as np
import matplotlib.pyplot as plt

# S = start, F = frozen, H = Hole, G = Goal

def main(num_episodes):
    # Initialize FrozenLake environment with the default 4x4 map
    environment = gym.make('FrozenLake-v1', map_name="4x4", render_mode="human", is_slippery=False)
    
    # discount factor
    gamma = 0.9 
    # learning rate   
    alpha = 0.1 
    # exploration rate  
    epsilon = 0.1  
    random_num = np.random.default_rng()
    episode_rewards = np.zeros(num_episodes)

    filename = "q_table.npy"
    try:
        if filename is not None:
            Q = np.load(filename)
    except FileNotFoundError:
        Q = np.zeros((environment.observation_space.n, environment.action_space.n))

    print("Start environmant")
    for i in range(num_episodes):
        # Resets the environment to an initial state; returns tuple
        state = environment.reset()[0]

        terminated = False
        truncated = False

        while not terminated and not truncated:
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

    total_rewards = np.zeros(num_episodes)
    for j in range(num_episodes):
        total_rewards[j] = np.sum(episode_rewards[max(0, j-100) : (j+1)]) 
    plt.plot(total_rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Total Rewards')
    plt.title('Q-learning on FrozenLake')
    plt.savefig('FL_Q-learning.png')

    if filename is not None:
        np.save(filename, Q)

if __name__ == '__main__':
    main(1)

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