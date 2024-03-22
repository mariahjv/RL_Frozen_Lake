import gym
import numpy as np
import matplotlib.pyplot as plt

# S = start, F = frozen, H = Hole, G = Goal

def main(num_episodes):
    # Initialize FrozenLake environment with the default 4x4 map
    environment = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=False)

    # discount factor
    gamma = 0.95
    # learning rate   
    alpha = 0.1 
    # state
    t = environment.observation_space.n
    # All returns & encounters
    num_returns = np.zeros(t)
    num_encounters = np.zeros(t)
    # value functions
    V = np.zeros(t)

    # loop for every episode
    for i in range(num_episodes):
        # Store total encounters of each state N(s)
        N_states = []

        # Return value for each state
        rewards = []

        # return to first state
        (current_state, probability) = environment.reset()
        N_states.append(current_state)

        # Go until episodes terminate
        while True:
            action = environment.action_space.sample() 
            # each step 
            state, reward, terminated, _, _ = environment.step(action)
            rewards.append(reward)                                
            
            # check if episode doesn't terminate
            if not terminated:
                N_states.append(state)
            else:
                break

        num_states = len(N_states)

        # G(t) = R(t+1) + gamma * R(t+2) + ... + gamma^(T-1) * R(t+T)
        total_return = 0

        # changed from starting in front for simplicity
        for i in reversed(range(num_states)):
            temp_s = N_states[i]
            temp_r = rewards[i]

            # total return from states = G(t)
            total_return = gamma * total_return + temp_r
            
            # state encountered for the 1st time
            if temp_s not in N_states[:i]:
                num_encounters[temp_s] += 1
                num_returns[temp_s] += total_return

    # Compute
    for s in range(t):
        if num_encounters[s] != 0:
            # Divide returns by number of times "s" is encountered: V(s)
            V[s] = num_returns[s] / num_encounters[s]

    environment.close()

    total_rewards = np.zeros(num_episodes)
    for j in range(num_episodes):
        total_rewards[j] = np.sum(num_returns[max(0, j-100) : (j+1)]) 
    plt.plot(total_rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Total Rewards')
    plt.title('Monte Carlo on FrozenLake')
    plt.savefig('FL_MC.png')
    

if __name__ == '__main__':
    main(100)
