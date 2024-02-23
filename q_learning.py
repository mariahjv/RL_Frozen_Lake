import gym

# S = start, F = frozen, H = Hole, G = Goal

# Initialize FrozenLake environment with the default 4x4 map
environment = gym.make('FrozenLake-v1', render_mode="human", is_slippery=False)

# Resets the environment to an initial state
init_state = environment.reset()

# Renders the environment to help visualize what the agent sees
environment.render()

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

observation, info = environment.reset()

for _ in range(25):
    action = environment.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = environment.step(action)

    print(f"Action: {action}, Observation: {observation}, Reward: {reward}, Truncated: {truncated}, Info: {info}")

    if terminated or truncated:
        observation, info = environment.reset()
    
environment.close()