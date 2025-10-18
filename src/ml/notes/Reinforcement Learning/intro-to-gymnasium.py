"""
The Four Key Functions of Gymnasium:

make()
Env.reset()
Env.step()
Env.render()

"""

# What's Env?
"""
Env is a high-level Python class that represents a markov
decision process (MDP) from Reinforcement Learning Theory.
"""

# What's the agent loop?
"""
Agent Observes, 
Agent chooses an action,
Environment Responds,
Repeat until end of episode.

"""

# The first RL program
import gymnasium as gym

env = gym.make('CartPole-v1', render_mode = "human")

observation,info = env.reset()

print(f"Info: {info}\n\nStarting Observation: {observation}")
# Starting Observation: [-0.02760633  0.01638571  0.02276838 -0.0267169 ]
# [cart_position, cart_velocity, pole_angle, pole_angular_velocity]

episode_over = False
total_reward = 0

while not episode_over:
    action = env.action_space.sample() # this is a random action

    # take a step with the action!
    observation, reward, terminated, truncated, info = env.step(action)

    total_reward += reward
    episode_over = terminated or truncated

print(f"Episode Finished. Total reward: {total_reward}")
env.close()
