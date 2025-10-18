from collections import defaultdict
import pickle
from typing import Tuple, Dict
import gymnasium as gym
import numpy as np

# -----------------------
# Agent
# -----------------------
class BlackjackAgent:
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
    ):
        self.env = env
        self.q_values: Dict[Tuple[int, int, bool], np.ndarray] = defaultdict(
            lambda: np.zeros(env.action_space.n)
        )
        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []

    def get_action(self, obs: Tuple[int, int, bool]) -> int:
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        return int(np.argmax(self.q_values[obs]))

    def update(
        self,
        obs: Tuple[int, int, bool],
        action: int,
        reward: float,
        terminated: bool,
        next_obs: Tuple[int, int, bool],
    ):
        future_q = (not terminated) * np.max(self.q_values[next_obs])
        target = reward + self.discount_factor * future_q
        td = target - self.q_values[obs][action]
        self.q_values[obs][action] += self.lr * td
        self.training_error.append(td)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

    # --- Persistence ---
    def save(self, filepath: str):
        """Save just what's necessary: Q-table and scalar params."""
        blob = {
            "q_values": {k: v.tolist() for k, v in self.q_values.items()},
            "action_space_n": self.env.action_space.n,
            "lr": self.lr,
            "discount_factor": self.discount_factor,
            "epsilon": self.epsilon,
            "epsilon_decay": self.epsilon_decay,
            "final_epsilon": self.final_epsilon,
        }
        with open(filepath, "wb") as f:
            pickle.dump(blob, f)

    @staticmethod
    def load(filepath: str, env: gym.Env) -> "BlackjackAgent":
        with open(filepath, "rb") as f:
            blob = pickle.load(f)
        agent = BlackjackAgent(
            env=env,
            learning_rate=blob["lr"],
            initial_epsilon=blob["epsilon"],
            epsilon_decay=blob["epsilon_decay"],
            final_epsilon=blob["final_epsilon"],
            discount_factor=blob["discount_factor"],
        )
        # restore q_values
        agent.q_values = defaultdict(
            lambda: np.zeros(blob["action_space_n"]),
            {k: np.array(v, dtype=float) for k, v in blob["q_values"].items()},
        )
        return agent


# -----------------------
# Training function
# -----------------------
def train_blackjack_agent(
    n_episodes: int = 10_000,
    learning_rate: float = 0.01,
    start_epsilon: float = 1.0,
    final_epsilon: float = 0.1,
    epsilon_half_life_fraction: float = 0.5,  # decay to final_epsilon by ~half training
    discount_factor: float = 0.95,
    sab: bool = False,
    save_path: str | None = "q_blackjack.pkl",
    progress: bool = True,
):
    """
    Train a Q-learning agent on Blackjack-v1 and (optionally) save its Q-table.

    Returns:
        env, agent
    """
    env = gym.make("Blackjack-v1", sab=sab)
    env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)

    epsilon_decay = (start_epsilon - final_epsilon) / max(1, int(n_episodes * epsilon_half_life_fraction))

    agent = BlackjackAgent(
        env=env,
        learning_rate=learning_rate,
        initial_epsilon=start_epsilon,
        epsilon_decay=epsilon_decay,
        final_epsilon=final_epsilon,
        discount_factor=discount_factor,
    )

    rng = range(n_episodes)
    if progress:
        try:
            from tqdm import tqdm
            rng = tqdm(rng)
        except Exception:
            pass

    for _ in rng:
        obs, _ = env.reset()
        done = False
        while not done:
            action = agent.get_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            agent.update(obs, action, reward, terminated, next_obs)
            obs = next_obs
            done = terminated or truncated
        agent.decay_epsilon()

    if save_path:
        agent.save(save_path)

    return env, agent


# -----------------------
# Test / Eval (no learning)
# -----------------------
def evaluate_agent(agent: BlackjackAgent, env: gym.Env, num_episodes: int = 1000):
    total_rewards = []
    old_epsilon = agent.epsilon
    agent.epsilon = 0.0  # pure exploitation

    for _ in range(num_episodes):
        obs, _ = env.reset()
        ep_reward = 0.0
        done = False
        while not done:
            action = agent.get_action(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            done = terminated or truncated
        total_rewards.append(ep_reward)

    agent.epsilon = old_epsilon
    total_rewards = np.array(total_rewards, dtype=float)
    return {
        "win_rate": float(np.mean(total_rewards > 0)),
        "avg_reward": float(np.mean(total_rewards)),
        "std_reward": float(np.std(total_rewards)),
    }


# -----------------------
# "Watch" (print) a full game
# -----------------------
ACTION_NAME = {0: "STAND", 1: "HIT"}

def play_full_game(agent: BlackjackAgent, env: gym.Env, verbose: bool = True):
    """
    Play ONE full episode and print each decision so you can 'watch' the agent.
    Blackjack doesn't have a GUI renderer; this prints the trajectory.
    """
    old_epsilon = agent.epsilon
    agent.epsilon = 0.0  # force greedy to see the learned policy

    obs, info = env.reset()
    done = False
    step = 0
    if verbose:
        print("\n=== New Blackjack Hand ===")
        print(f"Initial state: (player_sum={obs[0]}, dealer_showing={obs[1]}, usable_ace={obs[2]})")

    while not done:
        action = agent.get_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if verbose:
            step += 1
            print(
                f"Step {step:02d}: action={ACTION_NAME[action]:>5}  "
                f"-> next_state=(sum={next_obs[0]}, dealer={next_obs[1]}, usable_ace={next_obs[2]})  "
                f"reward={reward:+.1f}"
            )

        obs = next_obs

    if verbose:
        outcome = "WIN" if reward > 0 else ("LOSS" if reward < 0 else "PUSH")
        print(f"=== Hand over: {outcome}, final reward={reward:+.1f} ===\n")

    agent.epsilon = old_epsilon
    return reward


# -----------------------
# Example usage
# -----------------------
if __name__ == "__main__":
    # 1) Train & save
    env, agent = train_blackjack_agent(
        n_episodes=100_000,
        learning_rate=0.01,
        start_epsilon=1.0,
        final_epsilon=0.1,
        epsilon_half_life_fraction=0.5,
        discount_factor=0.95,
        sab=False,
        save_path="q_blackjack.pkl",
        progress=True,
    )

    # 2) Quick evaluation
    stats = evaluate_agent(agent, env, num_episodes=2000)
    print(
        f"Eval → win_rate={stats['win_rate']:.1%}, "
        f"avg_reward={stats['avg_reward']:.3f}, "
        f"std={stats['std_reward']:.3f}"
    )

    # 3) Load later & watch one hand (prints the trajectory)
    #    (You can comment training above and just do these two lines)
    # env = gym.make("Blackjack-v1", sab=False)
    # agent = BlackjackAgent.load("q_blackjack.pkl", env)

    play_full_game(agent, env, verbose=True)
