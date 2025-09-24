import gymnasium as gym
import ale_py

from collections import defaultdict
import gymnasium as gym
import numpy as np

from tqdm import tqdm

import os, gzip, pickle
import numpy as np
from collections import defaultdict
from pathlib import Path
from datetime import datetime, timezone
from zoneinfo import ZoneInfo  # Python 3.9+

class Agent:
    def __init__(self, env, learning_rate, initial_epsilon, epsilon_decay, final_epsilon, discount_factor=0.95):
        self.env = env
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n, dtype=np.float32))
        self.lr = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.training_error = []

    # --- compress pixels → small, hashable key (bytes) ---
    def encode(self, obs: np.ndarray) -> bytes:
        x = np.asarray(obs)
        if x.ndim == 3:                 # H, W, C (RGB)
            x = x.mean(axis=2)          # grayscale
        if x.ndim == 2:                 # image
            x = x[::8, ::8]             # downsample
        else:                           # e.g., RAM vector
            x = x[::4]
        x = (x // 32).astype(np.uint8)  # bin to 0..7
        return x.tobytes()

    def get_action(self, obs) -> int:
        key = self.encode(obs)
        if (np.random.random() < self.epsilon) or (key not in self.q_values):
            return self.env.action_space.sample()
        return int(np.argmax(self.q_values[key]))

    def update(self, obs, action: int, reward: float, terminated: bool, next_obs):
        k  = self.encode(obs)
        nk = self.encode(next_obs)
        future_q = 0.0 if terminated else float(np.max(self.q_values[nk]))
        target = reward + self.discount_factor * future_q
        td = target - self.q_values[k][action]
        self.q_values[k][action] += self.lr * td
        self.training_error.append(td)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

    # =======================
    # Persistence helpers
    # =======================
    def save(self, path: str = None):
        """Save Q-table + a bit of metadata to a compressed pickle."""
        if path is None:
            path = self.timestamped_path()

        payload = {
            "version": 1,
            "action_n": self.env.action_space.n,
            "q_values": dict(self.q_values),  # bytes -> np.ndarray(float32)
            "meta": {
                "lr": self.lr,
                "discount_factor": self.discount_factor,
                "epsilon": self.epsilon,
                "epsilon_decay": self.epsilon_decay,
                "final_epsilon": self.final_epsilon,
            },
        }
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with gzip.open(path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, path: str = None, strict_action_space: bool = True, latest: bool = True) -> int:
        """Load Q-table from file. Returns number of states loaded.
        If action-space size differs:
          - strict_action_space=True → raise ValueError
          - strict_action_space=False → skip mismatched entries
        """
        if latest or path is None:
            path = self.latest_checkpoint_by_name()            

        try:
            with gzip.open(path, "rb") as f:
                payload = pickle.load(f)
        except Exception as e:
            raise Exception(f"load(): can't seem to find the file. Creating a new one instead.")

        saved_action_n = int(payload.get("action_n", self.env.action_space.n))
        curr_action_n = int(self.env.action_space.n)
        if strict_action_space and saved_action_n != curr_action_n:
            raise ValueError(f"Action space mismatch: saved {saved_action_n}, current {curr_action_n}")

        raw = payload["q_values"]
        # rebuild defaultdict with correct default
        self.q_values = defaultdict(lambda: np.zeros(curr_action_n, dtype=np.float32))

        loaded = 0
        for k, v in raw.items():
            arr = np.asarray(v, dtype=np.float32)
            if arr.shape[0] != curr_action_n:
                if strict_action_space:
                    # unreachable due to check above, but keep as guard
                    continue
                # Non-strict mode: skip mismatched rows
                continue
            self.q_values[k] = arr
            loaded += 1

        # Optionally restore exploration hyperparams (comment out if you don’t want this)
        # meta = payload.get("meta", {})
        # self.lr = float(meta.get("lr", self.lr))
        # self.discount_factor = float(meta.get("discount_factor", self.discount_factor))
        # self.epsilon = float(meta.get("epsilon", self.epsilon))
        # self.epsilon_decay = float(meta.get("epsilon_decay", self.epsilon_decay))
        # self.final_epsilon = float(meta.get("final_epsilon", self.final_epsilon))

        return loaded

    def timestamped_path(self, prefix="q", ext=".pkl.gz", tz="UTC", ms=False, dir="checkpoints"):
        if tz == "UTC":
            now = datetime.now(timezone.utc)
            fmt = "%Y-%m-%dT%H-%M-%SZ"
        else:
            now = datetime.now(ZoneInfo(tz))
            fmt = "%Y-%m-%d_%H-%M-%S_%Z"
        if ms:
            # add milliseconds safely
            fmt = fmt.replace("%S", "%S_%f")  # append microseconds; trim below
            stamp = now.strftime(fmt)
            if "_%f" in fmt:
                stamp = stamp[:-3]            # drop last 3 digits → ms
        else:
            stamp = now.strftime(fmt)
        Path(dir).mkdir(parents=True, exist_ok=True)
        return Path(dir) / f"{prefix}_{stamp}{ext}"
    
    def latest_checkpoint_by_name(self, dir="checkpoints", prefix="galaxian_q", ext=".pkl.gz", recursive=True):
        root = Path(dir)
        pattern = f"**/{prefix}_*{ext}" if recursive else f"{prefix}_*{ext}"
        files = sorted(root.glob(pattern), key=lambda p: p.name)  # lexicographic
        return files[-1] if files else None

def train(env, agent, n_episodes):
    try:
        agent.load()
    except:
        pass

    for episode in tqdm(range(n_episodes)):
        obs, info = env.reset()
        done = False

        while not done:
            # Agent chooses action (initially random, gradually more intelligent)
            action = agent.get_action(obs)

            # Take action and observe result
            next_obs, reward, terminated, truncated, info = env.step(action)

            # Learn from this experience
            agent.update(obs, action, reward, terminated, next_obs)

            # Move to next state
            done = terminated or truncated
            obs = next_obs
        # Reduce exploration rate (agent becomes less random over time)
        agent.decay_epsilon()
    
    # save progress!
    agent.save()


def play():
    learning_rate = 0.01        # How fast to learn (higher = faster but less stable)
    n_episodes = 500        # Number of hands to practice
    start_epsilon = 1.0         # Start with 100% random actions
    epsilon_decay = start_epsilon / (n_episodes / 2)  # Reduce exploration over time
    final_epsilon = 0.1         # Always keep some exploration
    
    gym.register_envs(ale_py)
    env = gym.make('ALE/Galaxian-v5')
    env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)

    agent = Agent(
        env=env,
        learning_rate=learning_rate,
        initial_epsilon=start_epsilon,
        epsilon_decay=epsilon_decay,
        final_epsilon=final_epsilon,
    )
    train(env, agent, n_episodes)
    env.close()
    
    gym.register_envs(ale_py)
    env = gym.make('ALE/Galaxian-v5', render_mode="human")
    observation, info = env.reset(seed=42)
    
    for _ in range(1000):
        action = agent.get_action(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            observation, info = env.reset()

    env.close()

play()
