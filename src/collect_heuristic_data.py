"""
COLLECT HEURISTIC PONG DATA — Ball-tracking agent for reward-rich replay data.

The agent reads the current frame, finds the ball's y-position, and moves
the paddle toward it. This scores ~5-15 points per episode vs 0-1 for random.

Saves frames.npy, actions.npy, rewards.npy, dones.npy to data/Pong-v5-heuristic/
in the exact format that extract_spatial_tokens.py expects.

Usage:
    python src/collect_heuristic_data.py
    python src/collect_heuristic_data.py --target 100000 --episodes 500
"""

import os
import argparse
import time
import numpy as np

import ale_py
import gymnasium as gym


# PONG ACTION MAPPING
# 0=NOOP, 1=FIRE, 2=RIGHT(up), 3=LEFT(down), 4=RIGHTFIRE, 5=LEFTFIRE
ACTION_UP   = 2   # RIGHT moves paddle up in Pong
ACTION_DOWN = 3   # LEFT moves paddle down in Pong
ACTION_NOOP = 0
ACTION_FIRE = 1   # needed to serve the ball


def find_ball_y(frame: np.ndarray) -> float:
    """Find the ball's y-position from an 84x84 grayscale frame.

    The ball is a small bright pixel cluster in the middle region of the screen.
    We scan the playfield area (excluding score region at top and paddles at edges).

    Returns y-position (0-83) or -1 if ball not found.
    """
    # Ball lives roughly in columns 15-70 (avoid paddles at edges)
    # and rows 5-80 (avoid score display at top)
    playfield = frame[5:80, 15:70]

    # Ball is bright (>150 in uint8 space)
    bright = np.where(playfield > 150)

    if len(bright[0]) == 0:
        return -1.0  # ball not visible (between points, behind paddle, etc.)

    # Return mean y-position in FULL frame coordinates
    ball_y = bright[0].mean() + 5
    return ball_y


def find_paddle_y(frame: np.ndarray) -> float:
    """Find the right paddle (agent's paddle) y-position.

    In 84x84 preprocessed Pong, the agent's paddle is the bright bar
    on the right side of the screen (columns ~73-78).
    """
    paddle_region = frame[5:80, 70:80]
    bright = np.where(paddle_region > 150)

    if len(bright[0]) == 0:
        return 42.0  # default to center

    return bright[0].mean() + 5


def heuristic_action(frame: np.ndarray, prev_ball_y: float) -> tuple:
    """Pick action based on ball tracking.

    Returns (action, ball_y) where ball_y is used for next step's tracking.
    """
    ball_y = find_ball_y(frame)
    paddle_y = find_paddle_y(frame)

    # If ball not visible, use last known position or stay still
    if ball_y < 0:
        ball_y = prev_ball_y if prev_ball_y > 0 else 42.0

    # Simple proportional control: move toward ball
    diff = ball_y - paddle_y

    # Dead zone to prevent oscillation
    if abs(diff) < 3:
        action = ACTION_NOOP
    elif diff > 0:
        action = ACTION_DOWN  # ball is below paddle, move down
    else:
        action = ACTION_UP    # ball is above paddle, move up

    return action, ball_y


def collect_heuristic_episodes(
    target_transitions: int = 100_000,
    max_episodes: int = 1000,
    save_dir: str = "data/Pong-v5-heuristic",
):
    """Collect episodes using ball-tracking heuristic."""

    print(f"COLLECTING HEURISTIC PONG DATA")
    print(f"  Target transitions: {target_transitions:,}")
    print(f"  Max episodes: {max_episodes}")
    print(f"  Save dir: {save_dir}")
    print()

    env = gym.make("ALE/Pong-v5")
    env = gym.wrappers.AtariPreprocessing(
        env, frame_skip=1, screen_size=84,
        grayscale_obs=True, scale_obs=False,
    )
    env = gym.wrappers.FrameStackObservation(env, stack_size=4)

    # PRE-ALLOCATE (same format as collect_data.py)
    frames  = np.zeros((target_transitions, 4, 84, 84), dtype=np.uint8)
    actions = np.zeros(target_transitions, dtype=np.int32)
    rewards = np.zeros(target_transitions, dtype=np.float32)
    dones   = np.zeros(target_transitions, dtype=bool)

    total = 0
    ep_returns = []
    start_time = time.time()

    for ep in range(max_episodes):
        obs, _ = env.reset()
        episode_return = 0.0
        prev_ball_y = -1.0
        step_in_ep = 0

        while True:
            # STORE TRANSITION
            frames[total]  = obs
            current_frame  = obs[-1]  # most recent of 4 stacked frames

            # FIRE to serve at start of each point
            if step_in_ep < 2:
                action = ACTION_FIRE
                prev_ball_y = -1.0
            else:
                action, prev_ball_y = heuristic_action(current_frame, prev_ball_y)

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            actions[total] = action
            rewards[total] = reward
            dones[total]   = done

            episode_return += reward
            total += 1
            step_in_ep += 1
            obs = next_obs

            if done or total >= target_transitions:
                break

        ep_returns.append(episode_return)
        pos = sum(1 for r in rewards[:total] if r > 0)
        neg = sum(1 for r in rewards[:total] if r < 0)

        print(f"  Episode {ep+1:>4d} | return={episode_return:>6.0f} | "
              f"transitions={total:>6d} | +rewards={pos} -rewards={neg}")

        if total >= target_transitions:
            break

    elapsed = time.time() - start_time
    env.close()

    # TRIM TO ACTUAL SIZE
    frames  = frames[:total]
    actions = actions[:total]
    rewards = rewards[:total]
    dones   = dones[:total]

    # STATS
    pos_rewards = (rewards > 0).sum()
    neg_rewards = (rewards < 0).sum()
    mean_return = np.mean(ep_returns)

    print(f"\nCOLLECTION COMPLETE")
    print(f"  Episodes: {len(ep_returns)}")
    print(f"  Transitions: {total:,}")
    print(f"  Mean return: {mean_return:.1f}")
    print(f"  Positive rewards: {pos_rewards}")
    print(f"  Negative rewards: {neg_rewards}")
    print(f"  Time: {elapsed:.0f}s")

    # SAVE
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, "frames.npy"),  frames)
    np.save(os.path.join(save_dir, "actions.npy"), actions)
    np.save(os.path.join(save_dir, "rewards.npy"), rewards)
    np.save(os.path.join(save_dir, "dones.npy"),   dones)

    print(f"\nSaved to {save_dir}/")
    print(f"  frames.npy   {frames.shape} {frames.dtype}")
    print(f"  actions.npy  {actions.shape}")
    print(f"  rewards.npy  {rewards.shape}  (+{pos_rewards}/-{neg_rewards})")
    print(f"  dones.npy    {dones.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect heuristic Pong data")
    parser.add_argument("--target",   type=int, default=100_000)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--save_dir", type=str, default="data/Pong-v5-heuristic")
    args = parser.parse_args()

    collect_heuristic_episodes(
        target_transitions=args.target,
        max_episodes=args.episodes,
        save_dir=args.save_dir,
    )
