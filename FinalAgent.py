from ale_py import ALEInterface

import gymnasium as gym
import numpy as np
import cv2
import random
import os
import pickle

#Feature Extraction 
def extract_features_from_screen(current_frame, prev_frame=None, prev_ball_pos=None):
    gray = cv2.cvtColor(current_frame, cv2.COLOR_RGB2GRAY)
    pin_area = gray[110:170, 100:160]
    pin_area_norm = cv2.normalize(pin_area, None, 0, 255, cv2.NORM_MINMAX)
    pin_edges = cv2.Canny(pin_area_norm, 50, 150)
    kernel = np.ones((2, 2), np.uint8)
    pin_edges = cv2.dilate(pin_edges, kernel, iterations=1)
    contours, _ = cv2.findContours(pin_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) > 5]

    pins_remaining = min(len(contours), 10)
    pin_x_positions = [cv2.boundingRect(c)[0] + cv2.boundingRect(c)[2] // 2 for c in contours]
    avg_pin_x = np.mean(pin_x_positions) if pin_x_positions else 30
    pin_centroid_x_norm = avg_pin_x / 60.0

    ball_area = gray[160:200, 90:170]
    ball_pos_x = -1
    if prev_frame is not None:
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
        prev_ball_area = prev_gray[160:200, 90:170]
        diff = cv2.absdiff(ball_area, prev_ball_area)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        ball_contours = [c for c in contours if cv2.contourArea(c) > 10]
        if ball_contours:
            c = max(ball_contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            ball_pos_x = x + w // 2

    ball_pos_x_norm = ball_pos_x / 80.0 if ball_pos_x >= 0 else 0.5
    motion = 0
    if prev_ball_pos is not None and ball_pos_x >= 0:
        motion = ball_pos_x - prev_ball_pos

    features = np.array([
        1.0,
        pins_remaining / 10.0,
        pin_centroid_x_norm,
        ball_pos_x_norm,
        motion / 80.0,
        (ball_pos_x_norm - pin_centroid_x_norm),
        (pins_remaining / 10.0) * abs(ball_pos_x_norm - pin_centroid_x_norm),
        *[0]*13
    ], dtype=np.float32)

    return features, ball_pos_x

#Approximate Q-learning Agent
class ApproxQLearningAgent:
    def __init__(self, num_actions, feature_dim, alpha=0.01, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.05):
        self.num_actions = num_actions
        self.weights = np.zeros((num_actions, feature_dim), dtype=np.float32)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

    def get_q_values(self, features):
        return self.weights @ features

    def select_action(self, features):
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        return np.argmax(self.get_q_values(features))

    def update(self, features, action, reward, next_features, done):
        q_current = self.get_q_values(features)[action]
        q_next = np.max(self.get_q_values(next_features)) if not done else 0.0
        target = reward + self.gamma * q_next
        td_error = target - q_current
        self.weights[action] += self.alpha * td_error * features
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay

#Video Saving
def save_video(frames, video_filename):
    if not frames:
        print(f"No frames to save for {video_filename}. Skipping...")
        return

    height, width, _ = np.array(frames[0]).shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_filename, fourcc, 30.0, (width, height))

    for frame in frames:
        frame_bgr = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
        video_writer.write(frame_bgr)

    video_writer.release()

# ----- Main Training -----
env = gym.make("ALE/Bowling-v5", render_mode="rgb_array")
agent = ApproxQLearningAgent(num_actions=env.action_space.n, feature_dim=20)

num_episodes = 5000
log = []

for episode in range(1, num_episodes + 1):
    obs, _ = env.reset()
    prev_obs = None
    prev_ball_x = None
    total_reward = 0
    done = False
    episode_features = []
    episode_actions = []
    episode_rewards = []
    frames = []

    while not done:
        features, ball_x = extract_features_from_screen(obs, prev_obs, prev_ball_x)
        action = agent.select_action(features)
        next_obs, reward, done, truncated, _ = env.step(action)

        next_features, _ = extract_features_from_screen(next_obs, obs, ball_x)

        episode_features.append(features)
        episode_actions.append(action)
        episode_rewards.append(reward)

        agent.update(features, action, reward, next_features, done)
        total_reward += reward

        frames.append(obs)  # Store for recording

        prev_obs = obs
        prev_ball_x = ball_x
        obs = next_obs

    # Custom reward shaping
    shaped_reward = 0.0
    if total_reward == 0:
        shaped_reward -= 200.0
    elif total_reward < 25:
        shaped_reward -= 25.0
    elif total_reward >= 75:
        shaped_reward += 100.0
    if total_reward >= 40:
        shaped_reward += 40.0

    if episode_features:
        last_feat = episode_features[-1]
        last_action = episode_actions[-1]
        agent.update(last_feat, last_action, shaped_reward, np.zeros_like(last_feat), True)

    print(f"Episode {episode} — Score: {total_reward}")
    log.append((episode, total_reward))

    if total_reward > 85:
        save_video(frames, f"episode_{episode}_score_{total_reward}.mp4")

with open("bowling_log.pkl", "wb") as f:
    pickle.dump(log, f)

#Final Evaluation
obs, _ = env.reset()
prev_obs = None
prev_ball_x = None
done = False
total_reward = 0
frames = []

while not done:
    features, ball_x = extract_features_from_screen(obs, prev_obs, prev_ball_x)
    action = agent.select_action(features)
    obs, reward, done, truncated, _ = env.step(action)
    total_reward += reward
    frames.append(obs)
    prev_obs = obs
    prev_ball_x = ball_x

print(f"Final trained agent performance — Score: {total_reward}")
save_video(frames, "final_trained_agent.mp4")
