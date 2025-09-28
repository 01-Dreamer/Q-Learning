import random
import numpy as np

# 环境类
class MazeEnvironment:
    def __init__(self, maze, start_state=(0, 0)):
        self.maze = np.array(maze)
        self.height, self.width = self.maze.shape
        
        # 行动空间(0: up, 1: down, 2: left, 3: right)
        self.action_space = list(range(4))
        
        # 奖励
        self.reward_goal = 100
        self.reward_wall = -10
        self.reward_step = -1
        
        # 起点
        self.start_state = start_state
        self.current_state = self.start_state

    def reset(self):
        self.current_state = self.start_state
        return self.current_state

    def step(self, action):
        x, y = self.current_state
        
        if action == 0: y -= 1    # up
        elif action == 1: y += 1  # down
        elif action == 2: x -= 1  # left
        elif action == 3: x += 1  # right
        
        # 检查是否越界或撞墙
        if x < 0 or x >= self.width or y < 0 or y >= self.height or self.maze[y, x] == 1:
            return self.current_state, self.reward_wall, False
            
        # 检查是否到达终点
        self.current_state = (x, y)
        if self.maze[y, x] == 2:
            return self.current_state, self.reward_goal, True
        
        # 普通移动
        return self.current_state, self.reward_step, False

# Q-Learning Agent
class QLearningAgent:
    def __init__(self, actions, alpha=0.1, gamma=0.9, 
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
        self.actions = actions
        self.alpha = alpha # 学习率
        self.gamma = gamma # 折扣因子
        self.epsilon = epsilon_start # 探索率
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.q_table = {}

    def get_q_value(self, state, action):
        if state not in self.q_table:
            self.q_table[state] = {act: 0.0 for act in self.actions}
        return self.q_table[state][action]

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.actions)
        else:
            q_values = self.q_table.get(state, {act: 0.0 for act in self.actions})
            return max(q_values, key=q_values.get)

    def learn(self, state, action, reward, next_state, done):
        old_q = self.get_q_value(state, action)
        if done:
            target = reward
        else:
            next_max_q = max(self.get_q_value(next_state, act) for act in self.actions)
            target = reward + self.gamma * next_max_q
            
        new_q = old_q + self.alpha * (target - old_q)
        self.q_table[state][action] = new_q
        
    def decay_epsilon(self):
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay

def print_optimal_policy(q_table, maze_env):
        policy_maze = np.array(maze_env.maze, dtype=str)
        policy_maze[policy_maze == '0'] = ' '
        policy_maze[policy_maze == '1'] = '@'
        policy_maze[policy_maze == '2'] = 'G'
        
        for state in q_table:
            if maze_env.maze[state[1], state[0]] == 0:
                best_action = max(q_table[state], key=q_table[state].get)
                x, y = state
                if best_action == 0: policy_maze[y, x] = '^'
                elif best_action == 1: policy_maze[y, x] = 'v'
                elif best_action == 2: policy_maze[y, x] = '<'
                else: policy_maze[y, x] = '>'
        
        for row in policy_maze:
            print(' '.join(row))


if __name__ == "__main__":
    # 定义环境、智能体
    MAZE = [
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [1, 1, 1, 1, 0, 1, 0, 1, 1, 0],
    [0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
    [0, 1, 0, 1, 0, 1, 1, 1, 1, 1],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 0, 1, 1, 0],
    [0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
    [1, 1, 1, 1, 0, 1, 0, 1, 1, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 1, 2],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 0]
    ]

    env = MazeEnvironment(maze=MAZE)
    agent = QLearningAgent(actions=env.action_space)
    

    # 训练
    num_episodes = 5000
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.learn(state, action, reward, next_state, done)
            state = next_state
            
        agent.decay_epsilon()
        
        if (episode + 1) % 500 == 0:
            print(f"Episode: {episode + 1}/{num_episodes} |  Epsilon: {agent.epsilon:.4f}")
    
    # 展示结果
    print_optimal_policy(agent.q_table, env)