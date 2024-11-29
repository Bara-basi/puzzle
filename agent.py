import os
import csv
import numpy as np
import time
import random
from collections import defaultdict
from config import *
import tqdm


vector = np.array(([1,0],[-1,0],[0,-1],[0,1]))
class PuzzleEnv:
    def __init__(self, state, goal):
        self.original_state = state
        self.goal = goal
        self.state = state.copy()
        self.x = BLOCK_NUM-1
        self.y = BLOCK_NUM-1 # 0的初始位置始终为最后一块拼图
        
        self.action_space = [0,1,2,3]
    def reset(self):
        self.state = self.original_state.copy()
    def step(self,new_x,new_y):
        last_distance = self.cal_distance(self.state)
        next_state = self.state.copy()
        next_state[self.x][self.y], next_state[new_x][new_y] = next_state[new_x][new_y], next_state[self.x][self.y]
        cur_distance = self.cal_distance(next_state)
        if cur_distance == 0:  
            reward = 100
            done = True
        elif cur_distance < last_distance: 
            reward = 5
            done = False
        elif cur_distance > last_distance: 
            reward = -1
            done = False
        else:  
            reward = -0.1  
            done = False
        return next_state, reward, done
        

    
    def cal_distance(self,state):
        goal = self.goal
        distance = 0
        for i in range(BLOCK_NUM):
            for j in range(BLOCK_NUM):
                if state[i][j] == 0:
                    continue
                value = state[i][j]
                target_pos = np.where(goal == value)
                distance += abs(i-target_pos[0][0]) + abs(j-target_pos[1][0])
        return distance
    
        
        

class PuzzleAgent:
    def __init__(self):
        self.Q = defaultdict(lambda: np.zeros(4,dtype=float))
        self.lr = AGENT_LR
        self.epsilon = EPSILON
        self.gamma = GAMMA
        self.action_space = [0,1,2,3]
        

    def choose_action(self, state,x,y):
        action_space = self.action_space
        if random.random() < self.epsilon:
            action = np.random.choice(action_space)
        else:  
            max_Q = np.max(self.Q[tuple(state.flatten())])
            action = np.random.choice(np.where(self.Q[tuple(state.flatten())] == max_Q)[0])
        new_x = x + vector[action][0]
        new_y = y + vector[action][1]
        if new_x < 0 or new_x >= BLOCK_NUM or new_y < 0 or new_y >= BLOCK_NUM:
            return self.choose_action(state,x,y)
        return action,new_x,new_y


    def learn(self, state, action, reward, next_state):
        max_next_Q = np.max(self.Q[tuple(next_state.flatten())])
        current_Q = self.Q[tuple(state.flatten())][action]
        self.Q[tuple(state.flatten())][action] = current_Q + self.lr * (reward + self.gamma * max_next_Q - current_Q)

    def train(self, env):
        step_count = 0
        done = False
        total_reward = 0
        while not done:
            #选择动作
            action,new_x,new_y = self.choose_action(env.state,env.x,env.y)
            #执行动作
            next_state, reward ,done = env.step(new_x,new_y)
            total_reward += reward
            #更新Q值
            self.learn(env.state,action,reward,next_state)
            env.state = next_state
            env.x = new_x
            env.y = new_y
            step_count += 1
            if step_count > 200:
                reward = -100
                break
        self.epsilon = max(self.epsilon*EPSILON_DECAY**step_count,EPSILON_MIN)
        if done:
            self.epsilon *= 0.8
        return done,step_count

    def solve(self, env):
        
        done = False
        while not done :
            change_list = [[2,2]]
            for step in range(2000): 
                action,new_x,new_y = self.choose_action(env.state,env.x,env.y)
                next_state, _ , done = env.step(new_x,new_y)
                env.state = next_state
                change_list.append([new_x,new_y])
                if done:
                    break
                env.x = new_x
                env.y = new_y
            env.x = 2
            env.y = 2
            env.state = env.original_state.copy()
        
        if done:
            print('solved in {} steps'.format(step-1))
            return change_list
        else:
            print('can not solve,please check the puzzle is solvable')
            return None
           
    def save_Q(self, csv_path):
        with open(csv_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['key', 'value'])
            for key, value in self.Q.items():
                # 将key转化为float类型的tuple 转化为整形的字符串，将 ， 替换为 空格
                key = tuple(map(int, key))
                key = str(key).replace(',',' ').replace('(','').replace(')','')
                value = str(value).replace(',',' ').replace('[','').replace(']','')
                writer.writerow([key, value])
            
        

    def load_Q(self, csv_path):
        self.Q = defaultdict(lambda: np.zeros(4,dtype=float))
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                key = row[0].split()
                value = row[1].split()
                key = tuple(map(int, key))
                value = np.array(list(map(float, value)))
                self.Q[key] = value

def solvable(state):
    if(type(state) == list):
        state = np.array(state)+1
        state[8] = 0
    # 逆序数为偶数则可解   
    inversions = 0
    state = [x for x in state.flatten() if x != 0]
    for i in range(len(state)):
        for j in range(i+1,len(state)):
            if state[i] > state[j]:
                inversions += 1
    return inversions % 2 == 0

def generate_puzzle(size):
    goal = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            goal[i][j] = i*3+j+1
    state = goal.copy()
    state = state.flatten()
    
    random.shuffle(state)
    mark_value = state[-1]
    state[-1] = 0
    while not solvable(state):
        state[-1] = mark_value
        random.shuffle(state)
        mark_value = state[-1]
        state[-1] = 0
    state = state.reshape((size, size))
    goal[goal == mark_value] = 0
    return state, goal,mark_value                                                                                                                               
    
                                                                                                      
if __name__ == '__main__':                                                                                      
    agent = PuzzleAgent()
    
    # 训练
    # for i in range(EPISODES):
    #     # 生成新的图
    #     state,goal,mark_value = generate_puzzle(BLOCK_NUM)
    #     # mark_value的值决定打开那个Q表csv文件
    #     if os.path.exists('models/Q_label{}.csv'.format(mark_value)):
    #         agent.load_Q('models/Q_label{}.csv'.format(mark_value))
    #     # 建立环境
    #     env = PuzzleEnv(state,goal)
    #     # agent 训练
    #     done,step_count = agent.train(env)
    #     # 判断是否收敛
    #     if done:
    #         if os.path.exists('models/Q_label{}.csv'.format(mark_value)):
    #             os.remove('models/Q_label{}.csv'.format(mark_value))
    #         agent.save_Q('models/Q_label{}.csv'.format(mark_value))
    #         print('episode: {}, solved in {} steps'.format(i,step_count))
            

    state,goal,mark_value = generate_puzzle(BLOCK_NUM)
    agent.load_Q('models/Q_label{}.csv'.format(mark_value))
    env = PuzzleEnv(state,goal)
    change_list = agent.solve(env)