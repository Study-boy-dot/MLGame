"""
The template of the script for playing the game in the ml mode
"""
import torch
import random
import numpy as np
from collections import deque
from enum import Enum
# from game import SnakeGameAI, Direction, Point
from .model import Linear_QNet, QTrainer
from .helper import plot
# import agent
from collections import namedtuple
import pygame

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001
SCENE_WIDTH = 300
SCENE_HEIGHT = 300

Point = namedtuple('Point', 'x, y')

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

class MLPlay:
    def __init__(self):
        """
        Constructor
        """
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.6 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        # self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(Linear_QNet(11, 256, 3), lr=LR, gamma=self.gamma)
        self.state_old = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.direction = Direction.DOWN
        self.highest_score = 0
        self.previous_food = (0,0)
        self.final_move_old = [1,0,0]
        self.total_score = 0
        self.plot_scores = []
        self.plot_mean_scores = []


    def update(self, scene_info):
        """
        Generate the command according to the received scene information
        """
        # initialize
        self.snake_head = scene_info["snake_head"]
        self.food = scene_info["food"]
        self.snake_body = scene_info['snake_body']
        score = len(self.snake_body) + 1 -4
        reward = 0
        
        if self.snake_head == self.previous_food:
            reward = 10
        if scene_info["status"] == "GAME_OVER":
            print("game over")
            reward = -10

        # get old state
        state_new = self.get_state()
        
        # get move
        final_move = self.get_action(state_new)
        # print(final_move,self.direction)
        
        # train short memory
        self.train_short_memory(self.state_old, self.final_move_old, reward, state_new, False if scene_info["status"] != "GAME_OVER" else True)

        # remember
        self.remember(self.state_old, self.final_move_old, reward, state_new, False if scene_info["status"] != "GAME_OVER" else True)

        self.state_old = state_new
        self.previous_food = self.food
        self.final_move_old = final_move

        if scene_info["status"] == "GAME_OVER":
            self.n_games += 1
            self.train_long_memory()
            self.direction = Direction.DOWN
            if self.highest_score < score:
                self.highest_score = score
                self.trainer.model.save(f'model{self.n_games}.pth')
                self.trainer.model.save('best_model.pth')
            print('Games:',self.n_games,'Highest Score:',self.highest_score)
            self.plot_scores.append(score)
            self.total_score += score
            mean_score = self.total_score / self.n_games
            self.plot_mean_scores.append(mean_score)
            plot(self.plot_scores, self.plot_mean_scores, 'Training Score 11 states.jpg')
            return "RESET"

        del self.snake_head
        del self.food
        del self.snake_body
        if self.direction == Direction.UP:
            if np.array_equal(final_move, [0, 1, 0]):
                self.direction = Direction.LEFT
                return "LEFT"
            elif np.array_equal(final_move, [0, 0, 1]):
                self.direction = Direction.RIGHT
                return "RIGHT"
            else:
                return "UP"
        elif self.direction == Direction.DOWN:
            if np.array_equal(final_move, [0, 1, 0]):
                self.direction = Direction.RIGHT
                return "RIGHT"
            elif np.array_equal(final_move, [0, 0, 1]):
                self.direction = Direction.LEFT
                return "LEFT"
            else:
                return "DOWN"
        elif self.direction == Direction.LEFT:
            if np.array_equal(final_move, [0, 1, 0]):
                self.direction = Direction.DOWN
                return "DOWN"
            elif np.array_equal(final_move, [0, 0, 1]):
                self.direction = Direction.UP
                return "UP"
            else:
                return "LEFT"
        elif self.direction == Direction.RIGHT:
            if np.array_equal(final_move, [0, 1, 0]):
                self.direction = Direction.UP
                return "UP"
            elif np.array_equal(final_move, [0, 0, 1]):
                self.direction = Direction.DOWN
                return "DOWN"
            else:
                return "RIGHT"
        # if self.snake_head[0] > self.food[0]:
        #     return "LEFT"
        # elif self.snake_head[0] < self.food[0]:
        #     return "RIGHT"
        # elif self.snake_head[1] > self.food[1]:
        #     return "UP"
        # elif self.snake_head[1] < self.food[1]:
        #     return "DOWN"

    def reset(self):
        """
        Reset the status if needed
        """
        self.state_old = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.direction = Direction.DOWN
        self.previous_food = (0,0)
        self.final_move_old = [1,0,0]
        return "RESET"

    def is_collision(self, pt=None):
        """
        Check collision
        """
        # hits boundary
        if pt[0] >= SCENE_WIDTH or pt[0] < 0:
            return True
        if pt[1] >= SCENE_HEIGHT or pt[1] < 0:
            return True

        # hits body
        if pt in self.snake_body:
            return True
        
        return False
    def get_state(self):
        point_l = (self.snake_head[0] - 10, self.snake_head[1])
        point_r = (self.snake_head[0] + 10, self.snake_head[1])
        point_u = (self.snake_head[0], self.snake_head[1] - 10)
        point_d = (self.snake_head[0], self.snake_head[1] + 10)

        dir_l = self.direction == Direction.LEFT
        dir_r = self.direction == Direction.RIGHT
        dir_u = self.direction == Direction.UP
        dir_d = self.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and self.is_collision(point_r)) or 
            (dir_l and self.is_collision(point_l)) or 
            (dir_u and self.is_collision(point_u)) or 
            (dir_d and self.is_collision(point_d)),

            # Danger right
            (dir_u and self.is_collision(point_r)) or 
            (dir_d and self.is_collision(point_l)) or 
            (dir_l and self.is_collision(point_u)) or 
            (dir_r and self.is_collision(point_d)),

            # Danger left
            (dir_d and self.is_collision(point_r)) or 
            (dir_u and self.is_collision(point_l)) or 
            (dir_r and self.is_collision(point_u)) or 
            (dir_l and self.is_collision(point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            self.food[0] < self.snake_head[0],  # food left
            self.food[0] > self.snake_head[0],  # food right
            self.food[1] < self.snake_head[1],  # food up
            self.food[1] > self.snake_head[1]  # food down
            ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        #for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.trainer.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move

