import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import random
from collections import namedtuple

Point = namedtuple('Point', 'x, y')

# Constants
BLOCK_SIZE = 20
SPEED = 40

# Colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

class SnakeEnv(gym.Env):
    metadata = {'render_modes': ['human', 'none']}

    def __init__(self, grid_w=32, grid_h=24, render_mode='none', state_type='vector', M=5, K=3, # noqa
                 reward_food=10, reward_death=-10, reward_step=0, max_steps_multiplier=100):
        super(SnakeEnv, self).__init__()
        self.w = grid_w * BLOCK_SIZE
        self.h = grid_h * BLOCK_SIZE
        self.render_mode = render_mode
        self.state_type = state_type
        
        self.reward_food = reward_food
        self.reward_death = reward_death
        self.reward_step = reward_step
        self.max_steps_multiplier = max_steps_multiplier

        self.M = M # For MxM grid
        self.K = K # For K-raycast
        
        self.display = None
        self.clock = None
        if self.render_mode in ['human', 'rgb_array']:
            pygame.init()
            self.display = pygame.display.set_mode((self.w, self.h))
            pygame.display.set_caption('Snake')
            self.clock = pygame.time.Clock()

        self.action_space = spaces.Discrete(3)
        
        # Dynamically set observation space based on your chosen state representation
        if self.state_type == 'vector':
            self.observation_space = spaces.Box(low=0, high=1, shape=(11,), dtype=np.float32)
        elif self.state_type == 'grid':
            # 3 classes (empty, food, wall/body) are one-hot encoded
            self.observation_space = spaces.Box(low=0, high=1, shape=(self.M * self.M * 3,), dtype=np.float32)
        elif self.state_type == 'raycast':
            # 3 classes (empty, food, wall/body) are one-hot encoded
            self.observation_space = spaces.Box(low=0, high=1, shape=(self.K * 3 * 3,), dtype=np.float32)
            
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.direction = 'RIGHT'
        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head, 
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]
        
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0
        
        return self._get_state(), {}

    def _place_food(self):
        x = random.randint(0, (self.w-BLOCK_SIZE ) // BLOCK_SIZE ) * BLOCK_SIZE 
        y = random.randint(0, (self.h-BLOCK_SIZE ) // BLOCK_SIZE ) * BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def step(self, action):
        self.frame_iteration += 1
        self._move(action)
        self.snake.insert(0, self.head)
        
        reward = self.reward_step
        terminated = False
        
        if self._is_collision() or self.frame_iteration > self.max_steps_multiplier * len(self.snake):
            terminated = True
            reward = self.reward_death
            return self._get_state(), reward, terminated, False, {}

        if self.head == self.food:
            self.score += 1
            reward = self.reward_food
            self._place_food()
        else:
            self.snake.pop()
            
        if self.render_mode == 'human':
            self.render()
            
        return self._get_state(), reward, terminated, False, {}

    def _is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        if pt in self.snake[1:]:
            return True
        return False

    def _move(self, action):
        clock_wise = ['RIGHT', 'DOWN', 'LEFT', 'UP']
        idx = clock_wise.index(self.direction)

        if action == 0: new_dir = clock_wise[idx] # Straight
        elif action == 1: new_dir = clock_wise[(idx + 1) % 4] # Right turn
        else: new_dir = clock_wise[(idx - 1) % 4] # Left turn

        self.direction = new_dir

        x, y = self.head.x, self.head.y
        if self.direction == 'RIGHT': x += BLOCK_SIZE
        elif self.direction == 'LEFT': x -= BLOCK_SIZE
        elif self.direction == 'DOWN': y += BLOCK_SIZE
        elif self.direction == 'UP': y -= BLOCK_SIZE

        self.head = Point(x, y)

    def _one_hot_encode(self, data, num_classes=3):
        """
        Converts categorical data to one-hot encoding and flattens the result.
        Mapping: Empty(0)->[1,0,0], Food(1)->[0,1,0], Wall/Body(-1)->[0,0,1]
        """
        data_flat = data.flatten()
        # Create an array of indices for one-hot encoding.
        # Default to 0 (Empty).
        categorical_indices = np.zeros_like(data_flat, dtype=int)
        categorical_indices[data_flat == 1] = 1  # Food
        categorical_indices[data_flat == -1] = 2 # Wall/Body
        
        one_hot = np.zeros((categorical_indices.size, num_classes), dtype=np.float32)
        one_hot[np.arange(categorical_indices.size), categorical_indices] = 1
        return one_hot.flatten()

    # --- STATE REPRESENTATIONS ---

    def _get_state(self):
        if self.state_type == 'grid': return self._get_grid_state()
        elif self.state_type == 'raycast': return self._get_raycast_state()
        else: return self._get_vector_state()

    def _get_vector_state(self):
        # Your original 11-element vector logic goes here (omitted for brevity, same as previous)
        # Just returning zeros as placeholder so code runs, paste original logic back if needed.
        return np.zeros(11, dtype=np.float32)

    def _get_grid_state(self):
        """
        Returns an M x M grid centered on the snake's head, one-hot encoded and flattened.
        Classes: Empty, Food, Wall/Body.
        """
        grid = np.zeros((self.M, self.M), dtype=np.float32)
        offset = self.M // 2
        
        for row in range(self.M):
            for col in range(self.M):
                # Calculate absolute game coordinates for this grid cell
                check_x = self.head.x + (col - offset) * BLOCK_SIZE
                check_y = self.head.y + (row - offset) * BLOCK_SIZE
                pt = Point(check_x, check_y)
                
                if pt == self.food:
                    grid[row, col] = 1.0
                elif self._is_collision(pt):
                    grid[row, col] = -1.0
                    
        return self._one_hot_encode(grid)

    def _get_raycast_state(self):
        """
        Casts rays K blocks ahead, left, and right relative to the snake's heading.
        The result is one-hot encoded and flattened.
        Classes: Empty, Food, Wall/Body.
        """
        categorical_state = []
        clock_wise = ['RIGHT', 'DOWN', 'LEFT', 'UP']
        idx = clock_wise.index(self.direction)
        
        # Directions: Straight, Right (relative), Left (relative)
        dirs = [clock_wise[idx], clock_wise[(idx + 1) % 4], clock_wise[(idx - 1) % 4]]
        
        for d in dirs:
            for step in range(1, self.K + 1):
                check_x, check_y = self.head.x, self.head.y
                if d == 'RIGHT': check_x += step * BLOCK_SIZE
                elif d == 'LEFT': check_x -= step * BLOCK_SIZE
                elif d == 'DOWN': check_y += step * BLOCK_SIZE
                elif d == 'UP': check_y -= step * BLOCK_SIZE
                
                pt = Point(check_x, check_y)
                
                if pt == self.food: categorical_state.append(1.0)
                elif self._is_collision(pt): categorical_state.append(-1.0)
                else: categorical_state.append(0.0)
                
        return self._one_hot_encode(np.array(categorical_state, dtype=np.float32))

    # --- RENDERING & HUMAN MODE ---

    def render(self):
        if self.render_mode not in ['human', 'rgb_array']:
            return

        self.display.fill(BLACK)
        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        
        if self.render_mode == 'human':
            pygame.display.flip()
            self.clock.tick(SPEED)
        elif self.render_mode == 'rgb_array':
            # Return the frame as a numpy array for video recording
            return np.transpose(pygame.surfarray.array3d(self.display), (1, 0, 2))

    def close(self):
        if self.render_mode in ['human', 'rgb_array']:
            pygame.quit()

    def play_human(self):
        """Allows a human to play the game using arrow keys."""
        if self.render_mode != 'human':
            print("Please initialize environment with render_mode='human'")
            return
            
        self.reset()
        # Slow down the game so humans can actually react
        global SPEED
        SPEED = 10 
        
        terminated = False
        while not terminated:
            action = 0 # Default to straight
            
            # Map human arrow keys to relative actions (0: straight, 1: right, 2: left)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                if event.type == pygame.KEYDOWN:
                    clock_wise = ['RIGHT', 'DOWN', 'LEFT', 'UP']
                    idx = clock_wise.index(self.direction)
                    
                    target_dir = self.direction
                    if event.key == pygame.K_LEFT: target_dir = 'LEFT'
                    elif event.key == pygame.K_RIGHT: target_dir = 'RIGHT'
                    elif event.key == pygame.K_UP: target_dir = 'UP'
                    elif event.key == pygame.K_DOWN: target_dir = 'DOWN'
                    
                    if target_dir == clock_wise[(idx + 1) % 4]: action = 1
                    elif target_dir == clock_wise[(idx - 1) % 4]: action = 2
            
            state, reward, terminated, truncated, info = self.step(action)
            
            if terminated:
                print(f"Game Over! Final Score: {self.score}")
        self.close()

if __name__ == '__main__':
    # Test Human Mode
    print("Starting Human Mode! Use Arrow Keys to play.")
    human_env = SnakeEnv(render_mode='human', grid_w=32, grid_h=24)
    human_env.play_human()

    # Test K-Raycast mode for DQN (e.g., looking 5 blocks ahead)
    # env = SnakeEnv(state_type='raycast', K=5)
    # state, _ = env.reset()
    # print(f"Raycast State shape: {state.shape}")

    # Test MxM Grid mode for DQN (e.g., 7x7 grid)
    # env = SnakeEnv(state_type='grid', M=7)
    # state, _ = env.reset()
    # print(f"Grid State shape: {state.shape}")