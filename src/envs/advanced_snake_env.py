import numpy as np
import pygame
import random
from collections import deque
import gymnasium as gym
from gymnasium import spaces


class AdvancedSnakeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, rewards, width=20, height=20, block_size=20, render_mode=False, max_steps=400):
        super().__init__()
        self.rewards = rewards
        self.width = width
        self.height = height
        self.block_size = block_size
        self.render_mode = render_mode
        self.max_steps = max_steps

        self.color_black = (0, 0, 0)
        self.color_green = (0, 255, 0)
        self.color_red = (255, 0, 0)
        self.color_gray = (40, 40, 40)

        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(9,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)    # Update the action space

        self.screen = None
        self.clock = None

        self.reset()

        if self.render_mode:
            pygame.init()
            self.screen = pygame.display.set_mode((self.width * self.block_size, self.height * self.block_size))
            pygame.display.set_caption("Advanced Snake")
            self.clock = pygame.time.Clock()

    def reset(self, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.direction = random.choice(["UP", "DOWN", "LEFT", "RIGHT"])
        self.snake = deque([(self.width // 2, self.height // 2)])
        self.snake_length = 3
        self.spawn_food()
        self.score = 0
        self.steps = 0
        self.done = False
        self.prev_distance = self._distance(self.snake[0], self.food)
        return self._get_state(), {}

    def spawn_food(self):
        while True:
            pos = (random.randint(0, self.width - 1), random.randint(0, self.height - 1))
            if pos not in self.snake:
                self.food = pos
                break

    def _distance(self, p1, p2):
        return np.abs(p1[0] - p2[0]) + np.abs(p1[1] - p2[1])  # Use Manhattan distance

    def _get_state(self):
        head_x, head_y = self.snake[0]
        food_dx = (self.food[0] - head_x) / max(1, self.width - 1)
        food_dy = (self.food[1] - head_y) / max(1, self.height - 1)

        danger_straight = int(self._is_danger(self.direction))
        danger_left = int(self._is_danger(self._turn_left(self.direction)))
        danger_right = int(self._is_danger(self._turn_right(self.direction)))

        dir_up = int(self.direction == "UP")
        dir_down = int(self.direction == "DOWN")
        dir_left = int(self.direction == "LEFT")
        dir_right = int(self.direction == "RIGHT")

        obs = np.array(
            [dir_up, dir_down, dir_left, dir_right,
             food_dx, food_dy, danger_straight, danger_left, danger_right],
            dtype=np.float32,
        )
        return obs

    def _is_danger(self, direction):
        head_x, head_y = self.snake[0]
        if direction == "UP":
            head_y -= 1
        elif direction == "DOWN":
            head_y += 1
        elif direction == "LEFT":
            head_x -= 1
        elif direction == "RIGHT":
            head_x += 1

        if head_x < 0 or head_x >= self.width or head_y < 0 or head_y >= self.height:
            return True
        if (head_x, head_y) in list(self.snake)[:-1]:
            return True
        return False

    def _turn_left(self, direction):
        return {"UP": "LEFT", "LEFT": "DOWN", "DOWN": "RIGHT", "RIGHT": "UP"}[direction]

    def _turn_right(self, direction):
        return {"UP": "RIGHT", "RIGHT": "DOWN", "DOWN": "LEFT", "LEFT": "UP"}[direction]

    def step(self, action):
        # --- Handle turn actions ---
        if action == 0:
            self.direction = self._turn_left(self.direction)
            reward = self.rewards["turn"]
        elif action == 1:
            self.direction = self._turn_right(self.direction)
            reward = self.rewards["turn"]
        else:
            reward = self.rewards["forward"]

        # --- Move snake head ---
        head_x, head_y = self.snake[0]
        if self.direction == "UP":
            head_y -= 1
        elif self.direction == "DOWN":
            head_y += 1
        elif self.direction == "LEFT":
            head_x -= 1
        elif self.direction == "RIGHT":
            head_x += 1

        new_head = (head_x, head_y)

        # --- Initialize reward and flags ---
        terminated, truncated = False, False

        # --- Collision check (death) ---
        if (
            head_x < 0 or head_x >= self.width or
            head_y < 0 or head_y >= self.height or
            new_head in list(self.snake)[:-1]
        ):
            terminated = True
            reward += self.rewards["collision"]
            return self._get_state(), float(reward), terminated, truncated, {"score": self.score}

        # --- Normal movement ---
        self.snake.appendleft(new_head)

        # --- Food eaten ---
        if new_head == self.food:
            reward += self.rewards["food_eaten"]
            self.score += 1
            self.snake_length += 1
            self.spawn_food()
        else:
            # maintain correct length
            if len(self.snake) > self.snake_length:
                self.snake.pop()

        # --- Distance-based shaping reward ---
        new_dist = self._distance(new_head, self.food)
        if new_dist < self.prev_distance:
            reward += self.rewards["move_closer"]   # moved closer to food
        else:
            reward += self.rewards["move_away"]  # moved away slightly (less penalty than before)
        self.prev_distance = new_dist

        # --- Step counter ---
        self.steps += 1
        if self.steps >= self.max_steps:
            truncated = True

        return self._get_state(), float(reward), terminated, truncated, {"score": self.score}

    def render(self, mode="human"):
        if not self.render_mode and mode != "rgb_array":
            return None

        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.width * self.block_size, self.height * self.block_size))
            pygame.display.set_caption("Snake RL Environment")
            self.clock = pygame.time.Clock()

        pygame.event.pump()
        self.screen.fill(self.color_black)

        for x in range(0, self.width * self.block_size, self.block_size):
            for y in range(0, self.height * self.block_size, self.block_size):
                pygame.draw.rect(self.screen, self.color_gray, pygame.Rect(x, y, self.block_size, self.block_size), 1)

        for sx, sy in self.snake:
            pygame.draw.rect(self.screen, self.color_green,
                             pygame.Rect(sx * self.block_size, sy * self.block_size, self.block_size, self.block_size))

        fx, fy = self.food
        pygame.draw.rect(self.screen, self.color_red,
                         pygame.Rect(fx * self.block_size, fy * self.block_size, self.block_size, self.block_size))

        pygame.display.flip()
        if hasattr(self, "clock"):
            self.clock.tick(self.metadata.get("render_fps", 10))

        if mode == "rgb_array":
            arr = pygame.surfarray.array3d(self.screen)
            return np.transpose(arr, (1, 0, 2))
        return None

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
            self.clock = None

