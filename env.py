import numpy as np

class WarehouseEnv:
    def __init__(self, grid_size=(6, 6), max_steps=100):
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.reset()

    def reset(self):
        self.grid = np.zeros(self.grid_size, dtype=int)
        self.agent_pos = [0, 0]
        self.pickup_pos = [self.grid_size[0] - 1, 0]
        self.delivery_pos = [self.grid_size[0] - 1, self.grid_size[1] - 1]
        self.has_package = False
        self.steps = 0

        self.grid[2, 2] = -1  # obstacle
        self.grid[3, 1] = -1  # obstacle

        return self._get_state()

    def _get_state(self):
        return np.array(self.agent_pos + [int(self.has_package)], dtype=np.float32)

    def step(self, action):
        row, col = self.agent_pos
        if action == 0: row -= 1  # up
        elif action == 1: row += 1  # down
        elif action == 2: col -= 1  # left
        elif action == 3: col += 1  # right

        # Clamp position within grid
        row = np.clip(row, 0, self.grid_size[0] - 1)
        col = np.clip(col, 0, self.grid_size[1] - 1)

        if self.grid[row, col] != -1:  # not hitting obstacle
            self.agent_pos = [row, col]

        self.steps += 1
        done = False
        reward = -0.2  # small penalty for step

        if not self.has_package and self.agent_pos == self.pickup_pos:
            self.has_package = True
            reward = +20  # reward for picking up package

        elif self.has_package and self.agent_pos == self.delivery_pos:
            reward = +50  # reward for delivering package
            done = True

        # Shaping reward: encourage moving closer to goal
        target = self.delivery_pos if self.has_package else self.pickup_pos
        distance = np.linalg.norm(np.array(self.agent_pos) - np.array(target))
        reward += -0.05 * distance

        if self.steps >= self.max_steps:
            done = True

        return self._get_state(), reward, done, {}

    def render(self):
        view = np.array(self.grid, dtype=str)
        view[view == "0"] = "."
        view[view == "-1"] = "#"

        r, c = self.agent_pos
        pr, pc = self.pickup_pos
        dr, dc = self.delivery_pos

        view[pr, pc] = "P"
        view[dr, dc] = "D"
        view[r, c] = "A"

        print("\n".join(" ".join(row) for row in view))
        print()
