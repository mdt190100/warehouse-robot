# env.py
import numpy as np

class WarehouseEnv:
    def __init__(self, grid_size=(6, 6), max_steps=100):
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.reset()

    def reset(self):
        self.grid = np.zeros(self.grid_size, dtype=int)

        # Khởi tạo các vị trí
        self.agent_pos = [0, 0]  # Điểm xuất phát
        self.pickup_pos = [self.grid_size[0] - 1, 0]   # Vị trí lấy hàng (ví dụ dưới trái)
        self.delivery_pos = [self.grid_size[0] - 1, self.grid_size[1] - 1]  # Vị trí giao hàng (ví dụ dưới phải)

        self.has_package = False
        self.steps = 0

        # Vật cản
        self.grid[2, 2] = -1
        self.grid[3, 1] = -1

        return self._get_state()

    def _get_state(self):
        # Trạng thái = vị trí robot + cờ đã lấy hàng hay chưa
        return np.array(self.agent_pos + [int(self.has_package)], dtype=np.float32)

    def step(self, action):
        """
        Actions: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
        """
        row, col = self.agent_pos
        if action == 0: row -= 1
        elif action == 1: row += 1
        elif action == 2: col -= 1
        elif action == 3: col += 1

        # Giữ trong giới hạn
        row = np.clip(row, 0, self.grid_size[0] - 1)
        col = np.clip(col, 0, self.grid_size[1] - 1)

        # Tránh vật cản
        if self.grid[row, col] != -1:
            self.agent_pos = [row, col]

        self.steps += 1
        done = False
        reward = -0.2  # Phạt nhẹ mỗi bước để tối ưu hoá đường đi

        # Đến chỗ lấy hàng
        if not self.has_package and self.agent_pos == self.pickup_pos:
            self.has_package = True
            reward = +5  # Thưởng khi nhặt hàng

        # Giao hàng thành công
        elif self.has_package and self.agent_pos == self.delivery_pos:
            reward = +10
            done = True

        # Hết bước thì kết thúc
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

        view[pr, pc] = "P"  # Pickup
        view[dr, dc] = "D"  # Delivery
        view[r, c] = "A"    # Agent

        print("\n".join(" ".join(row) for row in view))
        print()
