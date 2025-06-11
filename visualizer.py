# visualizer.py
import pygame
import sys

class WarehouseVisualizer:
    def __init__(self, env, cell_size=80):
        self.env = env
        self.cell_size = cell_size
        self.width = env.grid_size[1] * cell_size
        self.height = env.grid_size[0] * cell_size
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Warehouse Robot")

    def draw(self):
        self.screen.fill((255, 255, 255))  # White background

        for row in range(self.env.grid_size[0]):
            for col in range(self.env.grid_size[1]):
                rect = pygame.Rect(col * self.cell_size, row * self.cell_size, self.cell_size, self.cell_size)
                cell = self.env.grid[row, col]

                # Draw obstacle
                if cell == -1:
                    pygame.draw.rect(self.screen, (0, 0, 0), rect)  # Black box
                else:
                    pygame.draw.rect(self.screen, (200, 200, 200), rect, 1)  # Gray border

        # Draw target (green)
        tr, tc = self.env.target_pos
        pygame.draw.rect(
            self.screen, (0, 255, 0),
            pygame.Rect(tc * self.cell_size, tr * self.cell_size, self.cell_size, self.cell_size)
        )

        # Draw agent (red circle)
        r, c = self.env.agent_pos
        center = (c * self.cell_size + self.cell_size // 2, r * self.cell_size + self.cell_size // 2)
        pygame.draw.circle(self.screen, (255, 0, 0), center, self.cell_size // 3)

        pygame.display.flip()
        pygame.time.delay(200)

    def quit(self):
        pygame.quit()
