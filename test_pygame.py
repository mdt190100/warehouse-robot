import pygame
pygame.init()
screen = pygame.display.set_mode((400, 400))
pygame.display.set_caption("Test Window")

running = True
while running:
    screen.fill((30, 30, 30))
    pygame.draw.circle(screen, (255, 0, 0), (200, 200), 50)
    pygame.display.flip()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

pygame.quit()
