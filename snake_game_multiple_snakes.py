import pygame
import random
from enum import Enum
from collections import namedtuple

pygame.init()
font = pygame.font.SysFont('arial', 25)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4
    
Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
RED = (200,0,0)
MAIN = [(0, 0, 255), (0, 255, 0)]
BORDER = [(0, 100, 255), (100, 255, 0)]
BLACK = (0,0,0)
GREY = (128,128,128)

BLOCK_SIZE = 20
SPEED = 20

class SnakeGame:
    def __init__(self, w=640, h=480, n_snakes=1):
        self.w = w
        self.h = h
        # Add option to have multiple snakes, and initialize each one to be a list of None elements for each snake
        # These value should all be set once the reset() function is called
        self.n_snakes = n_snakes
        # All snakes start facing right
        self.directions = [Direction.RIGHT] * n_snakes
        self.heads = [None] * n_snakes
        self.snakes = [None] * n_snakes
        # All snakes start with 0 score
        self.scores = [0] * n_snakes
        # All snakes start with no food set
        self.foods = [None] * n_snakes
        # All snakes start with 0 iterations played
        self.frame_iterations = [0] * n_snakes
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset()
        
    # init game state
    def reset(self):
        for i in range(self.n_snakes):
            self.directions[i] = Direction.RIGHT
            
            self.heads[i] = Point(self.w/2, self.h/2)
            self.snakes[i] = [self.heads[i], 
                        Point(self.heads[i].x-BLOCK_SIZE, self.heads[i].y),
                        Point(self.heads[i].x-(2*BLOCK_SIZE), self.heads[i].y)]
            
            self.scores[i] = 0
            self.foods[i] = None
            self._place_food(i)
            self.frame_iterations[i] = 0

    def _place_food(self, i):
        x = random.randint(0, (self.w-BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE 
        y = random.randint(0, (self.h-BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.foods[i] = Point(x, y)
        if self.foods[i] in self.snakes[i]:
            self._place_food(i)
        
    def play_step(self, i):
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    self.directions[i] = Direction.LEFT
                elif event.key == pygame.K_RIGHT:
                    self.directions[i] = Direction.RIGHT
                elif event.key == pygame.K_UP:
                    self.directions[i] = Direction.UP
                elif event.key == pygame.K_DOWN:
                    self.directions[i] = Direction.DOWN
        
        # 2. move
        self._move(self.directions[i], i) # update the head
        self.snakes[i].insert(0, self.heads[i])
        
        # 3. check if game over
        game_over = False
        if self._is_collision(i):
            game_over = True
            return game_over, self.scores[i]
            
        # 4. place new food or just move
        if self.heads[i] == self.foods[i]:
            self.scores[i] += 1
            self._place_food(i)
        else:
            self.snakes[i].pop()
        
        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)
        # 6. return game over and score
        return game_over, self.scores[i]
    
    def _is_collision(self, i):
        head = self.heads[i]
        snake = self.snakes[i]
        # hits boundary
        if head.x > self.w - BLOCK_SIZE or head.x < 0 or head.y > self.h - BLOCK_SIZE or head.y < 0:
            return True
        # hits itself
        if head in snake[1:]:
            return True
        
        return False
        
    def _update_ui(self):
        self.display.fill(BLACK)
        
        # Get 9x9 with the head at its center
        # c = 0
        # for i in range(-4, 5):
        #     for j in range(-4, 5):
        #         rect_x = max(self.head.x + i * BLOCK_SIZE, 0)
        #         rect_y = max(self.head.y + j * BLOCK_SIZE, 0)
        #         pygame.draw.rect(self.display, GREY, pygame.Rect(rect_x, rect_y, BLOCK_SIZE, BLOCK_SIZE))

        #         # text_surface = font.render(str(c), True, WHITE)
        #         # text_rect = text_surface.get_rect(center=(rect_x + BLOCK_SIZE // 2, rect_y + BLOCK_SIZE // 2))

        #         # # Draw the text onto the rectangle
        #         # self.display.blit(text_surface, text_rect)
        #         c += 1

        # print(f"C: {c}")
        # for pt in self.snake:
        for i in range(self.n_snakes):
            snake = self.snakes[i]
            food = self.foods[i]
            for pt in snake:
                pygame.draw.rect(self.display, MAIN[i % len(MAIN)], pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
                pygame.draw.rect(self.display, BORDER[i % len(BORDER)], pygame.Rect(pt.x+4, pt.y+4, 12, 12))
            
            pygame.draw.rect(self.display, RED, pygame.Rect(food.x, food.y, BLOCK_SIZE, BLOCK_SIZE))
        
        text = font.render("Scores: " + str(self.scores), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()
        
    def _move(self, direction, i):
        head = self.heads[i]
        x = head.x
        y = head.y
        if direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif direction == Direction.UP:
            y -= BLOCK_SIZE
            
        self.heads[i] = Point(x, y)
            

if __name__ == '__main__':
    n_snakes = 1
    game = SnakeGame(n_snakes=n_snakes)
    
    # game loop
    while True:
        game_overs = [False] * n_snakes
        for i in range(n_snakes):
            game_over, score = game.play_step(i)
            game_overs[i] = game_over
        
        if game_overs.count(True) == n_snakes:
            break
        
    print('Final Score', score)
        
        
    pygame.quit()