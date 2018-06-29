import pygame
import time
import settings as s
import arena
import random

class Tetris:
    def __init__(self):
        self.renders = False
        if self.renders:
            pygame.init()
            pygame.display.set_caption("tetris")
            self.screen = pygame.display.set_mode(s.screen_size)

        self.main_arena = arena.Arena(s.arena_size, renders=self.renders)
        self.steps = 0
        self.score = 0
        self.merge_delay = 5
        self.merge_counter = 0
        self.episode = 0


    def sample(self):
        random_value = random.randint(0, 3)
        return random_value

    def reset(self):
        self.episode += 1
        self.main_arena = arena.Arena(s.arena_size, self.renders)
        self.steps = 0
        self.score = 0
        self.merge_delay = 5
        self.merge_counter = 0
        return self.main_arena.state()

    def select_action(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        return 3
                    if event.key == pygame.K_RIGHT:
                        return 0
                    if event.key == pygame.K_LEFT:
                        return 1
                    if event.key == pygame.K_DOWN:
                        return 2

    def step(self, action):
        start_score = self.main_arena.score

        self.main_arena.update_moving_blocks()
        self.main_arena.check_block_merge()


        self.steps += 1
        self.merge_counter += 1
        if action == 0:
            self.main_arena.move_block_right()
        if action == 1:
            self.main_arena.move_block_left()
        if action == 3:
            self.main_arena.rotate_block()
        if action == 2:
            self.main_arena.place_block()

        reward = self.main_arena.score - start_score - self.main_arena.bumpyness()//2 - self.main_arena.aggregate_height()//5 + self.steps//2

        if self.renders and self.episode % self.episode == 10 :
            self.render()

        return self.main_arena.state(), reward, self.main_arena.running, None



    def render(self):
        self.main_arena.render(self.screen)
        pygame.display.flip()

def main():
    pygame.init()
    pygame.display.set_caption("tetris")
    screen = pygame.display.set_mode(s.screen_size)
    running = True
    step = 0
    score = 0
    merge_delay = 20
    merge_counter = 0

    main_arena = arena.Arena(s.arena_size, renders=True)
    while running:
        if step % 10 == 0:
            if main_arena.update_moving_blocks():
                merge_counter = 0
        if merge_counter > merge_delay:
            main_arena.check_block_merge()
            merge_counter = 0
        time.sleep(0.03)
        main_arena.render(screen)
        pygame.display.flip()
        screen.fill((220, 220, 220))
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    main_arena.rotate_block()
                if event.key == pygame.K_RIGHT:
                    main_arena.move_block_right()
                if event.key == pygame.K_LEFT:
                    main_arena.move_block_left()
                if event.key == pygame.K_DOWN:
                    main_arena.place_block()
                if event.key == pygame.K_0:
                    pass
                    # main_arena.add_moving_block()
        step += 1
        merge_counter += 1
        running = main_arena.running

if __name__ == '__main__':
    main()