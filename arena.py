import numpy as np
import pygame as pg
import tetrimino

class Arena:
    def __init__(self, size, renders):
        self.arena = np.zeros(size)
        self.score = 0
        self.size = size
        self.hidden_rows = 2
        self.block_size = 30
        self.moving_block = tetrimino.Tetrimino()
        self.next_moving_block = tetrimino.Tetrimino()
        self.color = (255, 255, 255)
        self.arena_only_ones = False
        self.running = True
        if renders:
            self.font = pg.font.SysFont("Comic Sans MS", 30)
        self.colors = [(20, 255, 255), (20, 20, 255), (255, 165, 20), (255, 255, 20), (128, 20, 128), (20, 128, 20), (255, 20, 20), (20, 20, 20)]

    def update_moving_blocks(self):
        moved = False
        if self.moving_block.bottom_y() < self.size[1] and not self.overlaps(0, 1):
            self.moving_block.y += 1
            moved = True
        self.adjust_block()
        return moved

    def render(self, screen):
        self.render_background(screen)
        self.render_arena(screen)
        self.render_score(screen)
        self.render_next_block(screen)

        for row in range(len(self.moving_block.shape)):
            for column in range(len(self.moving_block.shape[0])):
                if self.moving_block.shape[row][column]:
                    c = self.colors[ self.moving_block.number - 1]
                    self.render_square(screen, (self.moving_block.x+column)*self.block_size+self.block_size,
                                               (self.moving_block.y+row)*self.block_size,
                                                c)

    def render_background(self, screen):
        for row in range(510//self.block_size):
            for column in range(690//self.block_size):
                self.render_square(screen, row*self.block_size, column*self.block_size, (100, 100, 100))


    def render_arena(self, screen):
        for row in range(self.size[0]):
            for column in range(2, self.size[1]):
                if self.arena[row][column]:
                    c = self.colors[int(self.arena[row][column] - 1)]
                    self.render_square(screen, row*self.block_size+self.block_size, column*self.block_size, c)
                else:
                    self.render_square(screen, row*self.block_size+self.block_size, column*self.block_size)

    def render_score(self, screen):
        textsurface = self.font.render('Points: {}'.format(self.score), False, (0, 0, 0))
        screen.blit(textsurface, (self.size[0]*self.block_size*1.1+self.block_size, 35))

    def render_next_block(self, screen):
        for row in range(len(self.next_moving_block.shape)):
            for column in range(len(self.next_moving_block.shape[0])):
                if self.next_moving_block.shape[row][column]:
                    self.render_square(screen,
                                       self.size[0]*self.block_size*1.1 + column*self.block_size+self.block_size,
                                       100 +row * self.block_size,
                                       self.colors[self.next_moving_block.number - 1])


    def render_square(self, screen, x, y, color=(40, 40, 40)):
        outer_rim = self.block_size//15
        inner_square_size = self.block_size - outer_rim*2
        outer_color = np.subtract(color, (10, 10, 10))
        dark_outer_color = np.subtract(color, (20, 20, 20))
        pg.draw.rect(screen, color, (x, y, self.block_size, self.block_size))
        pg.draw.rect(screen, dark_outer_color, (x, y, self.block_size, outer_rim))
        pg.draw.rect(screen, dark_outer_color, (x, y, outer_rim, self.block_size))
        pg.draw.rect(screen, outer_color, (x + outer_rim, y + outer_rim, inner_square_size, inner_square_size))

    def add_moving_block(self):
        new_block = tetrimino.Tetrimino()
        self.moving_block = self.next_moving_block
        self.next_moving_block = new_block
        if self.overlaps(0, 0):
            self.running = False

    def move_block_right(self):
        if not self.overlaps(1, 0):
            self.moving_block.move_right()

    def move_block_left(self):
        if not self.overlaps(-1, 0):
            self.moving_block.move_left()

    def move_block_down(self):
        if not self.overlaps(0, 1):
            self.moving_block.move_down()
        else:
            print("mergu")
            self.check_block_merge()

    def rotate_block(self):
        self.moving_block.rotate()
        self.adjust_block()

    def place_block(self):
        for i in range(22):
            self.moving_block.move_down()
        self.check_block_merge()

    def check_block_merge(self):
        self.adjust_block()
        for row in range(len(self.moving_block.shape)):
            for column in range(len(self.moving_block.shape[0])):
                if self.moving_block.bottom_y() == self.size[1]:
                    self.merge()
                    self.add_moving_block()
                    return
                elif self.moving_block.shape[row][column] and \
                     self.arena[self.moving_block.x + column][self.moving_block.y + row + 1]:
                        self.merge()
                        self.add_moving_block()
                        return

    def merge(self):
        for row in range(len(self.moving_block.shape)):
            for column in range(len(self.moving_block.shape[0])):
                if self.moving_block.shape[row][column]:
                    if self.arena_only_ones:
                        self.arena[self.moving_block.x + column][self.moving_block.y + row] = 1
                    else:
                        self.arena[self.moving_block.x + column][self.moving_block.y + row] = self.moving_block.number
        self.check_row_filled()

    def check_row_filled(self):
        self.arena = [list(a) for a in zip(*self.arena)]
        points = 0
        changed = True
        while changed:
            changed = False
            for row in range(len(self.arena)):
                if 0 not in self.arena[row]:
                    del self.arena[row]
                    self.arena.insert(0, [0]*self.size[0])
                    changed = True
                    points += 100000
                    print("score!!")
        self.score += points
        self.arena = [list(a) for a in zip(*self.arena)]



    def adjust_block(self):
        adjusted = True
        attemps = 0
        while adjusted:
            attemps += 1
            adjusted = False
            for row in range(len(self.moving_block.shape)):
                for column in range(len(self.moving_block.shape[0])):
                    if self.moving_block.shape[row][column]:
                        if row + self.moving_block.y > self.size[1] - 1 or \
                           self.overlaps(0, 0):
                            self.moving_block.move_up()
                            adjusted = True
                            if attemps > 10:
                                self.running = False
                                return

    def overlaps(self, x, y):
        for row in range(len(self.moving_block.shape)):
            for column in range(len(self.moving_block.shape[0])):
                if self.moving_block.shape[row][column]:
                    if self.arena[(column + self.moving_block.x + x) % self.size[0]][(row + self.moving_block.y + y) % self.size[1]]:
                        return True

    def state(self):
        state = []
        arena_temp = list(map(list, self.arena))
        for row in range(len(self.moving_block.shape)):
            for column in range(len(self.moving_block.shape[0])):
                if self.moving_block.shape[row][column]:
                    arena_temp[self.moving_block.x + column][self.moving_block.y + row] = 1
        for row in arena_temp:
            state.extend(row)
        return state

    def bumpyness(self):
        arena_heights = []
        depth = 0
        for row in self.arena:
            for number in row:
                if number == 0:
                    depth += 1
                else:
                    break
            arena_heights.append(depth)
            depth = 0
        bumpyness = 0
        for i in range(len(arena_heights) - 1):
            bumpyness += abs(arena_heights[i] - arena_heights[i+1])

        return bumpyness

    def largest_height(self):
        arena_heights = []
        depth = 0
        for row in self.arena:
            for number in row:
                if number == 0:
                    depth += 1
                else:
                    break
            arena_heights.append(depth)
            depth = 0
        return max(arena_heights)

    def aggregate_height(self):
        arena_heights = []
        depth = 0
        for row in self.arena:
            for number in row:
                if number == 0:
                    depth += 1
                else:
                    break
            arena_heights.append(depth)
            depth = 0
        return sum(arena_heights)

    def state_easy(self):
        arena_state = []
        depth = 0
        for row in self.arena:
            for number in row:
                if number == 0:
                    depth += 1
                else:
                    break
            arena_state.append(depth)
            depth = 0
        block_state = [0]*10
        for x in range(self.moving_block.width() - 1):
            depth = 1
            for y in range(self.moving_block.height()-2, -1, -1):
                if self.moving_block.shape[x][y] == 0:
                    depth += 1
                else:
                    block_state[self.moving_block.x + x] = depth
                    break
        state = [*arena_state, *block_state]

        return state




