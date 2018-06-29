import random

i_block = [[1,1,1,1]]

j_block = [[2,0,0],
           [2,2,2]]

l_block = [[0,0,3],
           [3,3,3]]

o_block = [[4,4],
           [4,4]]

t_block = [[0,5,0],
           [5,5,5]]

s_block = [[0,6,6],
           [6,6,0]]

z_block = [[7,7,0],
           [0,7,7]]

dot_block = [[1]]

blocks = [i_block, j_block, l_block, o_block, s_block, t_block, z_block, dot_block]


class Tetrimino:
    def __init__(self, x=4 , y = 2):
        self.number = random.randint(1, 7)
        self.shape = blocks[self.number - 1].copy()
        self.x = x
        self.y = y
        self.color = (30, 200, 50)
        self.right_x = lambda: self.x + len(self.shape[0])
        self.bottom_y = lambda: self.y + len(self.shape)
        self.width = lambda: len(self.shape[0])
        self.height = lambda: len(self.shape)

    def rotate(self):
        self.shape = [list(a) for a in zip(*reversed(self.shape))]
        y_change = len(self.shape[0]) - len(self.shape)
        if y_change > 0:
            self.y += y_change
        change = True
        while change:
            if self.right_x() > 10:
                self.x -= 1
            elif self.x < 0:
                self.x += 1
            else:
                change = False

    def move_right(self):
        if self.right_x() < 10:
            self.x += 1

    def move_left(self):
        if self.x > 0:
            self.x -= 1

    def move_down(self):
        if self.bottom_y() < 22:
            self.y += 1

    def move_up(self):
        if self.y > 0:
            self.y -= 1
