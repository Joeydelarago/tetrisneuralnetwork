import pygame as pg
import tetrimino

def spawn_block(arena):
    block = tetrimino.Tetrimino()
    arena.add_moving_block(block)

def check_events():
    pass
