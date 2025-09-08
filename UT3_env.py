from enum import Enum
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch
import torch.nn as nn

DEBUG = 0 

import sys
import os
from UT3 import *

### Reward Table
INVALID_MOVE_REWARD = -10
STALL_REWARD = 0
TIE_REWARD = 0.5
WIN_REWARD = 1
LOSS_REWARD = -1

class UT3_Env(gym.Env):
    metadata = {"render_modes": ["ascii"], "render_fps": 4}

    def __init__(self, render_mode="ascii"):
        super().__init__()
        self.game = UltimateTicTacToe()
        self.render_mode = render_mode
        
        # Espacio de observación
        self.observation_space = spaces.Dict({
            "board": spaces.MultiDiscrete([3] * 81),  # 9x9 grid (0=empty, 1=X, 2=O)
            "active_board": spaces.MultiDiscrete([3, 3]),  # Which board is active
            "current_player": spaces.Discrete(2)  # 0=X, 1=O
        })
        
        # Espacio de acción: 81 movimientos posibles (9 boards × 9 cells)
        self.action_space = spaces.Discrete(81)
        
        # Para mapear acciones a movimientos
        self.action_map = {}
        idx = 0
        for br in range(3):
            for bc in range(3):
                for cr in range(3):
                    for cc in range(3):
                        self.action_map[idx] = (br, bc, cr, cc)
                        idx += 1

    def _get_obs(self):
        # Convert board state to numerical representation
        board_state = np.zeros(81, dtype=int)
        idx = 0
        for br in range(3):
            for bc in range(3):
                small_board = self.game.boards[br][bc]
                for cr in range(3):
                    for cc in range(3):
                        cell = small_board.board[cr][cc]
                        if cell == CellState.CROSS:
                            board_state[idx] = 1
                        elif cell == CellState.CIRCLE:
                            board_state[idx] = 2
                        idx += 1
        
        # Get active board
        if self.game.active_board:
            active_board = list(self.game.active_board)
        else:
            active_board = [0, 0]  # Default if no active board
            
        return {
            "board": board_state,
            "active_board": np.array(active_board, dtype=int),
            "current_player": 0 if self.game.current_player == CellState.CROSS else 1
        }

    def _get_info(self):
        return {"valid_moves": self.get_valid_moves()}
    
    def get_valid_moves(self):
        valid_moves = []
        
        # Determinar qué tableros están activos
        if self.game.active_board is None:
            # Todos los tableros están disponibles si no hay tablero activo
            active_boards = [(i, j) for i in range(3) for j in range(3)]
        else:
            # Verificar si el tablero activo ya está ganado
            br, bc = self.game.active_board
            if self.game.boards[br][bc].winner is not None:
                # Si el tablero activo está ganado, cualquier tablero no ganado es válido
                active_boards = [(i, j) for i in range(3) for j in range(3) 
                            if self.game.boards[i][j].winner is None]
            else:
                # Solo el tablero específico está activo
                active_boards = [self.game.active_board]
        
        # Iterar solo sobre los tableros activos
        for board_row, board_col in active_boards:
            small_board = self.game.boards[board_row][board_col]
            
            # Verificar si este tablero ya tiene un ganador
            if small_board.winner is not None:
                continue  # Saltar tableros ya ganados
            
            # Iterar sobre las celdas de este tablero
            for cell_row in range(3):
                for cell_col in range(3):
                    if small_board.board[cell_row][cell_col] == CellState.BLANK:
                        # Calcular el índice de la acción
                        action = board_row * 27 + board_col * 9 + cell_row * 3 + cell_col
                        valid_moves.append(action)
        
        return valid_moves

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game = UltimateTicTacToe()
        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action):
        # Convert action to move
        br, bc, cr, cc = self.action_map[action]
        move = UltimateMove(br, bc, cr, cc, self.game.current_player)
        
        # Check if move is valid
        valid_moves = self.get_valid_moves()
        if action not in valid_moves:
            observation = self._get_obs()
            return observation, INVALID_MOVE_REWARD, False, False, valid_moves
        
        # Make the move
        self.game.make_move(move)
        if DEBUG: self.game.display()

        # Check game status
        winner = self.game.winner
        terminated = self.game.winner is not None
        
        # Calculate reward
        if winner is None:
            reward = STALL_REWARD
        elif winner == CellState.BLANK:
            reward = TIE_REWARD
        elif winner == self.game.current_player:
            reward = WIN_REWARD
        else:
            reward = LOSS_REWARD
        
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "ascii":
            self.game.display()
