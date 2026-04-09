"""
Ultimate TicTacToe - Entorno Gymnasium Corregido
Correcciones:
- Recompensas calculadas ANTES del cambio de turno
- Observación active_board usa -1 para indicar "cualquier tablero"
- Info siempre retorna diccionario
- Soporte para self-play
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Dict, Any, List, Optional

from UT3 import UltimateTicTacToe, UltimateMove, CellState

# Recompensas
INVALID_MOVE_REWARD = -10
MOVE_REWARD = 0.0
WIN_REWARD = 1.0
LOSS_REWARD = -1.0
TIE_REWARD = 0.0

# Recompensas intermedias (shaping)
WIN_SMALL_BOARD = 0.1
LOSE_SMALL_BOARD = -0.05


class UT3_Env(gym.Env):
    """
    Entorno para Ultimate TicTacToe.
    
    Observación:
        - board: array de 81 valores (0=vacío, 1=X, 2=O)
        - meta_board: array de 9 valores (estado de cada sub-tablero)
        - active_board: array de 2 valores (fila, col) o (-1, -1) si cualquiera
        - current_player: 0=X, 1=O
    
    Acción:
        - Entero 0-80 que representa board_row*27 + board_col*9 + cell_row*3 + cell_col
    """
    
    metadata = {"render_modes": ["ascii", "human"], "render_fps": 4}

    def __init__(self, render_mode: str = "ascii", opponent: str = "random"):
        """
        Args:
            render_mode: "ascii" o "human"
            opponent: "random", "self", o "none"
        """
        super().__init__()
        self.game = UltimateTicTacToe()
        self.render_mode = render_mode
        self.opponent = opponent
        
        # El agente siempre juega como X (primer jugador)
        self.agent_player = CellState.CROSS
        
        # Espacio de observación
        self.observation_space = spaces.Dict({
            "board": spaces.Box(low=0, high=2, shape=(81,), dtype=np.int8),
            "meta_board": spaces.Box(low=0, high=2, shape=(9,), dtype=np.int8),
            "active_board": spaces.Box(low=-1, high=2, shape=(2,), dtype=np.int8),
            "current_player": spaces.Discrete(2),
        })
        
        # Espacio de acción: 81 movimientos posibles
        self.action_space = spaces.Discrete(81)
        
        # Mapeo acción -> coordenadas
        self.action_to_coords = {}
        self.coords_to_action = {}
        idx = 0
        for br in range(3):
            for bc in range(3):
                for cr in range(3):
                    for cc in range(3):
                        self.action_to_coords[idx] = (br, bc, cr, cc)
                        self.coords_to_action[(br, bc, cr, cc)] = idx
                        idx += 1
        
        # Tracking para reward shaping
        self._prev_small_wins = {CellState.CROSS: 0, CellState.CIRCLE: 0}

    def _count_small_wins(self) -> Dict[CellState, int]:
        """Cuenta tableros pequeños ganados por cada jugador."""
        counts = {CellState.CROSS: 0, CellState.CIRCLE: 0}
        for i in range(3):
            for j in range(3):
                winner = self.game.boards[i][j].winner
                if winner in counts:
                    counts[winner] += 1
        return counts

    def _get_obs(self) -> Dict[str, np.ndarray]:
        """Construye el diccionario de observación."""
        # Board state: 81 valores
        board_state = np.zeros(81, dtype=np.int8)
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
        
        # Meta board state: 9 valores
        meta_state = np.zeros(9, dtype=np.int8)
        for i in range(3):
            for j in range(3):
                winner = self.game.boards[i][j].winner
                if winner == CellState.CROSS:
                    meta_state[i * 3 + j] = 1
                elif winner == CellState.CIRCLE:
                    meta_state[i * 3 + j] = 2
        
        # Active board
        if self.game.active_board is not None:
            br, bc = self.game.active_board
            # Verificar si el tablero activo está disponible
            if self.game.boards[br][bc].winner is None:
                active = np.array([br, bc], dtype=np.int8)
            else:
                active = np.array([-1, -1], dtype=np.int8)
        else:
            active = np.array([-1, -1], dtype=np.int8)
        
        # Current player
        current = 0 if self.game.current_player == CellState.CROSS else 1
        
        return {
            "board": board_state,
            "meta_board": meta_state,
            "active_board": active,
            "current_player": current,
        }

    def _get_info(self) -> Dict[str, Any]:
        """Retorna información adicional."""
        return {
            "valid_moves": self.get_valid_moves(),
            "current_player": self.game.current_player.value,
            "winner": self.game.winner.value if self.game.winner else None,
        }

    def get_valid_moves(self) -> List[int]:
        """Retorna lista de acciones válidas."""
        moves = self.game.get_all_valid_moves()
        return [
            self.coords_to_action[(m.board_row, m.board_col, m.cell_row, m.cell_col)]
            for m in moves
        ]

    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[Dict] = None
    ) -> Tuple[Dict, Dict]:
        """Reinicia el entorno."""
        super().reset(seed=seed)
        self.game = UltimateTicTacToe()
        self._prev_small_wins = {CellState.CROSS: 0, CellState.CIRCLE: 0}
        
        return self._get_obs(), self._get_info()

    def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        Ejecuta una acción.
        
        Returns:
            observation, reward, terminated, truncated, info
        """
        valid_moves = self.get_valid_moves()
        
        # Acción inválida
        if action not in valid_moves:
            return (
                self._get_obs(),
                INVALID_MOVE_REWARD,
                False,
                False,
                self._get_info()
            )
        
        # Guardar estado previo para reward shaping
        prev_wins = self._count_small_wins()
        player_before_move = self.game.current_player
        
        # Ejecutar movimiento
        br, bc, cr, cc = self.action_to_coords[action]
        move = UltimateMove(br, bc, cr, cc, self.game.current_player)
        self.game.make_move(move)
        
        # Calcular recompensa
        reward = MOVE_REWARD
        terminated = False
        
        if self.game.winner is not None:
            terminated = True
            if self.game.winner == CellState.BLANK:
                reward = TIE_REWARD
            elif self.game.winner == player_before_move:
                # El jugador que acaba de mover ganó
                reward = WIN_REWARD
            else:
                reward = LOSS_REWARD
        else:
            # Reward shaping: bonus por ganar tableros pequeños
            current_wins = self._count_small_wins()
            if current_wins[player_before_move] > prev_wins[player_before_move]:
                reward += WIN_SMALL_BOARD
        
        # Si hay oponente y el juego no terminó, el oponente juega
        if not terminated and self.opponent == "random":
            reward += self._opponent_move()
            terminated = self.game.winner is not None
        
        return (
            self._get_obs(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _opponent_move(self) -> float:
        """Ejecuta movimiento del oponente aleatorio. Retorna recompensa adicional."""
        valid_moves = self.get_valid_moves()
        if not valid_moves:
            return 0.0
        
        action = self.np_random.choice(valid_moves)
        br, bc, cr, cc = self.action_to_coords[action]
        move = UltimateMove(br, bc, cr, cc, self.game.current_player)
        
        prev_wins = self._count_small_wins()
        player = self.game.current_player
        self.game.make_move(move)
        
        # Si el oponente ganó, es pérdida para el agente
        if self.game.winner == player:
            return LOSS_REWARD
        elif self.game.winner == CellState.BLANK:
            return TIE_REWARD
        
        # Penalización si oponente gana tablero pequeño
        current_wins = self._count_small_wins()
        if current_wins[player] > prev_wins[player]:
            return LOSE_SMALL_BOARD
        
        return 0.0

    def render(self) -> Optional[str]:
        """Renderiza el estado actual."""
        if self.render_mode == "ascii":
            output = self.game.display()
            print(output)
            print(f"Turno: {self.game.current_player.value}")
            print(f"Tablero activo: {self.game.active_board}")
            return output
        return None

    def close(self):
        """Cierra el entorno."""
        pass


# Función helper para preprocesar estado
def preprocess_state(obs: Dict) -> np.ndarray:
    """
    Convierte observación a vector plano para DQN.
    
    Input: Dict con board, meta_board, active_board, current_player
    Output: Array de tamaño 91 (81 + 9 + 1)
    """
    board = obs["board"].astype(np.float32)
    meta = obs["meta_board"].astype(np.float32)
    
    # Normalizar: 0, 0.5, 1 en lugar de 0, 1, 2
    board = board / 2.0
    meta = meta / 2.0
    
    # Active board: one-hot de 9 posiciones + 1 para "cualquiera"
    active_encoded = np.zeros(10, dtype=np.float32)
    if obs["active_board"][0] == -1:
        active_encoded[9] = 1  # Cualquier tablero válido
    else:
        idx = obs["active_board"][0] * 3 + obs["active_board"][1]
        active_encoded[idx] = 1
    
    # Current player
    player = np.array([obs["current_player"]], dtype=np.float32)
    
    return np.concatenate([board, meta, active_encoded, player])


# Estado size para el modelo
STATE_SIZE = 81 + 9 + 10 + 1  # = 101


if __name__ == "__main__":
    # Test del entorno
    env = UT3_Env(opponent="random")
    obs, info = env.reset()
    
    print("Observación inicial:")
    print(f"  Board shape: {obs['board'].shape}")
    print(f"  Meta board shape: {obs['meta_board'].shape}")
    print(f"  Active board: {obs['active_board']}")
    print(f"  Valid moves: {len(info['valid_moves'])} opciones")
    
    # Jugar algunos pasos aleatorios
    total_reward = 0
    for i in range(20):
        valid = info["valid_moves"]
        if not valid:
            break
        action = np.random.choice(valid)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated:
            print(f"\nJuego terminado en paso {i+1}")
            print(f"Ganador: {info['winner']}")
            break
    
    print(f"\nRecompensa total: {total_reward}")
    env.render()
