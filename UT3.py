"""
Ultimate TicTacToe - Versión Corregida
Correcciones:
- Eliminada sobrecarga de métodos (Python no la soporta)
- Corregido orden de cambio de turno en make_move
- Corregido is_move_valid para verificar celda vacía
- Corregido winner_announcement
"""
from enum import Enum
from typing import Optional, Tuple, List
from dataclasses import dataclass

COLS = 3
ROWS = 3


class CellState(Enum):
    BLANK = " "
    CROSS = "X"
    CIRCLE = "O"


@dataclass
class Move:
    row: int
    col: int
    player: CellState


@dataclass
class UltimateMove:
    board_row: int
    board_col: int
    cell_row: int
    cell_col: int
    player: CellState


class TicTacToe:
    def __init__(self):
        self.board: List[List[CellState]] = [
            [CellState.BLANK for _ in range(COLS)] for _ in range(ROWS)
        ]
        self.winner: Optional[CellState] = None

    def check_winner(self) -> Optional[CellState]:
        """Verifica si hay ganador o empate."""
        # Filas
        for i in range(ROWS):
            if self.board[i][0] == self.board[i][1] == self.board[i][2] != CellState.BLANK:
                self.winner = self.board[i][0]
                return self.winner

        # Columnas
        for j in range(COLS):
            if self.board[0][j] == self.board[1][j] == self.board[2][j] != CellState.BLANK:
                self.winner = self.board[0][j]
                return self.winner

        # Diagonales
        if self.board[0][0] == self.board[1][1] == self.board[2][2] != CellState.BLANK:
            self.winner = self.board[0][0]
            return self.winner
        if self.board[0][2] == self.board[1][1] == self.board[2][0] != CellState.BLANK:
            self.winner = self.board[0][2]
            return self.winner

        # Empate
        if all(self.board[i][j] != CellState.BLANK for i in range(3) for j in range(3)):
            self.winner = CellState.BLANK
            return self.winner

        return None

    def is_move_valid(self, row: int, col: int) -> bool:
        """Verifica si un movimiento es válido."""
        if self.winner is not None:
            return False
        if not (0 <= row < ROWS and 0 <= col < COLS):
            return False
        return self.board[row][col] == CellState.BLANK

    def make_move(self, row: int, col: int, player: CellState) -> bool:
        """Realiza un movimiento si es válido."""
        if not self.is_move_valid(row, col):
            return False
        self.board[row][col] = player
        self.check_winner()
        return True

    def display(self) -> str:
        """Retorna representación string del tablero."""
        lines = []
        for i, row in enumerate(self.board):
            lines.append(" │ ".join(cell.value for cell in row))
            if i < 2:
                lines.append("──┼───┼──")
        return "\n".join(lines)


class UltimateTicTacToe:
    def __init__(self):
        self.boards: List[List[TicTacToe]] = [
            [TicTacToe() for _ in range(COLS)] for _ in range(ROWS)
        ]
        self.meta_board = TicTacToe()
        self.active_board: Optional[Tuple[int, int]] = None
        self.winner: Optional[CellState] = None
        self.current_player: CellState = CellState.CROSS
        self.move_history: List[UltimateMove] = []

    def reset(self):
        """Reinicia el juego."""
        self.__init__()

    def get_valid_boards(self) -> List[Tuple[int, int]]:
        """Retorna lista de tableros donde se puede jugar."""
        if self.active_board is not None:
            br, bc = self.active_board
            # Si el tablero activo no está terminado, solo ese es válido
            if self.boards[br][bc].winner is None:
                return [self.active_board]
        
        # Si no hay tablero activo o el activo está terminado,
        # cualquier tablero no terminado es válido
        valid = []
        for i in range(3):
            for j in range(3):
                if self.boards[i][j].winner is None:
                    valid.append((i, j))
        return valid

    def is_move_valid(self, move: UltimateMove) -> bool:
        """Verifica si el movimiento es válido."""
        # Juego terminado
        if self.winner is not None:
            return False

        # Turno incorrecto
        if self.current_player != move.player:
            return False

        # Coordenadas fuera de rango
        if not (0 <= move.board_row < 3 and 0 <= move.board_col < 3):
            return False
        if not (0 <= move.cell_row < 3 and 0 <= move.cell_col < 3):
            return False

        # Verificar si el tablero elegido es válido
        valid_boards = self.get_valid_boards()
        if (move.board_row, move.board_col) not in valid_boards:
            return False

        # Verificar si la celda está vacía
        small_board = self.boards[move.board_row][move.board_col]
        return small_board.board[move.cell_row][move.cell_col] == CellState.BLANK

    def make_move(self, move: UltimateMove) -> bool:
        """Realiza un movimiento si es válido."""
        if not self.is_move_valid(move):
            return False

        # Realizar movimiento en el tablero pequeño
        small_board = self.boards[move.board_row][move.board_col]
        small_board.make_move(move.cell_row, move.cell_col, move.player)
        
        # Guardar en historial
        self.move_history.append(move)

        # Actualizar tablero activo basado en celda jugada
        self.active_board = (move.cell_row, move.cell_col)

        # Si el tablero pequeño tiene ganador, actualizar meta-tablero
        if small_board.winner in [CellState.CROSS, CellState.CIRCLE]:
            self.meta_board.board[move.board_row][move.board_col] = small_board.winner
            self.meta_board.check_winner()
            self.winner = self.meta_board.winner

        # Verificar empate global (todos los tableros terminados sin ganador)
        if self.winner is None:
            all_finished = all(
                self.boards[i][j].winner is not None
                for i in range(3) for j in range(3)
            )
            if all_finished:
                self.winner = CellState.BLANK

        # Cambiar turno DESPUÉS de que todo fue exitoso
        self.current_player = (
            CellState.CIRCLE if self.current_player == CellState.CROSS 
            else CellState.CROSS
        )
        
        return True

    def get_all_valid_moves(self) -> List[UltimateMove]:
        """Retorna todos los movimientos válidos posibles."""
        moves = []
        valid_boards = self.get_valid_boards()
        
        for br, bc in valid_boards:
            small_board = self.boards[br][bc]
            for cr in range(3):
                for cc in range(3):
                    if small_board.board[cr][cc] == CellState.BLANK:
                        moves.append(UltimateMove(
                            br, bc, cr, cc, self.current_player
                        ))
        return moves

    def display(self) -> str:
        """Retorna representación string del tablero completo."""
        lines = []
        
        for meta_row in range(3):
            for row_offset in range(3):
                row_parts = []
                for meta_col in range(3):
                    small_board = self.boards[meta_row][meta_col]
                    small_row = small_board.board[row_offset]
                    
                    # Marcar tablero activo
                    if self.active_board == (meta_row, meta_col) and small_board.winner is None:
                        cells = [f"[{cell.value}]" if cell == CellState.BLANK else f" {cell.value} " 
                                for cell in small_row]
                    else:
                        cells = [f" {cell.value} " for cell in small_row]
                    
                    row_parts.append("".join(cells))
                
                lines.append(" ║ ".join(row_parts))
            
            if meta_row < 2:
                lines.append("═" * 11 + "╬" + "═" * 11 + "╬" + "═" * 11)
        
        return "\n".join(lines)

    def winner_announcement(self) -> str:
        """Retorna mensaje del resultado."""
        if self.winner is None:
            return "Juego en progreso..."
        elif self.winner == CellState.BLANK:
            return "¡Empate!"
        else:
            return f"¡Ganador: {self.winner.value}!"


if __name__ == "__main__":
    game = UltimateTicTacToe()
    
    # Demo de movimientos
    test_moves = [
        UltimateMove(1, 1, 1, 1, CellState.CROSS),
        UltimateMove(1, 1, 0, 0, CellState.CIRCLE),
        UltimateMove(0, 0, 1, 1, CellState.CROSS),
    ]
    
    for move in test_moves:
        if game.make_move(move):
            print(game.display())
            print(f"Turno de: {game.current_player.value}")
            print(f"Tablero activo: {game.active_board}")
            print("-" * 40)
        else:
            print(f"Movimiento inválido: {move}")
