from enum import Enum
from typing import Optional, Tuple

# Other CONSTANTS
COLS=3
ROWS=3
SQRS_PER_GAME = 9 
TEAMS=2
CELL_STATEs = 3

class CellState(Enum):
    BLANK = " " 
    CROSS = "X"
    CIRCLE = "O"

class Move:
    def __init__(self, row: int, col: int, player: CellState):
        self.row = row
        self.col = col
        self.player = player

class TicTacToe:
    def __init__(self):
        # Inicializar un tablero 3x3 con celdas vacías
        self.board = [[CellState.BLANK for _ in range(COLS)] for _ in range(ROWS)]
        
        # None= no determinado
        # Blank = Empate 
        self.winner: Optional[CellState] = None

    def check_winner(self):
        # Verificar filas
        for i in range(ROWS):
            if self.board[i][0] == self.board[i][1] == self.board[i][2] != CellState.BLANK:
                self.winner = self.board[i][0]
                return

        # Verificar columnas
        for j in range(COLS):
            if self.board[0][j] == self.board[1][j] == self.board[2][j] != CellState.BLANK:
                self.winner = self.board[0][j]
                return

        # Verificar diagonales
        if self.board[0][0] == self.board[1][1] == self.board[2][2] != CellState.BLANK:
            self.winner = self.board[0][0]
            return
        if self.board[0][2] == self.board[1][1] == self.board[2][0] != CellState.BLANK:
            self.winner = self.board[0][2]
            return

        # Verificar empate
        if all(self.board[i][j] != CellState.BLANK for i in range(3) for j in range(3)):
            self.winner = CellState.BLANK  # Empate

    def display(self) -> None:
            """Muestra el tablero en consola."""
            for row in self.board:
                print("|".join(cell.value for cell in row))
                print("-" * 5)
            print("\n")


    def is_move_valid(self, row: int, col: int, player: CellState) -> bool:
        """Chequea si el moviento es válido"""
        if self.winner is not None:
            return False  # Juego terminado
        if 0 <= row < ROWS and 0 <= col < COLS and self.board[row][col] == CellState.BLANK:
            return True
        
    def is_move_valid(self, move: Move) -> bool:
        """Chequea si el moviento es válido"""
        return 0 <= move.row < ROWS and 0 <= move.col < COLS and self.board[move.row][move.col] == CellState.BLANK and not self.winner

    def make_move(self, row: int, col: int, player: CellState) -> bool:
        """Realiza una jugada en la posición (row, col) si es válida."""
        if self.is_move_valid(row, col, player):
            self.board[row][col] = player
            self.check_winner()
            return True
        return False

    def make_move(self, move: Move) -> bool:
        """Realiza una jugada en la posición (row, col) si es válida."""
        if self.is_move_valid(move):
            self.board[move.row][move.col] = move.player
            self.check_winner()
            return True
        return False
    
    def winner_annoucement(self):
        if self.game.winner:
            print("The winner is", self.game.winner.value)
        else:
            print("There is no winner yet!.")


class UltimateMove:
    def __init__(self, board_row: int, board_col: int, cell_row: int, cell_col: int, player: CellState):
        self.board_row = board_row
        self.board_col = board_col
        self.cell_row = cell_row
        self.cell_col = cell_col
        self.player = player

class UltimateTicTacToe:
    def __init__(self):
        # Cuadrícula de 3x3 de tableros pequeños
        self.boards = [[TicTacToe() for _ in range(COLS)] for _ in range(ROWS)]
        # Meta-tablero para rastrear ganadores
        self.meta_board = TicTacToe()
        # Tablero activo: None significa cualquier tablero es válido
        self.active_board = None
        # Ganador total
        self.winner = None
        # Mantengo esta variable para no imprimir el tablero si la jugada era mala.
        self.valid_last_move=True
        # Turn - Empieza Cruz
        self.current_player=CellState.CROSS;

    # Reset works as initiation
    reset = __init__


    def is_move_valid(self, move: UltimateMove) -> bool:
        """Chequea si el moviento es válido"""
        if self.winner is not None:
            return False  # Juego terminado

        if self.current_player is not move.player:
            return False
        
        # Verificar si el tablero elegido es válido
        if self.active_board is None or self.boards[self.active_board[0]][self.active_board[1]].winner is not None:
            # Cualquier tablero es permitido
            return True
        else:
            # Debe jugar en el tablero activo
            if (move.board_row, move.board_col) != self.active_board:
                self.valid_last_move= False
                print("Wrong movement!!\n")
                return False
        return True

    def make_move(self, move: UltimateMove) -> bool:
        """Realiza un movimiento si es válido."""
        if self.is_move_valid(move):
            # Cambio turno
            self.current_player = CellState.CROSS if self.current_player == CellState.CIRCLE else CellState.CIRCLE
            # Realizar el movimiento en el tablero pequeño
            small_board = self.boards[move.board_row][move.board_col]
            small_move = Move(move.cell_row, move.cell_col, move.player)
            if small_board.make_move(small_move):
                # Movimiento exitoso, actualizar tablero activo
                self.active_board = (move.cell_row, move.cell_col)
                self.valid_last_move= True

                # Verificar si el tablero pequeño tiene ganador
                small_winner = small_board.winner
                if small_winner in [CellState.CROSS, CellState.CIRCLE]:
                    self.meta_board.board[move.board_row][move.board_col] = small_winner

                # Verificar ganador en el meta-tablero
                self.check_meta_winner()
                return True
            return False

    def check_meta_winner(self):
            """Verifica si hay un ganador en el meta-tablero."""
            self.meta_board.check_winner()
            self.winner = self.meta_board.winner
            return self.winner 
    
    def display(self) -> None:
        """Muestra el tablero completo de Ultimate Tic-Tac-Toe en consola."""
        if not self.valid_last_move:
            return 
        
        for meta_row in range(3):
            for row_offset in range(3):
                line = ''
                for meta_col in range(3):
                    small_board = self.boards[meta_row][meta_col]
                    small_row = small_board.board[row_offset]
                    small_line = ' '.join(cell.value for cell in small_row)
                    if meta_col > 0:
                        line += ' | '
                    line += small_line
                print(line)
            if meta_row < 2:
                print('-' * 21)
        print("\n")

    def winner_annoucement(self):
        if self.game.winner:
            print("The winner is", self.game.winner.value)
        else:
            print("There is no winner yet!.")


# Use Example
if __name__ == "__main__":
    
    example = input("What example do you want \n - Simple TicTacToe (1) \n - Ultimate TicTacToe (2)\n")
    
    if example==1:
        game = TicTacToe()
        moves= [
                Move(1,1, CellState.CROSS),
                Move(0,0, CellState.CIRCLE),
                Move(1,0, CellState.CROSS),
                Move(0,1, CellState.CIRCLE),
                Move(1,2, CellState.CROSS)
                ]
        
        for move in moves:
            #print(move) 
            game.make_move(move)
            game.display()

        print("The winner is ",game.winner.value)
    else:
        game = UltimateTicTacToe()
        moves = [
            UltimateMove(0, 0, 0, 0, CellState.CROSS),  # X en tablero (0,0), celda (0,0)
            UltimateMove(0, 0, 1, 1, CellState.CIRCLE), # O en tablero (0,0), celda (1,1)
            UltimateMove(1, 1, 0, 0, CellState.CROSS),  # X en tablero (1,1), celda (0,0)
            UltimateMove(1, 1, 1, 1, CellState.CIRCLE), # O en tablero (1,1), celda (1,1)
            UltimateMove(0, 0, 2, 2, CellState.CROSS),  # X en tablero (0,0), celda (2,2)
        ]

        for move in moves:
            game.make_move(move)
            game.display()
        
        if input("¿Continuar juego? sí[1]/no[0]") == "1":
            print("Formato de entrada: ultimate_row ultimate_col row col team(0=X, 1=O)")
            
            while game.winner is None:
                inputs = input().split()
                
                # Check good input 
                if len(inputs)!=5:
                    print("Invalid input, please retry\n")
                    continue
            
                ## Parse String
                ultimate_row = int(inputs[0])
                ultimate_col = int(inputs[1])
                row = int(inputs[2])
                col = int(inputs[3])
                team = CellState.CROSS if int(inputs[4]) == 0 else CellState.CIRCLE
                move = UltimateMove(ultimate_row, ultimate_col, row, col, team)
                game.make_move(move)
                game.display()
                print(game.active_board)
            
            game.winner_annoucement()


        

