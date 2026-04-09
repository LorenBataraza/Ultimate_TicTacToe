"""
Interfaz de Terminal para Ultimate TicTacToe usando Textual
"""
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical, Grid
from textual.widgets import (
    Static, Button, Header, Footer, Label, 
    Select, Switch, ProgressBar, RichLog
)
from textual.binding import Binding
from textual.message import Message
from textual import events
from textual.screen import Screen
from rich.text import Text
from rich.panel import Panel
from rich.table import Table
import numpy as np
from typing import Optional, Tuple
import os

from UT3 import UltimateTicTacToe, UltimateMove, CellState


# Símbolos para renderizado
SYMBOLS = {
    CellState.BLANK: " ",
    CellState.CROSS: "X",
    CellState.CIRCLE: "O",
}

COLORS = {
    CellState.BLANK: "white",
    CellState.CROSS: "red",
    CellState.CIRCLE: "blue",
}


class Cell(Button):
    """Celda clickeable del tablero."""
    
    DEFAULT_CSS = """
    Cell {
        width: 3;
        height: 1;
        min-width: 3;
        min-height: 1;
        padding: 0;
        margin: 0;
        border: none;
        background: $surface;
    }
    
    Cell:hover {
        background: $accent;
    }
    
    Cell.valid {
        background: $success-darken-2;
    }
    
    Cell.valid:hover {
        background: $success;
    }
    
    Cell.cross {
        color: red;
    }
    
    Cell.circle {
        color: blue;
    }
    
    Cell.won-cross {
        background: $error-darken-3;
    }
    
    Cell.won-circle {
        background: $primary-darken-3;
    }
    """
    
    def __init__(
        self, 
        board_row: int, 
        board_col: int, 
        cell_row: int, 
        cell_col: int,
        **kwargs
    ):
        super().__init__(" ", **kwargs)
        self.board_row = board_row
        self.board_col = board_col
        self.cell_row = cell_row
        self.cell_col = cell_col
    
    def update_state(self, state: CellState, valid: bool, board_winner: Optional[CellState]):
        """Actualiza el estado visual de la celda."""
        self.label = SYMBOLS[state]
        
        # Limpiar clases
        self.remove_class("cross", "circle", "valid", "won-cross", "won-circle")
        
        if state == CellState.CROSS:
            self.add_class("cross")
        elif state == CellState.CIRCLE:
            self.add_class("circle")
        
        if board_winner == CellState.CROSS:
            self.add_class("won-cross")
        elif board_winner == CellState.CIRCLE:
            self.add_class("won-circle")
        elif valid and state == CellState.BLANK:
            self.add_class("valid")


class SmallBoard(Container):
    """Un tablero pequeño de 3x3."""
    
    DEFAULT_CSS = """
    SmallBoard {
        layout: grid;
        grid-size: 3 3;
        grid-gutter: 0;
        padding: 0;
        margin: 0;
        width: 9;
        height: 3;
        border: solid $primary;
    }
    
    SmallBoard.active {
        border: heavy $success;
    }
    
    SmallBoard.won-cross {
        border: heavy red;
    }
    
    SmallBoard.won-circle {
        border: heavy blue;
    }
    """
    
    def __init__(self, board_row: int, board_col: int, **kwargs):
        super().__init__(**kwargs)
        self.board_row = board_row
        self.board_col = board_col
        self.cells: list[list[Cell]] = []
    
    def compose(self) -> ComposeResult:
        for cr in range(3):
            row = []
            for cc in range(3):
                cell = Cell(
                    self.board_row, self.board_col, cr, cc,
                    id=f"cell_{self.board_row}_{self.board_col}_{cr}_{cc}"
                )
                row.append(cell)
                yield cell
            self.cells.append(row)
    
    def update_state(
        self, 
        board: "TicTacToe", 
        is_active: bool, 
        valid_cells: list[Tuple[int, int]]
    ):
        """Actualiza el estado visual del tablero."""
        self.remove_class("active", "won-cross", "won-circle")
        
        if board.winner == CellState.CROSS:
            self.add_class("won-cross")
        elif board.winner == CellState.CIRCLE:
            self.add_class("won-circle")
        elif is_active:
            self.add_class("active")
        
        for cr in range(3):
            for cc in range(3):
                cell_valid = (cr, cc) in valid_cells
                self.cells[cr][cc].update_state(
                    board.board[cr][cc],
                    cell_valid,
                    board.winner
                )


class GameBoard(Container):
    """Tablero completo de Ultimate TicTacToe."""
    
    DEFAULT_CSS = """
    GameBoard {
        layout: grid;
        grid-size: 3 3;
        grid-gutter: 1;
        padding: 1;
        width: 35;
        height: 14;
        border: double $primary;
    }
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.small_boards: list[list[SmallBoard]] = []
    
    def compose(self) -> ComposeResult:
        for br in range(3):
            row = []
            for bc in range(3):
                board = SmallBoard(br, bc, id=f"board_{br}_{bc}")
                row.append(board)
                yield board
            self.small_boards.append(row)
    
    def update_from_game(self, game: UltimateTicTacToe):
        """Actualiza todo el tablero desde el estado del juego."""
        valid_boards = game.get_valid_boards()
        
        for br in range(3):
            for bc in range(3):
                is_active = (br, bc) in valid_boards
                
                # Obtener celdas válidas para este tablero
                valid_cells = []
                if is_active and game.boards[br][bc].winner is None:
                    for cr in range(3):
                        for cc in range(3):
                            if game.boards[br][bc].board[cr][cc] == CellState.BLANK:
                                valid_cells.append((cr, cc))
                
                self.small_boards[br][bc].update_state(
                    game.boards[br][bc],
                    is_active,
                    valid_cells
                )


class StatusPanel(Static):
    """Panel de estado del juego."""
    
    DEFAULT_CSS = """
    StatusPanel {
        width: 100%;
        height: auto;
        padding: 1;
        border: round $primary;
        margin: 1;
    }
    """
    
    def update_status(
        self, 
        current_player: CellState,
        winner: Optional[CellState],
        move_count: int,
        mode: str
    ):
        """Actualiza el panel de estado."""
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Key", style="bold")
        table.add_column("Value")
        
        # Estado del juego
        if winner is not None:
            if winner == CellState.BLANK:
                status = "[yellow]EMPATE[/yellow]"
            else:
                color = "red" if winner == CellState.CROSS else "blue"
                status = f"[{color}]¡GANADOR: {winner.value}![/{color}]"
        else:
            color = "red" if current_player == CellState.CROSS else "blue"
            status = f"[{color}]Turno de: {current_player.value}[/{color}]"
        
        table.add_row("Estado", status)
        table.add_row("Movimientos", str(move_count))
        table.add_row("Modo", mode)
        
        self.update(table)


class UT3App(App):
    """Aplicación principal de Ultimate TicTacToe."""
    
    CSS = """
    Screen {
        layout: horizontal;
    }
    
    #left-panel {
        width: 40;
        height: 100%;
    }
    
    #right-panel {
        width: 1fr;
        height: 100%;
        padding: 1;
    }
    
    #game-log {
        height: 1fr;
        border: round $primary;
        margin-top: 1;
    }
    
    #controls {
        height: auto;
        padding: 1;
        border: round $primary;
    }
    
    .control-row {
        height: auto;
        margin: 1 0;
    }
    """
    
    BINDINGS = [
        Binding("n", "new_game", "Nueva Partida"),
        Binding("u", "undo", "Deshacer"),
        Binding("q", "quit", "Salir"),
        Binding("a", "toggle_ai", "Toggle IA"),
    ]
    
    def __init__(self):
        super().__init__()
        self.game = UltimateTicTacToe()
        self.game_mode = "pvp"  # pvp, pve, eve
        self.ai_player: Optional[CellState] = None
        self.agent = None
    
    def compose(self) -> ComposeResult:
        yield Header()
        
        with Horizontal():
            with Vertical(id="left-panel"):
                yield GameBoard(id="game-board")
                yield StatusPanel(id="status")
            
            with Vertical(id="right-panel"):
                with Container(id="controls"):
                    yield Label("Controles", classes="control-title")
                    
                    with Horizontal(classes="control-row"):
                        yield Button("Nueva Partida", id="btn-new", variant="primary")
                        yield Button("Deshacer", id="btn-undo", variant="warning")
                    
                    with Horizontal(classes="control-row"):
                        yield Label("Modo:")
                        yield Select(
                            [
                                ("Jugador vs Jugador", "pvp"),
                                ("Jugador vs IA", "pve"),
                                ("IA vs IA", "eve"),
                            ],
                            value="pvp",
                            id="mode-select"
                        )
                
                yield RichLog(id="game-log", highlight=True, markup=True)
        
        yield Footer()
    
    def on_mount(self):
        """Inicialización al montar la app."""
        self.update_display()
        self.log_message("¡Bienvenido a Ultimate TicTacToe!")
        self.log_message("Haz clic en una celda verde para jugar.")
    
    def update_display(self):
        """Actualiza toda la interfaz."""
        board = self.query_one("#game-board", GameBoard)
        board.update_from_game(self.game)
        
        status = self.query_one("#status", StatusPanel)
        status.update_status(
            self.game.current_player,
            self.game.winner,
            len(self.game.move_history),
            self.game_mode.upper()
        )
    
    def log_message(self, message: str):
        """Añade mensaje al log."""
        log = self.query_one("#game-log", RichLog)
        log.write(message)
    
    def on_button_pressed(self, event: Button.Pressed):
        """Maneja clicks en botones."""
        button_id = event.button.id
        
        if button_id == "btn-new":
            self.action_new_game()
        elif button_id == "btn-undo":
            self.action_undo()
        elif button_id and button_id.startswith("cell_"):
            self.handle_cell_click(event.button)
    
    def handle_cell_click(self, cell: Cell):
        """Procesa click en una celda."""
        if self.game.winner is not None:
            self.log_message("[yellow]El juego ha terminado. Inicia una nueva partida.[/yellow]")
            return
        
        move = UltimateMove(
            cell.board_row,
            cell.board_col,
            cell.cell_row,
            cell.cell_col,
            self.game.current_player
        )
        
        if self.game.make_move(move):
            player = "X" if move.player == CellState.CROSS else "O"
            self.log_message(
                f"[{'red' if move.player == CellState.CROSS else 'blue'}]"
                f"{player}[/] → Tablero ({move.board_row},{move.board_col}), "
                f"Celda ({move.cell_row},{move.cell_col})"
            )
            
            self.update_display()
            
            if self.game.winner is not None:
                self.announce_winner()
            elif self.game_mode == "pve" and self.game.current_player == CellState.CIRCLE:
                self.ai_move()
        else:
            self.log_message("[red]Movimiento inválido[/red]")
    
    def ai_move(self):
        """Ejecuta movimiento de la IA."""
        if self.agent is None:
            # IA aleatoria si no hay modelo
            moves = self.game.get_all_valid_moves()
            if moves:
                move = np.random.choice(moves)
                self.game.make_move(move)
                self.log_message(
                    f"[blue]IA[/blue] → Tablero ({move.board_row},{move.board_col}), "
                    f"Celda ({move.cell_row},{move.cell_col})"
                )
                self.update_display()
                
                if self.game.winner is not None:
                    self.announce_winner()
    
    def announce_winner(self):
        """Anuncia el ganador."""
        if self.game.winner == CellState.BLANK:
            self.log_message("[yellow bold]¡EMPATE![/yellow bold]")
        else:
            color = "red" if self.game.winner == CellState.CROSS else "blue"
            self.log_message(f"[{color} bold]¡{self.game.winner.value} GANA![/{color} bold]")
    
    def on_select_changed(self, event: Select.Changed):
        """Maneja cambio de modo de juego."""
        if event.select.id == "mode-select":
            self.game_mode = event.value
            self.log_message(f"Modo cambiado a: {self.game_mode.upper()}")
            self.action_new_game()
    
    def action_new_game(self):
        """Inicia nueva partida."""
        self.game = UltimateTicTacToe()
        self.update_display()
        self.log_message("[green]Nueva partida iniciada[/green]")
    
    def action_undo(self):
        """Deshace el último movimiento."""
        if self.game.move_history:
            # Recrear el juego sin el último movimiento
            history = self.game.move_history[:-1]
            self.game = UltimateTicTacToe()
            for move in history:
                # Ajustar jugador según turno original
                self.game.make_move(move)
            
            self.update_display()
            self.log_message("[yellow]Movimiento deshecho[/yellow]")
        else:
            self.log_message("[red]No hay movimientos para deshacer[/red]")
    
    def action_toggle_ai(self):
        """Alterna modo IA."""
        modes = ["pvp", "pve", "eve"]
        current_idx = modes.index(self.game_mode)
        self.game_mode = modes[(current_idx + 1) % len(modes)]
        
        select = self.query_one("#mode-select", Select)
        select.value = self.game_mode
        
        self.log_message(f"Modo: {self.game_mode.upper()}")


def main():
    """Punto de entrada."""
    app = UT3App()
    app.run()


if __name__ == "__main__":
    main()
