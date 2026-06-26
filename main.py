import sys

import chess
import typer
from typing import Optional, Annotated

from play.gui import BoardWidget
from play.terminal import play_term
from training.data_pipeline import process_pgn
from training.train_xgboost import train_fischer_model
from PySide6.QtWidgets import QApplication

app = typer.Typer(help="Khoai Chess AI - Engine danh co bang XGBoost", 
                  rich_markup_mode="rich",
                  pretty_exceptions_enable=False)

@app.command()
def train(
    player: Annotated[str, typer.Argument(help="Ten nguoi choi (VD: Fischer de tim Fischer.pgn)")],
    force: Annotated[bool, typer.Option("--force", "-f", help="Ghi de model cu neu da ton tai")] = False
):
    """
    Huan luyen mo hinh AI tu file PGN cua mot kien tuong.
    """
    typer.secho(f"Dang doc du lieu cua {player}.pgn...", fg=typer.colors.CYAN)

    typer.secho(f"1. Sử lý dữ liệu từ PGN sang CSV...", fg=typer.colors.CYAN)
    process_pgn(player, force)
    typer.secho(f"2. Huấn luyen mo hinh...", fg=typer.colors.CYAN)
    train_fischer_model(player)

@app.command()
def play(
    model_name: Annotated[str, typer.Argument(help="Ten mo hinh de thach dau (VD: Fischer)")],
    color: Annotated[str, typer.Option("--color", "-c", help="Chon quan: white hoac black")] = "white",
    ui: Annotated[str, typer.Option("--ui", "-u", help="Giao dien: term hoac gui")] = "term"
):
    """
    Bat dau tran dau voi mo hinh AI da duoc huan luyen.
    """
    typer.secho(f"Khoi tao tran dau voi {model_name}. Ban cam quan {color}, UI: {ui}", fg=typer.colors.GREEN)
    
    board = chess.Board()
    if (ui == "term"):
        play_term(board, color, model_name)
    elif (ui == "gui"):
        app_gui = QApplication(sys.argv)
        bw = BoardWidget(board, model_name)
        bw.show()
        sys.exit(app_gui.exec())
        
if __name__ == "__main__":
    app()