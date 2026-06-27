import sys

import chess
import typer
from typing import Optional, Annotated

from config.Setting import get_version_info
from core.utils import game_over
from play.gui import BoardWidget
from play.terminal import play_term
from train.data_pipeline import process_pgn
from train.train_xgboost import train_xgboost_model
from PySide6.QtWidgets import QApplication

app = typer.Typer(help="Khoai Chess AI - Engine danh co bang XGBoost", 
                  rich_markup_mode="rich",
                  pretty_exceptions_enable=False)

@app.command()
def train(
    player: Annotated[str, typer.Argument(help="Ten nguoi choi (VD: Fischer de tim Fischer.pgn)")],
    force: Annotated[bool, typer.Option("--force", "-f", help="Ghi de model cu neu da ton tai")] = False,
    new: Annotated[bool, typer.Option("--new", "-n", help="Tao phien ban model moi (v2, v3...)")] = False
):
    """
    Huan luyen mo hinh AI tu file PGN cua mot kien tuong.
    """

    if new:
        typer.secho(f"Ban dang tao phien ban moi cho mo hinh cua {player}.", fg=typer.colors.YELLOW)
        force = True  # Khi tạo phiên bản mới, luôn ghi đè mô hình cũ nếu có

    typer.secho(f"Dang doc du lieu cua {player}.pgn...", fg=typer.colors.CYAN)

    typer.secho(f"1. Sử lý dữ liệu từ PGN sang CSV...", fg=typer.colors.CYAN)
    process_pgn(player, force)
    typer.secho(f"2. Huấn luyen mo hinh...", fg=typer.colors.CYAN)
    train_xgboost_model(player, is_new=new)

@app.command()
def play(
    player: Annotated[str, typer.Argument(help="Ten mo hinh de thach dau (VD: Fischer)")],
    version: Annotated[str, typer.Argument(help="Chon phien ban model v1, v2, v3...")] = None,
    color: Annotated[str, typer.Option("--color", "-c", help="Chon quan: white hoac black")] = "white",
    ui: Annotated[str, typer.Option("--ui", "-u", help="Giao dien: term hoac gui")] = "term",
):
    """
    Bat dau tran dau voi mo hinh AI da duoc huan luyen.
    """

    if not version:
        version = get_version_info(player)["latest"]
        
    typer.secho(f"Khoi tao tran dau voi {player} (Phiên bản {version}). Ban cam quan {color}, UI: {ui}", fg=typer.colors.GREEN)
    model_name = f"{player}_{version}"
    print(model_name)
    board = chess.Board()
    if ui == "term":
        # Yêu cầu cập nhật tham số version bên trong hàm play_term
        play_term(board, color, model_name)
    elif ui == "gui":
        app_gui = QApplication(sys.argv)
        # Yêu cầu cập nhật tham số version bên trong class BoardWidget
        bw = BoardWidget(board, model_name, color)
        bw.show()
        app_gui.exec()
        
    game_over(board, color, model_name, version)

if __name__ == "__main__":
    app()