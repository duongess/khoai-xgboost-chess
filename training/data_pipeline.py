import chess.pgn
import numpy as np
import pandas as pd
import sys
from config.Setting import get_processed_csv_path, get_raw_pgn_path
from core.utils import board_to_matrix

def process_pgn(player, force=False):
    # Trich xuat du lieu tu file PGN
    data = []
    move_count = 0

    input_file = get_raw_pgn_path(player)
    output_csv = get_processed_csv_path(player)

    if (output_csv.exists()):
        if not force:
            print(f"File CSV da ton tai: {output_csv}. Neu muon ghi de, su dung tham so --force")
            sys.exit(1)
        else:
            print(f"Ghi de file CSV: {output_csv}")

    with open(input_file, "r") as pgn:
        while True:
            game = chess.pgn.read_game(pgn)
            if game is None:
                break
                
            board = game.board()
            for move in game.mainline_moves():
                features = board_to_matrix(board)
                label = board.san(move)
                
                row = features.tolist()
                row.append(label)
                data.append(row)
                
                board.push(move)
                move_count += 1
                
                if move_count % 1000 == 0:
                    print(f"Da xu ly {move_count} nuoc di\r", end="")

    print("\nDang luu ra file CSV...")
    cols = [f"sq_{i}" for i in range(64)]
    cols.append("target")
    
    df = pd.DataFrame(data, columns=cols)
    df.to_csv(output_csv, index=False)
    print("So dong:", len(df))