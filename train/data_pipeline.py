import chess.pgn
import numpy as np
import pandas as pd
import sys
import random
from config.Setting import get_processed_csv_path, get_raw_pgn_path

from core.utils import extract_features 

def process_pgn(player, force=False):
    # Trich xuat du lieu tu file PGN
    data = []
    move_count = 0

    input_file = get_raw_pgn_path(player)
    output_csv = get_processed_csv_path(player)

    if output_csv.exists():
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
                
                # 1. POSITIVE SAMPLE: Nước đi Kiện tướng chọn (Nhãn 1)
                board.push(move) # Tiến 1 bước để lấy bối cảnh SAU khi đi
                features_played = extract_features(board).tolist()
                features_played.append(1) # Label = 1 (Kiện tướng thích)
                data.append(features_played)
                board.pop() # Hoàn tác để thử các nước khác
                
                # 2. NEGATIVE SAMPLES: Nước đi Kiện tướng bỏ qua (Nhãn 0)
                legal_moves = list(board.legal_moves)
                legal_moves.remove(move)
                
                # Chọn ngẫu nhiên tối đa 3 nước đi không được chọn
                num_negatives = min(3, len(legal_moves))
                if num_negatives > 0:
                    negative_moves = random.sample(legal_moves, num_negatives)
                    for neg_move in negative_moves:
                        board.push(neg_move)
                        features_not_played = extract_features(board).tolist()
                        features_not_played.append(0) # Label = 0 (Kiện tướng chê)
                        data.append(features_not_played)
                        board.pop()
                
                # 3. Tiến cờ lên nước thật để xét lượt tiếp theo
                board.push(move)
                move_count += 1
                
                if move_count % 1000 == 0:
                    print(f"Da xu ly {move_count} nuoc di\r", end="")

    print("\nDang luu ra file CSV...")
    
    cols = [f"sq_{i}" for i in range(64)]
    cols.extend(["mat_diff", "mobility", "center_ctrl", "is_check", "king_safety"])
    cols.append("target")
    
    df = pd.DataFrame(data, columns=cols)
    df.to_csv(output_csv, index=False)
    print("So dong:", len(df))