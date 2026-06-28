import chess.pgn
import numpy as np
import pandas as pd
import sys
import random
from config.Setting import get_processed_csv_path, get_raw_pgn_path
from core.utils import extract_features 

def process_pgn(player, force=False):
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
            
            # Lấy tên người chơi Trắng và Đen từ Header của ván cờ
            white_player = game.headers.get("White", "")
            black_player = game.headers.get("Black", "")
            
            # Kiểm tra xem tham số 'player' (VD: "Fischer") có nằm trong tên người chơi không
            is_player_white = player.lower() in white_player.lower()
            is_player_black = player.lower() in black_player.lower()
            
            # Bỏ qua ván đấu nếu file PGN lẫn tạp ván cờ không có tên Kiện tướng
            if not is_player_white and not is_player_black:
                continue
                
            board = game.board()
            for move in game.mainline_moves():
                
                # CHỈ CHỌN LỌC: Kiểm tra xem lượt hiện tại có phải của Kiện tướng không
                # board.turn = True (Trắng), False (Đen)
                is_player_turn = (board.turn == chess.WHITE and is_player_white) or \
                                 (board.turn == chess.BLACK and is_player_black)
                
                # Nếu ĐÚNG là lượt của Kiện tướng thì mới cho AI học
                if is_player_turn:
                    # 1. POSITIVE SAMPLE: Nước đi Kiện tướng chọn (Nhãn 1)
                    board.push(move) 
                    features_played = extract_features(board).tolist()
                    features_played.append(1) 
                    data.append(features_played)
                    board.pop() 
                    
                    # 2. NEGATIVE SAMPLES: Nước đi Kiện tướng bỏ qua (Nhãn 0)
                    legal_moves = list(board.legal_moves)
                    legal_moves.remove(move)
                    
                    num_negatives = min(3, len(legal_moves))
                    if num_negatives > 0:
                        negative_moves = random.sample(legal_moves, num_negatives)
                        for neg_move in negative_moves:
                            board.push(neg_move)
                            features_not_played = extract_features(board).tolist()
                            features_not_played.append(0) 
                            data.append(features_not_played)
                            board.pop()
                
                # Quan trọng: Vẫn phải đánh nước cờ đó lên bàn cờ để hệ thống tiến tới lượt tiếp theo
                # (Dù không học nước của đối thủ, bàn cờ vẫn phải thay đổi trạng thái)
                board.push(move)
                
                if is_player_turn:
                    move_count += 1
                    if move_count % 1000 == 0:
                        print(f"Da xu ly {move_count} nuoc di cua {player}\r", end="")

    print("\nDang luu ra file CSV...")
    
    # Header chuẩn 73 cột (tránh lỗi Pandas DataFrame)
    cols = [f"sq_{i}" for i in range(64)]
    cols.extend([
        "mat_diff", 
        "mobility", 
        "center_ctrl", 
        "is_check", 
        "king_safety",
        "pst_diff",
        "rook_open_file",
        "king_attack"
    ])
    cols.append("target")
    
    df = pd.DataFrame(data, columns=cols)
    df.to_csv(output_csv, index=False)
    print("So dong:", len(df))