import random
import chess.pgn
import numpy as np
import pandas as pd
import sys
from config.Setting import get_processed_parquet_path, get_raw_pgn_path, BASE_DIR, PROCESSED_BASE_DIR
from core.utils import extract_features     

def process_game_base(game, white_score, black_score):
    # Giữ nguyên logic xử lý base
    data = []
    board = game.board()
    moves = list(game.mainline_moves())
    total_moves = len(moves)
    
    for ply, move in enumerate(moves):
        player_who_moved = board.turn
        game_outcome = white_score if player_who_moved == chess.WHITE else black_score
        progress = (ply + 1) / total_moves
        
        if game_outcome > 0:
            target_score = 0.5 + (0.5 * progress)
        elif game_outcome < 0:
            target_score = -1.0 * (progress ** 2)
        else:
            target_score = 0.0
            
        board.push(move)
        features = extract_features(board).tolist()
        features.append(0)
        features.append(target_score)
        data.append(features)
        board.pop()
        
        legal_moves = list(board.legal_moves)
        legal_moves.remove(move)
        num_negatives = min(3, len(legal_moves))
        
        if num_negatives > 0:
            negative_moves = random.sample(legal_moves, num_negatives)
            for neg_move in negative_moves:
                board.push(neg_move)
                features_neg = extract_features(board).tolist()
                features_neg.append(0)
                features_neg.append(-1.0) 
                data.append(features_neg)
                board.pop()
                
        board.push(move)
        
    return data

def process_game_style(game, player_name, white_score, black_score):
    # Giữ nguyên logic xử lý style
    data = []
    white_player = game.headers.get("White", "")
    black_player = game.headers.get("Black", "")
    is_player_white = player_name.lower() in white_player.lower()
    is_player_black = player_name.lower() in black_player.lower()
    
    if not is_player_white and not is_player_black:
        return data
        
    board = game.board()
    moves = list(game.mainline_moves())
    total_moves = len(moves)
    
    for ply, move in enumerate(moves):
        is_player_turn = (board.turn == chess.WHITE and is_player_white) or \
                         (board.turn == chess.BLACK and is_player_black)
                         
        board.push(move)
        
        if is_player_turn:
            game_outcome = white_score if is_player_white else black_score
            decay = (ply + 1) / total_moves
            target_score = game_outcome * decay
            
            features = extract_features(board).tolist()
            features.append(1) 
            features.append(target_score)
            data.append(features)
            
    return data

def count_games_in_pgn(file_path):
    # Đếm nhanh tổng số ván cờ bằng cách đọc header để tránh tốn RAM
    count = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith("[Event "):
                count += 1
    return count

def print_progress(current, total, prefix='Tien trinh', length=40):
    # Cập nhật thanh tiến trình trên cùng một dòng console
    if total == 0:
        return
    percent = f"{100 * (current / float(total)):.1f}"
    filled = int(length * current // total)
    bar = '=' * filled + '-' * (length - filled)
    sys.stdout.write(f"\r{prefix}: [{bar}] {percent}% ({current}/{total})")
    sys.stdout.flush()
    if current == total:
        print()

def process_pgn(player_focus="Fischer", mode="style", force=False):
    cols = [f"sq_{i}" for i in range(64)]
    cols.extend(["mat_diff", "mobility", "center_ctrl", "is_check", "king_safety_old"])
    cols.extend(["pst_diff", "rook_open_file", "king_attack", "game_phase"])
    cols.extend(["my_hung_val", "opp_hung_val", "my_mvh", "opp_mvh", "my_attacked_val", "opp_attacked_val"])
    cols.extend(["my_doubled", "opp_doubled", "my_isolated", "opp_isolated", "my_passed", "opp_passed"])
    cols.extend(["my_knight_outpost", "opp_knight_outpost", "my_bishop_pair", "opp_bishop_pair", "my_space", "opp_space"])
    cols.extend(["my_king_danger", "opp_king_danger"])
    cols.append("is_style_focus")
    cols.append("target")

    if mode == "style":
        if not player_focus:
            print("Loi: Vui long cung cap ten Kien tuong de hoc phong cach.")
            sys.exit(1)
            
        input_file = get_raw_pgn_path(player_focus)
        output_parquet = get_processed_parquet_path(player_focus)
        
        if output_parquet.exists() and not force:
            print(f"File Parquet {output_parquet} da ton tai. Dung --force de ghi de.")
            sys.exit(1)
            
        print(f"Dang quet file {input_file.name} de dem tong so van co...")
        total_games = count_games_in_pgn(input_file)
        
        print(f"Bat dau hoc phong cach cua {player_focus}. Tong cong: {total_games} van.")
        all_data = []
        current_game = 0
        
        with open(input_file, "r", encoding="utf-8") as pgn:
            while True:
                game = chess.pgn.read_game(pgn)
                if game is None: break
                    
                result_str = game.headers.get("Result", "*")
                if result_str == "1-0": white_score, black_score = 1.0, -1.0
                elif result_str == "0-1": white_score, black_score = -1.0, 1.0
                elif result_str == "1/2-1/2": white_score, black_score = 0.0, 0.0
                else: 
                    current_game += 1
                    print_progress(current_game, total_games, prefix="Dang xu ly")
                    continue 
                    
                game_data = process_game_style(game, player_focus, white_score, black_score)
                all_data.extend(game_data)
                
                current_game += 1
                print_progress(current_game, total_games, prefix="Dang xu ly")

        df = pd.DataFrame(all_data, columns=cols)
        df.to_parquet(output_parquet, index=False)
        print(f"Hoan tat Style! Luu thanh cong {len(df)} dong du lieu.")

    elif mode == "base":
        input_folder = BASE_DIR
        output_parquet = PROCESSED_BASE_DIR

        if output_parquet.exists() and not force:
            print(f"File Parquet {output_parquet} da ton tai. Dung --force de ghi de.")
            sys.exit(1)
        elif output_parquet.exists() and force:
            output_parquet.unlink()

        if not input_folder.exists() or not input_folder.is_dir():
            print(f"Vui long tao thu muc {input_folder} va copy cac file PGN tong hop vao do!")
            input_folder.mkdir(parents=True, exist_ok=True)
            sys.exit(1)

        pgn_files = list(input_folder.rglob("*.pgn"))
        
        print("Dang quet cac file de dem tong so van co...")
        total_games_all_files = 0
        for f in pgn_files:
            total_games_all_files += count_games_in_pgn(f)
            
        print(f"Tim thay {len(pgn_files)} file PGN. Tong cong: {total_games_all_files} van co.")

        CHUNK_SIZE = 2000
        buffer_data = []
        current_game_global = 0

        for pgn_file in pgn_files:
            print(f"Dang doc file: {pgn_file.name}")
            with open(pgn_file, "r", encoding="utf-8") as pgn:
                while True:
                    game = chess.pgn.read_game(pgn)
                    if game is None: break
                        
                    result_str = game.headers.get("Result", "*")
                    if result_str == "1-0": white_score, black_score = 1.0, -1.0
                    elif result_str == "0-1": white_score, black_score = -1.0, 1.0
                    elif result_str == "1/2-1/2": white_score, black_score = 0.0, 0.0
                    else:
                        current_game_global += 1
                        print_progress(current_game_global, total_games_all_files, prefix="Tong tien trinh")
                        continue 
                        
                    game_data = process_game_base(game, white_score, black_score)
                    buffer_data.extend(game_data)
                    current_game_global += 1
                    
                    print_progress(current_game_global, total_games_all_files, prefix="Tong tien trinh")

                    if current_game_global % CHUNK_SIZE == 0:
                        df_chunk = pd.DataFrame(buffer_data, columns=cols)
                        
                        if not output_parquet.exists():
                            df_chunk.to_parquet(output_parquet, engine='fastparquet', index=False)
                        else:
                            df_chunk.to_parquet(output_parquet, engine='fastparquet', append=True, index=False)
                            
                        buffer_data = []

        if buffer_data:
            df_chunk = pd.DataFrame(buffer_data, columns=cols)
            if not output_parquet.exists():
                df_chunk.to_parquet(output_parquet, engine='fastparquet', index=False)
            else:
                df_chunk.to_parquet(output_parquet, engine='fastparquet', append=True, index=False)

        print(f"Hoan tat Base! Da luu {current_game_global} van co vao: {output_parquet}")

    else:
        print("Che do huan luyen khong hop le. Chi chap nhan 'base' hoac 'style'.")
        sys.exit(1)