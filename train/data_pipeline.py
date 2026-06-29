import random
import chess.pgn
import numpy as np
import pandas as pd
import sys
from config.Setting import get_processed_parquet_path, get_raw_pgn_path, BASE_DIR, PROCESSED_BASE_DIR
from core.utils import extract_features     

def process_game_base(game, white_score, black_score):
    """CASE 1: Học nền tảng cơ bản (CÓ TẠO DỮ LIỆU ÂM)"""
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
            
        # 1. MẪU TỐT (POSITIVE SAMPLE): Nước cờ thực tế
        board.push(move)
        features = extract_features(board).tolist()
        features.append(0) # is_style_focus = 0
        features.append(target_score)
        data.append(features)
        board.pop()
        
        # 2. MẪU XẤU (NEGATIVE SAMPLES): Dạy AI biết sợ các nước đi bậy bạ
        legal_moves = list(board.legal_moves)
        legal_moves.remove(move)
        num_negatives = min(3, len(legal_moves))
        
        if num_negatives > 0:
            negative_moves = random.sample(legal_moves, num_negatives)
            for neg_move in negative_moves:
                board.push(neg_move)
                features_neg = extract_features(board).tolist()
                features_neg.append(0)
                # Phạt kịch khung (-1.0) cho những nước Đại kiện tướng bỏ qua
                features_neg.append(-1.0) 
                data.append(features_neg)
                board.pop()
                
        # Tiến cờ để qua vòng lặp tiếp theo
        board.push(move)
        
    return data

def process_game_style(game, player_name, white_score, black_score):
    """CASE 2: Học phong cách cá nhân (Chỉ học đúng Kiện tướng được chọn)"""
    data = []
    
    white_player = game.headers.get("White", "")
    black_player = game.headers.get("Black", "")
    is_player_white = player_name.lower() in white_player.lower()
    is_player_black = player_name.lower() in black_player.lower()
    
    if not is_player_white and not is_player_black:
        return data # Không có Kiện tướng này thì bỏ qua
        
    board = game.board()
    moves = list(game.mainline_moves())
    total_moves = len(moves)
    
    for ply, move in enumerate(moves):
        # Kiểm tra trước khi đi: Nước tiếp theo CÓ PHẢI của Kiện tướng không?
        is_player_turn = (board.turn == chess.WHITE and is_player_white) or \
                         (board.turn == chess.BLACK and is_player_black)
                         
        board.push(move)
        
        # Chỉ học những nước do Kiện tướng đi
        if is_player_turn:
            game_outcome = white_score if is_player_white else black_score
            decay = (ply + 1) / total_moves
            target_score = game_outcome * decay
            
            features = extract_features(board).tolist()
            # Thêm cột is_style_focus = 1 (Đánh dấu đây là tuyệt chiêu của GM)
            features.append(1) 
            features.append(target_score)
            data.append(features)
            
    return data

def process_pgn(player_focus="Fischer", mode="style", force=False):
    """
    mode = "base": Đọc TẤT CẢ các file .pgn trong thư mục data/raw/base_pgns
    mode = "style": Đọc 1 file .pgn của Kiện tướng cụ thể
    """
    # 1. Định nghĩa Header chuẩn hóa cho cả 2 trường hợp
    cols = [f"sq_{i}" for i in range(64)]
    cols.extend(["mat_diff", "mobility", "center_ctrl", "is_check", "king_safety"])
    cols.extend(["pst_diff", "rook_open_file", "king_attack", "hanging_pieces_diff"])
    cols.append("is_style_focus") 
    cols.append("target")


    if mode == "style":
        if not player_focus:
            print("Lỗi: Vui lòng cung cấp tên Kiện tướng để học phong cách.")
            sys.exit(1)
            
        input_file = get_raw_pgn_path(player_focus)
        output_parquet = get_processed_parquet_path(player_focus)
        
        if output_parquet.exists() and not force:
            print(f"File Parquet {output_parquet} đã tồn tại. Dùng --force để ghi đè.")
            sys.exit(1)
            
        print(f"Đang học phong cách của {player_focus} từ file {input_file.name}...")
        all_data = []
        game_count = 0
        
        with open(input_file, "r", encoding="utf-8") as pgn:
            while True:
                game = chess.pgn.read_game(pgn)
                if game is None: break
                    
                result_str = game.headers.get("Result", "*")
                if result_str == "1-0": white_score, black_score = 1.0, -1.0
                elif result_str == "0-1": white_score, black_score = -1.0, 1.0
                elif result_str == "1/2-1/2": white_score, black_score = 0.0, 0.0
                else: continue 
                    
                game_data = process_game_style(game, player_focus, white_score, black_score)
                all_data.extend(game_data)
                
                game_count += 1
                if game_count % 100 == 0:
                    print(f"Đã xử lý {game_count} ván cờ...\r", end="")

        df = pd.DataFrame(all_data, columns=cols)
        df.to_parquet(output_parquet, index=False)
        print(f"\nHoàn tất Style! Lưu thành công {len(df)} dòng dữ liệu.")


    elif mode == "base":
        input_folder = BASE_DIR
        output_parquet = PROCESSED_BASE_DIR

        if output_parquet.exists() and not force:
            print(f"File Parquet {output_parquet} đã tồn tại. Dùng --force để ghi đè.")
            sys.exit(1)
        elif output_parquet.exists() and force:
            output_parquet.unlink() # Xóa file cũ đi nếu bật force

        if not input_folder.exists() or not input_folder.is_dir():
            print(f"Vui lòng tạo thư mục {input_folder} và copy các file PGN tổng hợp vào đó!")
            input_folder.mkdir(parents=True, exist_ok=True)
            sys.exit(1)

        pgn_files = list(input_folder.rglob("*.pgn"))
        print(f"Tìm thấy {len(pgn_files)} file PGN tổng hợp trong {input_folder}...")

        CHUNK_SIZE = 2000
        buffer_data = []
        total_games = 0

        for pgn_file in pgn_files:
            print(f"\nĐang đọc file: {pgn_file.name}")
            with open(pgn_file, "r", encoding="utf-8") as pgn:
                while True:
                    game = chess.pgn.read_game(pgn)
                    if game is None: break
                        
                    result_str = game.headers.get("Result", "*")
                    if result_str == "1-0": white_score, black_score = 1.0, -1.0
                    elif result_str == "0-1": white_score, black_score = -1.0, 1.0
                    elif result_str == "1/2-1/2": white_score, black_score = 0.0, 0.0
                    else: continue 
                        
                    game_data = process_game_base(game, white_score, black_score)
                    buffer_data.extend(game_data)
                    total_games += 1

                    # Cơ chế chống tràn RAM (Xả đệm vào Parquet từng đợt)
                    if total_games % CHUNK_SIZE == 0:
                        df_chunk = pd.DataFrame(buffer_data, columns=cols)
                        
                        # Dùng fastparquet và append=True, tuyệt đối KHÔNG dùng pyarrow với mode='a'
                        if not output_parquet.exists():
                            df_chunk.to_parquet(output_parquet, engine='fastparquet', index=False)
                        else:
                            df_chunk.to_parquet(output_parquet, engine='fastparquet', append=True, index=False)
                            
                        print(f"Đã xả đệm {CHUNK_SIZE} ván. Tổng cộng: {total_games} ván")
                        buffer_data = [] # Xóa sạch RAM

        # Xả nốt lượng ván cờ còn dư
        if buffer_data:
            df_chunk = pd.DataFrame(buffer_data, columns=cols)
            if not output_parquet.exists():
                df_chunk.to_parquet(output_parquet, engine='fastparquet', index=False)
            else:
                df_chunk.to_parquet(output_parquet, engine='fastparquet', append=True, index=False)

        print(f"\nHoàn tất Base! Đã lưu tổng cộng {total_games} ván cờ vào: {output_parquet}")

    else:
        print("Chế độ huấn luyện không hợp lệ. Chỉ chấp nhận 'base' hoặc 'style'.")
        sys.exit(1)