import chess
import xgboost as xgb
from config.Setting import get_chess_piece_unicode, get_model_path
from core.model_inference import get_ai_move


def print_board(board: chess.Board, color: str = "white"):
    
    fen = board.fen().split(' ')[0]
    rows = fen.split('/')
    
    # Nếu chơi quân Đen, lật ngược bàn cờ để hàng 1 lên trên
    if color == "black":
        rows.reverse()
        row_indices = [1, 2, 3, 4, 5, 6, 7, 8]
    else:
        # Nếu chơi quân Trắng, hàng 8 ở trên
        row_indices = [8, 7, 6, 5, 4, 3, 2, 1]

    print("\n  a b c d e f g h")
    print("  ---------------")
    
    for idx, row in zip(row_indices, rows):
        line = f"{idx}|"
        for char in row:
            if char.isdigit():
                line += (get_chess_piece_unicode('.') + " ") * int(char)
            else:
                line += get_chess_piece_unicode(char) + " "
        print(line + f"|{idx}")
        
    print("  ---------------")
    print("  a b c d e f g h\n")

def play_term(board, color, model_name):
    """
    Giữa người và AI.
    """
    model = xgb.XGBRegressor()
    model.load_model(str(get_model_path(model_name)))

    while not board.is_game_over():
        print_board(board, color)
        
        # Kiểm tra lượt của người
        if (board.turn == chess.WHITE and color == "white") or (board.turn == chess.BLACK and color == "black"):
            move_san = input("Nhap nuoc di (VD: e4, b4, Nf3): ")
            try:
                # Thay push_uci bằng parse_san
                move = board.parse_san(move_san) 
                board.push(move)
            except ValueError:
                print("Nuoc di khong hop le hoặc định dạng sai. Thu lai.")
        else:
            # AI RA ĐÒN
            print("AI dang suy nghi...")
            move = get_ai_move(board, model)
            board.push(move)