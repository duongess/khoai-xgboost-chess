import chess
import xgboost as xgb
from config.Setting import get_model_path
from core.ui import print_board
from core.model_inference import get_ai_move

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

    print("Tran dau ket thuc!")
    print(board.result())