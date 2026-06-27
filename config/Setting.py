from pathlib import Path

import chess

# Thư mục chứa file Setting.py (config/)
CONFIG_DIR = Path(__file__).resolve().parent
SIZE = 650
SIZE_SQUARE = SIZE // 8

# Gốc của toàn bộ dự án (thư mục cha của config/)
BASE_DIR = CONFIG_DIR.parent

# Định nghĩa các thư mục chức năng chính
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
PLAY_DIR = DATA_DIR / "play"
MODEL_DIR = BASE_DIR / "models"
PUBLIC_DIR = BASE_DIR / "public"

# Tự động tạo thư mục nếu chưa tồn tại trong hệ thống để tránh lỗi FileNotFoundError
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
PUBLIC_DIR.mkdir(parents=True, exist_ok=True)
PLAY_DIR.mkdir(parents=True, exist_ok=True)

PIECE_VALUES = {
    chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
    chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0
}

# Các hàm sinh đường dẫn động phục vụ trực tiếp cho các tham số từ CLI
def get_raw_pgn_path(player_name: str) -> Path:
    # Trả về đường dẫn file PGN thô đầu vào (Ví dụ: data/raw/Fischer.pgn)
    return RAW_DATA_DIR / f"{player_name}.pgn"

def get_processed_csv_path(player_name: str) -> Path:
    # Trả về đường dẫn file CSV sau khi chạy vòng for bóc tách (Ví dụ: data/processed/Fischer.csv)
    return PROCESSED_DATA_DIR / f"{player_name}.csv"

def get_model_path(model_name: str, version: str = "v1") -> Path:
    # Trả về đường dẫn file lưu mô hình XGBoost sau khi train (Ví dụ: models/Fischer_v1.json)
    print(f"Model path: {MODEL_DIR / f'{model_name}_{version}.json'}")
    return MODEL_DIR / f"{model_name}_{version}.json"

def get_play_path(model_name: str) -> Path:
    # Trả về đường dẫn file lưu trữ các trận đấu đã chơi (Ví dụ: play/Fischer_played.pgn)
    return PLAY_DIR / f"{model_name}_played.pgn"

def get_chess_pieces(key: str) -> str:
    # Trả về ký hiệu Unicode của quân cờ dựa trên ký hiệu chữ
    chess_pieces = {
        'R': 'Chess_rlt45.svg', 'N': 'Chess_nlt45.svg', 'B': 'Chess_blt45.svg', 'Q': 'Chess_qlt45.svg', 'K': 'Chess_klt45.svg', 'P': 'Chess_plt45.svg',
        'r': 'Chess_rdt45.svg', 'n': 'Chess_ndt45.svg', 'b': 'Chess_bdt45.svg', 'q': 'Chess_qdt45.svg', 'k': 'Chess_kdt45.svg', 'p': 'Chess_pdt45.svg',
        '.': '.'
    }
    return PUBLIC_DIR / chess_pieces.get(key, '.')

def get_chess_piece_unicode(key: str) -> str:
    # Trả về ký hiệu Unicode của quân cờ dựa trên ký hiệu chữ
    chess_pieces_unicode = {
         # Trắng (chữ hoa) - ông muốn là ♖ (nhìn đặc/đậm hơn)
        'R': '♜', 'N': '♞', 'B': '♝', 'Q': '♛', 'K': '♚', 'P': '♟',
        # Đen (chữ thường) - ông muốn là ♜ (nhìn rỗng/sáng hơn)
        'r': '♖', 'n': '♘', 'b': '♗', 'q': '♕', 'k': '♔', 'p': '♙',
        '.': '·'
    }
    return chess_pieces_unicode.get(key, '.')