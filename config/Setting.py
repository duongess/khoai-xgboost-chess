from datetime import datetime
import json
from pathlib import Path

import chess

# Thư mục chứa file Setting.py (config/)
CONFIG_DIR = Path(__file__).resolve().parent
SIZE = 650
SIZE_SQUARE = SIZE // 8

# Gốc của toàn bộ dự án (thư mục cha của config/)
ROOT_DIR = CONFIG_DIR.parent

# Định nghĩa các thư mục chức năng chính
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
PLAY_DIR = DATA_DIR / "played"
MODEL_DIR = DATA_DIR / "models"
BASE_DIR = RAW_DATA_DIR / "train-base"
PROCESSED_BASE_DIR = PROCESSED_DATA_DIR / "train-base.parquet"
PUBLIC_DIR = ROOT_DIR / "public"

METADATA_FILE = DATA_DIR / "metadata.json"

# Tự động tạo thư mục nếu chưa tồn tại trong hệ thống để tránh lỗi FileNotFoundError
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
PUBLIC_DIR.mkdir(parents=True, exist_ok=True)
PLAY_DIR.mkdir(parents=True, exist_ok=True)
if not METADATA_FILE.exists() or METADATA_FILE.stat().st_size == 0:
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump({}, f)

PIECE_VALUES = {
    chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
    chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0
}

PAWN_PST = [
      0,   0,   0,   0,   0,   0,   0,   0,
     50,  50,  50,  50,  50,  50,  50,  50,
     10,  10,  20,  30,  30,  20,  10,  10,
      5,   5,  10,  25,  25,  10,   5,   5,
      0,   0,   0,  20,  20,   0,   0,   0,
      5,  -5, -10,   0,   0, -10,  -5,   5,
      5,  10,  10, -20, -20,  10,  10,   5,
      0,   0,   0,   0,   0,   0,   0,   0
]

KNIGHT_PST = [
    -50, -40, -30, -30, -30, -30, -40, -50,
    -40, -20,   0,   0,   0,   0, -20, -40,
    -30,   0,  10,  15,  15,  10,   0, -30,
    -30,   5,  15,  20,  20,  15,   5, -30,
    -30,   0,  15,  20,  20,  15,   0, -30,
    -30,   5,  10,  15,  15,  10,   5, -30,
    -40, -20,   0,   5,   5,   0, -20, -40,
    -50, -40, -30, -30, -30, -30, -40, -50
]

BISHOP_PST = [
    -20, -10, -10, -10, -10, -10, -10, -20,
    -10,   0,   0,   0,   0,   0,   0, -10,
    -10,   0,   5,  10,  10,   5,   0, -10,
    -10,   5,   5,  10,  10,   5,   5, -10,
    -10,   0,  10,  10,  10,  10,   0, -10,
    -10,  10,  10,  10,  10,  10,  10, -10,
    -10,   5,   0,   0,   0,   0,   5, -10,
    -20, -10, -10, -10, -10, -10, -10, -20
]

ROOK_PST = [
      0,   0,   0,   0,   0,   0,   0,   0,
      5,  10,  10,  10,  10,  10,  10,   5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
      0,   0,   0,   5,   5,   0,   0,   0
]

QUEEN_PST = [
    -20, -10, -10,  -5,  -5, -10, -10, -20,
    -10,   0,   0,   0,   0,   0,   0, -10,
    -10,   0,   5,   5,   5,   5,   0, -10,
     -5,   0,   5,   5,   5,   5,   0,  -5,
      0,   0,   5,   5,   5,   5,   0,  -5,
    -10,   5,   5,   5,   5,   5,   0, -10,
    -10,   0,   5,   0,   0,   0,   0, -10,
    -20, -10, -10,  -5,  -5, -10, -10, -20
]

KING_PST = [
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -20, -30, -30, -40, -40, -30, -30, -20,
    -10, -20, -20, -20, -20, -20, -20, -10,
     20,  20,   0,   0,   0,   0,  20,  20,
     20,  30,  10,   0,   0,  10,  30,  20
]

def get_version_info(model_name: str) -> dict:
    if not METADATA_FILE.exists():
        return {"latest": "v1", "versions": []}
    with open(METADATA_FILE, "r") as f:
        data = json.load(f)
    return data.get(model_name, {"latest": "v1", "versions": []})

# Ghi nhận phiên bản mới vào metadata
def save_new_version(model_name: str, new_version: str):
    data = {}
    if METADATA_FILE.exists():
        with open(METADATA_FILE, "r") as f:
            data = json.load(f)
            
    player_data = data.get(model_name, {"latest": "v1", "versions": []})
    if new_version not in player_data["versions"]:
        player_data["versions"].append(new_version)
    player_data["latest"] = new_version
    data[model_name] = player_data
    
    with open(METADATA_FILE, "w") as f:
        json.dump(data, f, indent=4)

def save_game_history(model_name: str, version: str, color: str, result: str):
    # Xác định kết quả từ góc nhìn của người chơi (Human)
    if result == "1-0":
        outcome = "win" if color == "white" else "loss"
    elif result == "0-1":
        outcome = "win" if color == "black" else "loss"
    elif result == "1/2-1/2":
        outcome = "draw"
    else:
        outcome = "unfinished"
        
    if METADATA_FILE.exists():
        with open(METADATA_FILE, "r") as f:
            data = json.load(f)
    else:
        data = {}

    # Khởi tạo cấu trúc nếu chưa tồn tại
    player_data = data.get(model_name, {"latest": version, "versions": [version], "played": []})
    if "played" not in player_data:
        player_data["played"] = []

    # Tạo bản ghi mới
    record = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "color": color,
        "result": result,
        "outcome": outcome
    }

    # Chèn vào đầu danh sách để ván mới nhất luôn ở trên cùng
    player_data["played"].insert(0, record)
    
    data[model_name] = player_data

    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

# Các hàm sinh đường dẫn động phục vụ trực tiếp cho các tham số từ CLI
def get_raw_pgn_path(player_name: str) -> Path:
    # Trả về đường dẫn file PGN thô đầu vào (Ví dụ: data/raw/Fischer.pgn)
    return RAW_DATA_DIR / f"{player_name}.pgn"

def get_processed_parquet_path(player_name: str) -> Path:
    # Trả về đường dẫn file Parquet sau khi chạy vòng for bóc tách (Ví dụ: data/processed/Fischer.parquet)
    return PROCESSED_DATA_DIR / f"{player_name}.parquet"

def get_model_path(model_name: str) -> Path:
    # Trả về đường dẫn file lưu mô hình XGBoost sau khi train (Ví dụ: models/Fischer_v1.json)
    print(f"Model path: {MODEL_DIR / f'{model_name}.json'}")
    return MODEL_DIR / f"{model_name}.json"

def get_play_path(model_name: str) -> Path:
    # Trả về đường dẫn file lưu trữ các trận đấu đã chơi (Ví dụ: played/Fischer_played.pgn)
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