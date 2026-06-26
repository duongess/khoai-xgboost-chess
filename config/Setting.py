from pathlib import Path

# Thư mục chứa file Setting.py (config/)
CONFIG_DIR = Path(__file__).resolve().parent

# Gốc của toàn bộ dự án (thư mục cha của config/)
BASE_DIR = CONFIG_DIR.parent

# Định nghĩa các thư mục chức năng chính
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = BASE_DIR / "models"

# Tự động tạo thư mục nếu chưa tồn tại trong hệ thống để tránh lỗi FileNotFoundError
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Các hàm sinh đường dẫn động phục vụ trực tiếp cho các tham số từ CLI
def get_raw_pgn_path(player_name: str) -> Path:
    # Trả về đường dẫn file PGN thô đầu vào (Ví dụ: data/raw/Fischer.pgn)
    return RAW_DATA_DIR / f"{player_name}.pgn"

def get_processed_csv_path(player_name: str) -> Path:
    # Trả về đường dẫn file CSV sau khi chạy vòng for bóc tách (Ví dụ: data/processed/Fischer.csv)
    return PROCESSED_DATA_DIR / f"{player_name}.csv"

def get_model_path(player_name: str, version: str = "v1") -> Path:
    # Trả về đường dẫn file lưu mô hình XGBoost sau khi train (Ví dụ: models/Fischer_v1.json)
    return MODEL_DIR / f"{player_name}_{version}.json"