# Khoai XGBoost Chess

## Tổng quan dự án
Khoai XGBoost Chess là một Chess Engine được phát triển để mô phỏng phong cách thi đấu của các Đại kiện tướng (Grandmasters). 

Dự án tiếp cận bằng Machine Learning. Cụ thể, mô hình XGBoost sẽ học cách đánh giá (evaluate) từng trạng thái bàn cờ dựa trên dữ liệu hàng ngàn ván đấu thực tế của một kỳ thủ cụ thể (ví dụ: Bobby Fischer). Mục tiêu là tạo ra một AI có khả năng ra quyết định nhanh, mang đậm phong cách cá nhân của kỳ thủ được huấn luyện, và có thể tương tác trực tiếp với người dùng qua giao diện dòng lệnh hoặc đồ họa.

## Cấu trúc thư mục
Dự án được tổ chức theo kiến trúc module như sau:

```
/khoai-xgboost-chess
  ├── config/                  # Chứa file Setting.py quản lý các đường dẫn và tham số cấu hình chung.
  ├── core/                    # Xử lý logic cốt lõi.
  │    ├── model_inference.py  # Load mô hình XGBoost và dự đoán điểm số cho các nước đi.
  │    └── utils.py            # Các hàm hỗ trợ chuyển đổi bàn cờ sang mảng dữ liệu.
  ├── data/                    # Thư mục chứa dữ liệu (cần được người dùng tạo).
  │    ├── models/             # Nơi lưu file .json của mô hình sau khi huấn luyện.
  │    └── ...                 # Nơi chứa các file .pgn gốc.
  ├── play/                    # Xử lý tương tác với người chơi.
  │    ├── gui.py              # Giao diện đồ họa sử dụng PySide6.
  │    └── terminal.py         # Giao diện dòng lệnh (CLI).
  ├── public/                  # Chứa các file ảnh SVG thiết kế quân cờ cho giao diện GUI.
  ├── training/                # Pipeline xử lý dữ liệu và huấn luyện.
  │    ├── data_pipeline.py    # Đọc file PGN, trích xuất nước đi và chuyển thành định dạng CSV.
  │    └── train_xgboost.py    # Huấn luyện mô hình XGBRegressor từ dữ liệu CSV.
  ├── main.py                  # Entry point chính của ứng dụng sử dụng Typer.
  ├── poetry.lock              # Khóa phiên bản các package phụ thuộc.
  └── pyproject.toml           # Tệp cấu hình dự án và quản lý thư viện của Poetry.
```

## Hướng dẫn sử dụng
1. Cài đặt môi trường: Chạy lệnh `poetry install`.
2. Huấn luyện: Đặt file PGN vào thư mục data, sau đó chạy `poetry run python main.py train <tên_kỳ_thủ>`.
3. Chơi game: Chạy `poetry run python main.py play <tên_kỳ_thủ> --ui gui` (hoặc `--ui term`).