# khoai-xgboost-chess

/khoai-xgboost-chess
  ├── /data
  │    ├── /raw                // Bỏ file Fischer.pgn nguyên bản vào đây
  │    └── /processed          // Chứa file CSV/Ma trận số sau khi chạy vòng for
  ├── /models                  // Nơi chứa "não" của AI (các file .json hoặc .bin)
  ├── /core                    // 4 file luồng suy luận lúc nãy ông chốt
  ├── /training
  │    ├── data_pipeline.py    // Chứa vòng for để cào PGN thành CSV
  │    └── train_xgboost.py    // Đọc CSV, train AI và xuất ra file ném vào /models
  └── .gitignore