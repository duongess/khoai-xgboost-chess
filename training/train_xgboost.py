from operator import le

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from config.Setting import get_processed_csv_path, get_model_path

def train_fischer_model(player: str):
    # Duong dan lay file CSV da tiền xử lý
    csv_path = get_processed_csv_path(player)
    df = pd.read_csv(csv_path)
    
    # Tach bien dac trung (X la 64 o co) va nhan target (y la diem so the co)
    X = df.drop(columns=["target"])
    y_text = df["target"]
    le = LabelEncoder()
    y = le.fit_transform(y_text)
    
    # Tách tap dữ liệu để kiem dinh chat luong mo hinh
    # Cho du an ngan han, tam thoi dung test_size=0.2
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Khoi tao mo hinh hoi quy XGBoost voi cac tham so co ban chong overfitting
    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist", # Toi uu toc do huan luyen tren CPU
        random_state=42
    )
    
    # Tien hanh huan luyen
    print(f"Bat dau huan luyen mo hinh cho {player}...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=True
    )
    
    # Danh gia sai so tren tap kiem thu
    predictions = model.predict(X_test)
    rmse = mean_squared_error(y_test, predictions, squared=False)
    print(f"Chi so RMSE tren tap test: {rmse:.4f}")
    
    # Luu mo hinh xuong thu muc models theo dinh dang chuan JSON cua XGBoost
    output_model_path = get_model_path(player, version="v1")
    model.save_model(str(output_model_path))
    print(f"Da luu mo hinh tai: {output_model_path}")