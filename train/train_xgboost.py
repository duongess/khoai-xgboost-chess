from operator import le

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from config.Setting import get_processed_csv_path, get_model_path, get_version_info, save_new_version

def train_xgboost_model(player: str, is_new: bool = False):
    csv_path = get_processed_csv_path(player)
    df = pd.read_csv(csv_path)
    
    X = df.drop(columns=["target"])
    y_text = df["target"]
    le = LabelEncoder()
    y = le.fit_transform(y_text)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = xgb.XGBRegressor(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, tree_method="hist", random_state=42
    )
    
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=True)
    
    predictions = model.predict(X_test)
    rmse = mean_squared_error(y_test, predictions, squared=False)
    
    # Xử lý logic phiên bản
    version_info = get_version_info(player)
    version = "v1"
    
    if is_new and version_info["versions"]:
        last_v = int(version_info["latest"].replace("v", ""))
        version = f"v{last_v + 1}"
    elif version_info["versions"]:
        version = version_info["latest"]
        
    output_model_path = get_model_path(player + f"_{version}")
    model.save_model(str(output_model_path))
    
    # Cập nhật metadata sau khi lưu model
    save_new_version(player, version)
    print(f"Da luu mo hinh tai: {output_model_path}")