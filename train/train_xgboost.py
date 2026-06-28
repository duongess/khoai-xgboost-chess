import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from config.Setting import get_model_path, get_processed_parquet_path, get_version_info, save_new_version, PROCESSED_BASE_DIR

def train_xgboost_model(player, mode, is_new: bool = False):
    # 1. Xac dinh Data Path va Model Name dua tren Mode
    if mode == "style":
        parquet_path = get_processed_parquet_path(player)
        model_name = player # Ten thu muc luu model la ten kien tuong
    elif mode == "base":
        parquet_path = PROCESSED_BASE_DIR 
        model_name = PROCESSED_BASE_DIR.stem # Trich xuat ten tu duong dan (VD: Base_Grandmasters)
    else:
        raise ValueError("Mode chi duoc la 'base' hoac 'style'")

    print(f"Dang load du lieu tu: {parquet_path}")
    df = pd.read_parquet(parquet_path, engine='fastparquet')
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Tach tap dac trung (X) va nhan (y)
    X = df.drop(columns=["target"])
    y = df["target"] 
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 2. Xu ly logic Huan luyen
    if mode == "style":
        base_model_name = PROCESSED_BASE_DIR.stem
        print(f"Dang tinh chinh (Fine-tuning) tu mo hinh goc: {base_model_name}...")
        
        xgb_params = {
            "n_estimators": 50,
            "max_depth": 6,
            "learning_rate": 0.01,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "tree_method": "hist",
            "random_state": 42
        }
        model = xgb.XGBRegressor(**xgb_params)
        
        # Load thong tin phien ban cua Base Model de ke thua
        base_v_info = get_version_info(base_model_name)
        base_model_path = str(get_model_path(f"{base_model_name}_{base_v_info['latest']}"))
        
        # Transfer Learning
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], xgb_model=base_model_path, verbose=True)
    else:
        print(f"Dang huan luyen mo hinh goc tu dau...")
        model = xgb.XGBRegressor(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, tree_method="hist", random_state=42
        )
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=True)
    
    # 3. Danh gia mo hinh
    predictions = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, predictions)
    print(f"Sai so RMSE tren tap Test: {rmse:.4f}")
    
    # 4. Luu Model va Cap nhat Metadata
    version_info = get_version_info(model_name)
    version = "v1"
    
    # Tang phien ban neu is_new duoc bat
    if is_new and version_info["versions"]:
        last_v = int(version_info["latest"].replace("v", ""))
        version = f"v{last_v + 1}"
    elif version_info["versions"]:
        version = version_info["latest"]
        
    output_model_path = get_model_path(f"{model_name}_{version}")
    model.save_model(str(output_model_path))
    
    save_new_version(model_name, version)
    print(f"Da luu mo hinh tai: {output_model_path}")