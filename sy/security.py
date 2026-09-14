import hashlib
import json
from datetime import datetime, timedelta

# 配置
MAX_FAILED_ATTEMPTS = 5
LOCKOUT_DURATION = 30  # 锁定分钟数

# 用户数据文件路径
USERS_FILE = "users.json"

# 简单的密码哈希函数
def hash_password(password):
    """使用 SHA-256 哈希密码"""
    return hashlib.sha256(password.encode('utf-8')).hexdigest()

def verify_password(plain_password, hashed_password):
    """验证密码"""
    return hash_password(plain_password) == hashed_password

def load_users():
    """加载用户数据"""
    try:
        with open(USERS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        # 初始化用户数据
        init_data = {
            "users": [
                {
                    "username": "admin",
                    "password_hash": hash_password("123456"),
                    "failed_attempts": 0,
                    "last_failed_login": None
                },
                {
                    "username": "user",
                    "password_hash": hash_password("user123"),
                    "failed_attempts": 0,
                    "last_failed_login": None
                }
            ]
        }
        save_users(init_data)
        return init_data

def save_users(users_data):
    """保存用户数据"""
    with open(USERS_FILE, 'w', encoding='utf-8') as f:
        json.dump(users_data, f, indent=2, ensure_ascii=False)

def check_user_locked(username):
    """检查用户是否被锁定"""
    users_data = load_users()
    for user in users_data["users"]:
        if user["username"] == username:
            if user["failed_attempts"] >= MAX_FAILED_ATTEMPTS:
                last_failed = user["last_failed_login"]
                if last_failed:
                    lockout_time = datetime.fromisoformat(last_failed)
                    if datetime.now() - lockout_time < timedelta(minutes=LOCKOUT_DURATION):
                        return True
                    else:
                        # 解锁用户
                        user["failed_attempts"] = 0
                        user["last_failed_login"] = None
                        save_users(users_data)
            return False
    return False

def increment_failed_attempts(username):
    """增加失败尝试次数"""
    users_data = load_users()
    for user in users_data["users"]:
        if user["username"] == username:
            user["failed_attempts"] += 1
            user["last_failed_login"] = datetime.now().isoformat()
            save_users(users_data)
            break

def reset_failed_attempts(username):
    """重置失败尝试次数"""
    users_data = load_users()
    for user in users_data["users"]:
        if user["username"] == username:
            user["failed_attempts"] = 0
            user["last_failed_login"] = None
            save_users(users_data)
            break

def authenticate_user(username, password):
    """认证用户"""
    # 检查用户是否被锁定
    if check_user_locked(username):
        return False, "账户已被锁定，请30分钟后再试"
    
    users_data = load_users()
    for user in users_data["users"]:
        if user["username"] == username:
            if verify_password(password, user["password_hash"]):
                reset_failed_attempts(username)
                return True, None
            else:
                increment_failed_attempts(username)
                return False, "用户名或密码错误"
    return False, "用户不存在"
