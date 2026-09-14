import sqlite3
import pandas as pd

DB_NAME = "complaints_ledger.db"

def init_db():
    """初始化数据库：创建线索表"""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    # 创建表结构
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS ledger (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT,              -- 投诉内容（已脱敏）
            pred_category INTEGER,     -- 预测类别 (1-12)
            status TEXT,               -- 处理状态
            manual_review TEXT,        -- 人工复核结果
            create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def save_to_db(df):
    """将数据帧 DataFrame 批量存入数据库"""
    conn = sqlite3.connect(DB_NAME)
    # 将 DataFrame 的列名映射到数据库字段
    # 注意：这里只保存关键字段
    db_df = df[['投诉详情', '预测类别', '处理状态', '人工复核']].copy()
    db_df.columns = ['content', 'pred_category', 'status', 'manual_review']
    db_df.to_sql('ledger', conn, if_exists='append', index=False)
    conn.close()

def load_from_db():
    """从数据库读取所有台账数据"""
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql_query("SELECT * FROM ledger ORDER BY create_time DESC", conn)
    conn.close()
    return df

def update_db_record(row_id, field, value):
    """更新单条记录（用于人工复核修改）"""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    query = f"UPDATE ledger SET {field} = ? WHERE id = ?"
    cursor.execute(query, (value, row_id))
    conn.commit()
    conn.close()