import sqlite3
import os

def txt_to_sqlite(txt_path, db_path, table_name="news_data", batch_size=10000):
    """
    将 label\\ttext 格式的文本文件存入 SQLite 数据库

    Args:
        txt_path:  输入的文本文件路径
        db_path:   输出的 SQLite 数据库路径
        table_name: 表名，默认为 news_data
        batch_size: 批量插入的行数，提升大数据量时的插入效率
    """
    if not os.path.exists(txt_path):
        print(f"文件不存在: {txt_path}")
        return

    # 连接数据库（如果不存在会自动创建）
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 建表：label(文本分类) + text(文本内容) + id(自增主键)
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            label TEXT NOT NULL,
            text  TEXT NOT NULL
        )
    """)

    # 清空已有数据（如需保留旧数据可注释掉这一行）
    cursor.execute(f"DELETE FROM {table_name}")

    # 逐行读取并批量插入
    batch = []
    total = 0

    with open(txt_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue  # 跳过空行

            parts = line.split("\t", 1)  # 只按第一个 \\t 分割
            if len(parts) != 2:
                print(f"第 {line_no} 行格式异常，已跳过: {line[:50]}...")
                continue

            label, text = parts
            batch.append((label, text))

            # 达到批次大小时执行插入
            if len(batch) >= batch_size:
                cursor.executemany(
                    f"INSERT INTO {table_name} (label, text) VALUES (?, ?)",
                    batch
                )
                conn.commit()
                total += len(batch)
                print(f"已插入 {total} 条记录...")
                batch = []

    # 插入剩余的记录
    if batch:
        cursor.executemany(
            f"INSERT INTO {table_name} (label, text) VALUES (?, ?)",
            batch
        )
        conn.commit()
        total += len(batch)

    # 验证结果
    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
    count = cursor.fetchone()[0]
    cursor.execute(f"SELECT DISTINCT label FROM {table_name} ORDER BY label")
    labels = [row[0] for row in cursor.fetchall()]

    print(f"\n===== 完成 =====")
    print(f"数据库路径: {os.path.abspath(db_path)}")
    print(f"表名: {table_name}")
    print(f"总记录数: {count}")
    print(f"标签类别 ({len(labels)}): {labels}")

    conn.close()


if __name__ == "__main__":
    # ===== 配置区 =====
    # 输入文本文件路径（支持 simulated_data.txt、thucnews_train_10w.txt 等）
    txt_path = r"./dataset/simulated_data.txt"
    # 输出 SQLite 数据库路径
    db_path = r"./news_data1.db"
    # ==================

    txt_to_sqlite(txt_path, db_path)
