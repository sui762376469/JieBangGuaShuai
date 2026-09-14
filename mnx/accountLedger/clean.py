# import sqlite3
# conn = sqlite3.connect('./complaints_ledger.db')
# cursor = conn.cursor()
# cursor.execute("SELECT * FROM sqlite_sequence WHERE name='ledger'")
# print(cursor.fetchone())  # 如果返回 None 说明已被删除
# cursor.execute("SELECT min(id), max(id) FROM ledger")
# print(cursor.fetchone())  # 查看当前最小/最大ID
# conn.close()


import sqlite3
conn = sqlite3.connect('./complaints_ledger.db')
cursor = conn.cursor()
cursor.execute("DELETE FROM ledger")
cursor.execute("DELETE FROM sqlite_sequence WHERE name='ledger'")
conn.commit()
conn.close()
print("数据库已彻底清空，下次入库ID将从1开始")