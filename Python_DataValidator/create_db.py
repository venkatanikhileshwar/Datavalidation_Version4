import sqlite3, os

db_path = os.path.abspath("test_data.sqlite")
con = sqlite3.connect(db_path)
cur = con.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS sales (
    record_id TEXT PRIMARY KEY,
    name TEXT,
    amount REAL,
    status TEXT
)
""")

cur.execute("DELETE FROM sales")
cur.executemany(
    "INSERT INTO sales(record_id,name,amount,status) VALUES(?,?,?,?)",
    [
        ("1001","Alice",500,"ACTIVE"),
        ("1002","Bob",520,"ACTIVE"),
        ("1003","Cara",None,"INACTIVE"),
        ("1004","Dan",615,"ACTIVE"),
    ]
)

con.commit()
con.close()
print("✅ Created SQLite DB at:", db_path)
