import sqlite3
conn = sqlite3.connect('AdventureWorks-sqlite.db')
conn.text_factory = lambda x: str(x, "utf-8", "ignore") 

cursor = conn.cursor()
query = '''SELECT FirstName FROM Customer'''
cursor.execute(query)
result = cursor.fetchall()
print(result)