import pymysql

conn = pymysql.connect(host='localhost',
                       user='root',
                       password='1234',
                       db='parking_protecter',
                       charset='utf8')

sql = "SELECT * FROM exam where Car_number = %s"

with conn:
    with conn.cursor() as cur:
        cur.execute(sql, ('67무7664'))
        result = cur.fetchall()
        for data in result:
            print(data)