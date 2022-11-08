import pymysql

conn = pymysql.connect(host='localhost',
                       user='root',
                       password='1234',
                       db='parking_protecter',
                       charset='utf8')

sql = "SELECT * FROM disabled_vehicle"

with conn:
    with conn.cursor() as cur:
        cur.execute(sql)
        result = cur.fetchall()
        for data in result:
            print(data)