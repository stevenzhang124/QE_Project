import mysql.connector

def db_connection():
    try:
        mydb = mysql.connector.connect(
            host="192.168.1.113",
            user="QE_manager",
            port="3306",
            database="QE",
            passwd="edgeimcl",
            autocommit=True)
        return mydb
    except mysql.connector.Error as err:
      if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
        print("Something is wrong with your user name or password")
      elif err.errno == errorcode.ER_BAD_DB_ERROR:
        print("Database does not exist")
      else:
        print(err)
    else:
        return mydb

mydb = db_connection()
cursor = mydb.cursor()

action = 'leaving bed'
info = "insert into leavebed (time, bed, status) values ({}, {}, '{}')".format('NOW()', '1', action)
cursor.execute(info)
info = "insert into leavebed (time, bed, status) values ({}, {}, '{}')".format('NOW()', '2', 'on bed')
cursor.execute(info)

case = 0
info = "insert into handhygiene (time, person, incompliance) values ({}, '{}', {})".format('NOW()', 'doctor', case)
cursor.execute(info)

