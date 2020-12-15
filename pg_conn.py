# YOU PROBABLY WANT TO DO THIS IN A SMARTER WAY
# VALUES MUST BE EDITED BELOW

import psycopg2

database = ''
tablespace = ''
username=''
password=''
host='localhost'
port=5432
dbtype='postgres'

def make_conn():
    conn = psycopg2.connect(
        database=database,
        user=username,
        host=host,
        port=port,
        password=password
    )
    return conn
