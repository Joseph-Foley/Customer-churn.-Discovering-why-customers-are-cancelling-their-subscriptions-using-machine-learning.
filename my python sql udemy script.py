# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 17:50:33 2019

@author: josep
"""
####COMPLETE GUIDE####
import sqlalchemy
import pandas as pd

#make engine (or 'mysql' or 'mysql + pymysql')
engine=sqlalchemy.create_engine('postgresql://postgres:nicaragua13@localhost/dvdrental')

#read
test2 = pd.read_sql('SELECT * FROM customer' , engine)

#write
test2.to_sql('pandas customer', engine)

#thats it :)






#edit again

import psycopg2 as pg2
#connect
conn = pg2.connect(database = 'dvdrental', user='postgres', password='nicaragua13')

#cursor (enables traversal over records in a database)
cur = conn.cursor()

#execute query
cur.execute('SELECT * FROM payment')

#one row (tuple)
cur.fetchone()

#n rows (list of tuples)
cur.fetchmany(10)

#all rows
#cur.fetchall()


#close connection
conn.close()



###SQLalchemy

database = 'dvdrental'
user='postgres'
password='nicaragua13'

engine=sqlalchemy.create_engine('postgresql://postgres:nicaragua13@localhost/dvdrental')

q1 = engine.execute('SELECT * FROM customer')

test = pd.DataFrame(q1.fetchall())


#this works great!
test2 = pd.read_sql('SELECT * FROM customer' , engine)

####COMPLETE GUIDE####
import sqlalchemy
import pandas as pd

#make engine (or 'mysql' or 'mysql + pymysql')
engine=sqlalchemy.create_engine('postgresql://postgres:nicaragua13@localhost/dvdrental')

#read
test2 = pd.read_sql('SELECT * FROM customer' , engine)

#write
test2.to_sql('pandas customer', engine)

#thats it :)
