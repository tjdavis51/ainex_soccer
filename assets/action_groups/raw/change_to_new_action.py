#!/usr/bin/env python3
# encoding: utf-8
# 2022/7/20 by Aiden
# 新的偏差5号和6号与原来差40，需要转换
import os
import sqlite3
from PyQt5.QtSql import QSqlDatabase, QSqlQuery

servo_num = 22

old_path = '/home/ubuntu/software/ainex_controller/test/new/'
new_path = old_path + 'new'

def servo_id_change(act_old, act_new):
    conn = sqlite3.connect(act_new)

    c = conn.cursor()                    
    c.execute('''CREATE TABLE ActionGroup([Index] INTEGER PRIMARY KEY AUTOINCREMENT
    NOT NULL ON CONFLICT FAIL
    UNIQUE ON CONFLICT ABORT,
    Time INT,
    Servo1 INT,
    Servo2 INT,
    Servo3 INT,
    Servo4 INT,
    Servo5 INT,
    Servo6 INT,
    Servo7 INT,
    Servo8 INT,
    Servo9 INT,
    Servo10 INT,
    Servo11 INT,
    Servo12 INT,
    Servo13 INT,
    Servo14 INT,
    Servo15 INT,
    Servo16 INT,
    Servo17 INT,
    Servo18 INT,
    Servo19 INT,
    Servo20 INT,
    Servo21 INT,
    Servo22 INT);''')     

    rbt = QSqlDatabase.addDatabase("QSQLITE")
    rbt.setDatabaseName(act_old)
    if rbt.open():
        actgrp = QSqlQuery()
        if (actgrp.exec("select * from ActionGroup ")):
            while (actgrp.next()):
                insert_sql = "INSERT INTO ActionGroup(Time,Servo1,Servo2,Servo3,Servo4,Servo5,Servo6,Servo7,Servo8,Servo9,Servo10,Servo11,Servo12,Servo13,Servo14,Servo15,Servo16,Servo17,Servo18,Servo19,Servo20,Servo21,Servo22) VALUES("
                action_data = []
                for i in range(1, servo_num + 2):
                    action_data.append(str(actgrp.value(i)))
                insert_sql += action_data[0] + ','
                for j in range(1, servo_num + 1):
                    servo_value = int(action_data[j])
                    if j == 5:
                        servo_value += 40
                    elif j == 6:
                        servo_value -= 40
                    if j == servo_num:
                        insert_sql += str(servo_value)
                    else:
                        insert_sql += str(servo_value) + ','
                insert_sql += ");"
                c.execute(insert_sql)
            conn.commit()
            conn.close()
    rbt.close()

if __name__ == '__main__':
    d6a_list = []
    for f in os.listdir(old_path):
        if os.path.splitext(f)[1] == '.d6a':
            d6a_list.append(f)
    if not os.path.exists(new_path):
        os.system('mkdir ' + new_path)
    for f in d6a_list:
        old = os.path.join(old_path, f)
        new = os.path.join(new_path, f)
        servo_id_change(old, new)
        print('{}---->{}'.format(old, new))
