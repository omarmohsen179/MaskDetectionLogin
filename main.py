import Mask_detection_class
import tkinter
import mysql.connector
import Login_page

# Create a window and pass it to the Application object
mydb = mysql.connector.connect(
    host="localhost",
    database='pythondatabase',
    user="root",
    password="Medad123456", )
print(mydb.is_connected())


def scann(user):
    Mask_detection_class.App(user,tkinter.Tk(), "Tkinter and OpenCV",mydb)


Login_page.Login(tkinter.Tk(), "Log In", mydb, scann)
