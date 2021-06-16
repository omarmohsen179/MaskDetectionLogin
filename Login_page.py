from tkinter import *
import easygui
import Mask_detection_class


class Login:
    def __init__(self, window, window_title, db, second):
        self.db = db
        self.login_screen = window
        self.second = second
        self.login_screen.title(window_title)
        self.login_screen.geometry("420x380")
        self.login_screen.resizable(width=False, height=False)
        Label(self.login_screen, text="Please enter login details").pack()
        Label(self.login_screen, text="").pack()
        Label(self.login_screen, text="Username").pack()
        self.username_login_entry = Entry(self.login_screen, textvariable="username")
        self.username_login_entry.pack()
        Label(self.login_screen, text="").pack()
        Label(self.login_screen, text="Password").pack()
        self.password__login_entry = Entry(self.login_screen, textvariable="password", show='*')
        self.password__login_entry.pack()
        Label(self.login_screen, text="").pack()
        Button(self.login_screen, text="Login", width=10, height=1, command=self.login).pack()

        self.login_screen.mainloop()

    def login(self):


        if self.password__login_entry.get() and self.username_login_entry.get():
            mycursor = self.db.cursor()
            mycursor.execute("SELECT * FROM users where username = '"
                             + self.username_login_entry.get() +
                             "' and password = '" +
                             self.password__login_entry.get()+"';")
            myresult = mycursor.fetchall()

            if myresult:
                self.login_screen.destroy()
                easygui.msgbox("take your test", title="Mask Test")

                self.second(myresult[0][0])
            else:
                print(myresult)
        else:
            easygui.msgbox("you must login", title="simple gui")
