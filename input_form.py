# import tkinter as tk
# from tkinter import messagebox

# def get_states():
#     """باز کردن فرم برای گرفتن initial و goal state"""
#     def submit():
#         nonlocal user_input
#         init = entry_initial.get().strip().upper()
#         goal = entry_goal.get().strip().upper()
#         if not init or not goal:
#             messagebox.showwarning("Warning", "هر دو فیلد را پر کنید.")
#             return
#         user_input = (init, goal)
#         root.destroy()

#     user_input = None
#     root = tk.Tk()
#     root.attributes("-fullscreen", True)
#     root.title("Search Configuration")
#     root.geometry("300x200")
#     root.resizable(False, False)

#     # استایل و متن‌ها
#     tk.Label(root, text="Initial State:", font=("Segoe UI", 11)).pack(pady=5)
#     entry_initial = tk.Entry(root, font=("Segoe UI", 11))
#     entry_initial.pack(pady=5)
    
#     tk.Label(root, text="Goal State:", font=("Segoe UI", 11)).pack(pady=5)
#     entry_goal = tk.Entry(root, font=("Segoe UI", 11))
#     entry_goal.pack(pady=5)
    
#     tk.Button(
#         root, text="Start Search", font=("Segoe UI", 11, "bold"),
#         bg="#4CAF50", fg="white", width=15, command=submit
#     ).pack(pady=15)
    
#     root.mainloop()
#     return user_input

##-------------------------------------------------------------------------------
import tkinter as tk
from tkinter import messagebox

def get_states():
    """باز کردن فرم برای گرفتن initial و goal state"""
    def submit():
        nonlocal user_input
        init = entry_initial.get().strip().upper()
        goal = entry_goal.get().strip().upper()
        if not init or not goal:
            messagebox.showwarning("Warning", "هر دو فیلد را پر کنید.")
            return
        user_input = (init, goal)
        root.destroy()

    user_input = None
    root = tk.Tk()
    root.title("Search Configuration")

    # ✅ فول‌اسکرین
    root.attributes("-fullscreen", True)
    root.bind("<Escape>", lambda e: root.attributes("-fullscreen", False))

    # ✅ پس‌زمینه کاملاً مشکی
    root.configure(bg="black")

    # ✅ فریم مرکزی
    frame = tk.Frame(root, bg="black", padx=40, pady=40)
    frame.place(relx=0.5, rely=0.5, anchor="center")

    # عنوان
    tk.Label(
        frame, text="Search Configuration", font=("Segoe UI", 22, "bold"),
        bg="black", fg="white"
    ).pack(pady=(0, 30))

    # ورودی initial
    tk.Label(
        frame, text="Initial State:", font=("Segoe UI", 16),
        bg="black", fg="white"
    ).pack(pady=10)
    entry_initial = tk.Entry(frame, font=("Segoe UI", 16), width=25, justify="center")
    entry_initial.pack(pady=5)

    # ورودی goal
    tk.Label(
        frame, text="Goal State:", font=("Segoe UI", 16),
        bg="black", fg="white"
    ).pack(pady=10)
    entry_goal = tk.Entry(frame, font=("Segoe UI", 16), width=25, justify="center")
    entry_goal.pack(pady=5)

    # دکمه شروع
    tk.Button(
        frame, text="Start Search", font=("Segoe UI", 15, "bold"),
        bg="#4CAF50", fg="white", activebackground="#45a049",
        relief="flat", width=20, height=2, command=submit
    ).pack(pady=(30, 10))

    root.mainloop()
    return user_input
