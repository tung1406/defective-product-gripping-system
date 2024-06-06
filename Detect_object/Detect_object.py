import sys
import tkinter as tk
from tkinter import ttk
import cv2
import PIL.Image, PIL.ImageTk
from tkinter import font, Label, Button
import time
from threading import Thread
from multiprocessing import Queue
# from queue import Queue

Main_thread_running = True
Camera_main_loop = True
RESET_BUTTON = False
from deep_sort_tracking_id import track_up
from utils.torch_utils import time_synchronized


class Application(tk.Frame):
    def __init__(self, master, video_source=0):
        super().__init__(master)
        self.q_img_origin = Queue(maxsize= 1)
        self.q1_img_track = Queue(maxsize= 1)
        self.q2_count_error = Queue(maxsize= 1)
        self.q_count_height = None
        # self.q_track_count_tmp= None
        screen_width = master.winfo_screenwidth()
        screen_height = master.winfo_screenheight()
        self.master = master
        self.master.geometry(f"{screen_width}x{screen_height}")  # 1440
        self.master.title("Tkinter with Video Streaming and Capture")
        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.font_setup()
        self.vcap = cv2.VideoCapture(video_source)

        self.width = self.vcap.get(cv2.CAP_PROP_FRAME_WIDTH)

        self.height = self.vcap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        print(self.height)
        self.create_widgets()
        self.create_frame_info(self.master)
        self.delay = 1  # [ms]
        self.q_track_count_tmp = {}
        self.on_button_click()


        # self.update()
        self.Thread_update = Thread(target=self.update, args=())
        self.Thread_update.setDaemon(True)
        self.Thread_update.start()
        # self.update()

    def yolov7_thread(self):

        from deep_sort_tracking_id import detect
        from option_parser import get_parser
        opt = get_parser()

        detect(self.q_img_origin, self.q1_img_track, self.q2_count_error, opt)



    def on_button_click(self):

        self.process = Thread(target=self.yolov7_thread, args=())
        self.process.setDaemon(True)
        self.process.start()



    def create_widgets(self):

        self.Frame_height = None
        # Frame_Camera
        self.frame_cam = tk.LabelFrame(self.master, text='Camera', font=self.font_frame)
        self.frame_cam.place(x=30)
        self.frame_cam.configure(width=self.width + 30, height=self.height + 50)
        self.frame_cam.grid_propagate(0)

        # Frame_Tracking
        self.frame_track = tk.LabelFrame(self.master, text='Tracking', font=self.font_frame)
        self.frame_track.place(x=730)
        self.frame_track.configure(width=self.width + 30, height=self.height + 50)
        self.frame_track.grid_propagate(0)

        # Canvas2_camera
        self.canvas2 = tk.Canvas(self.frame_cam)
        self.canvas2.configure(width=self.width, height=self.height)
        self.canvas2.grid(column=0, row=0, padx=10, pady=10)

        # Canvas_tracking
        self.canvas1 = tk.Canvas(self.frame_track)
        self.canvas1.configure(width=self.width, height=self.height)
        self.canvas1.grid(column=0, row=0, padx=10, pady=10)

        # Scale Pull push
        self.scale_count = tk.LabelFrame(self.master, text='Control', font=self.font_frame)
        self.scale_count.place(x=730 + self.width + 30)
        self.scale_count.configure(width=200, height=self.height + 50)
        self.scale_count.grid_propagate(0)


    def update_Scale(self):
        # label pull track height
        global Camera_main_loop

        L_Track_height = Label(self.scale_count, text="Track", font=self.font_lbl_small, borderwidth=2, relief="solid",
                               fg="purple")
        L_Track_height.place(x=20, y=5)

        L_Count_height = Label(self.scale_count, text="Count", font=self.font_lbl_small, borderwidth=2, relief="solid",
                               fg="green")
        L_Count_height.place(x=80, y=5)


        if self.Frame_height is not None:
            self.pull_track_height = tk.Scale(self.scale_count, from_=0, to=self.Frame_height, bg="purple",
                                              orient=tk.VERTICAL)
            self.pull_track_height.config(length=200)
            self.pull_track_height.place(x=20, y=50)
            self.pull_count_height = tk.Scale(self.scale_count, from_=0, to=self.Frame_height, bg="green",
                                              orient=tk.VERTICAL)
            # self.q_track_count = self.pull_count_height.get()
            self.pull_count_height.config(length=200)

            self.pull_count_height.place(x=80, y=50)
            while (True):

                self.q_track_height = self.pull_track_height.get()

                self.q_count_height = self.pull_count_height.get()

                track_up(self.q_count_height, self.q_track_height, Camera_main_loop, RESET_BUTTON)

    def update(self):
        # Get a frame from the video source
        global t0_gui
        global t1_gui
        global fps_exc_gui

        global timeN
        global timeT

        timeN = time_synchronized()

        datc = 0
        loi1 = 0
        loi2 = 0
        sum = 0
        scale_percent = 100  # percent of original size, 100 for Cam
        # ------------------------------------------------------------------------------------
        t0_gui = time.time()

        data = self.q2_count_error.get()
        frame = self.q1_img_track.get()
        frame2 = self.q_img_origin.get()
        # print("size of with ")
        self.frame_label = frame2

        if self.Frame_height is None:
            self.Frame_height = frame.shape[0]
            self.Thread1 = Thread(target=self.update_Scale, args=())
            self.Thread1.setDaemon(True)
            self.Thread1.start()

        #
        if not data.get("dat chuan"):
            datc = 0
        else:
            datc = data.get("dat chuan")

        if not data.get("loi 1"):
            loi1 = 0
        else:
            loi1 = data.get("loi 1")

        if not data.get("loi 2"):
            loi2 = 0
        else:
            loi2 = data.get("loi 2")

        sum = datc + loi1 + loi2



        # print('........................................................................')


        width = int(self.width * scale_percent / 100)

        height = int(self.height * scale_percent / 100)

        dim = (width, height)
        frameTrack = cv2.resize(frame2, dim, interpolation=cv2.INTER_AREA)

        # # Convert mau BGR -> RGB
        frameTrack = cv2.cvtColor(frameTrack, cv2.COLOR_BGR2RGB)

        # Convert array -> Img -> Img TK
        self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frameTrack))

        # self.photo -> Canvas
        self.canvas2.create_image(0, 0, image=self.photo, anchor=tk.NW)


        frameCam = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
        frameCam = cv2.cvtColor(frameCam, cv2.COLOR_BGR2RGB)
        self.photo2 = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frameCam))
        self.canvas1.create_image(0, 0, image=self.photo2, anchor=tk.NW)

        # # xu ly hien thi count
        self.lbl1_count.configure(text=datc)
        self.lbl2_count.configure(text=loi1)
        self.lbl3_count.configure(text=loi2)
        self.lbl4_sum.configure(text=sum)
        # fram 2 la frame nguyen thuy

        t1_gui = time_synchronized()
        fps_exc_gui = 1 / (t1_gui - t0_gui)

        #print(f'Done per previous frame of GUI.  ({(1E3 * (t1_gui - t0_gui)):.1f}ms) GUI,  FPS = {fps_exc_gui:.0f} (Frame/s) ')
        # print("------------------------------------")

        self.master.after(self.delay, self.update)


    def create_frame_info(self, root):

        self.frame_info = tk.LabelFrame(root, text='Infomation', font=self.font_frame)
        self.frame_info.place(x=30, y=530)
        self.frame_info.configure(width=1370, height=230)
        self.frame_info.grid_propagate(0)

        # Name
        lbl_info = Label(self.frame_info, text="Đếm sản phẩm lỗi", font=self.font_lbl_big)
        lbl_info.place(x=30, y=20)
        lbl_info = Label(self.frame_info, text="cúc áo", font=self.font_lbl_big)
        lbl_info.place(x=135, y=70)

        # Label count
        lbl0 = Label(self.frame_info, text="Số lượng", font=self.font_lbl_big_sl, borderwidth=2, relief="solid")
        lbl0.place(x=600, y=5)

        lbl1 = Label(self.frame_info, text="Đạt chuẩn", font=self.font_lbl_middle)
        lbl1.place(x=600, y=60)
        self.lbl1_count = Label(self.frame_info, text="0", font=self.font_lbl_middle, borderwidth=1, relief="sunken")
        self.lbl1_count.place(x=720, y=60)

        lbl2 = Label(self.frame_info, text="Lỗi 1", font=self.font_lbl_middle)
        lbl2.place(x=600, y=95)
        self.lbl2_count = Label(self.frame_info, text="0", font=self.font_lbl_middle, borderwidth=1, relief="sunken")
        self.lbl2_count.place(x=720, y=95)

        lbl3 = Label(self.frame_info, text="Lỗi 2", font=self.font_lbl_middle)
        lbl3.place(x=600, y=130)
        self.lbl3_count = Label(self.frame_info, text="0", font=self.font_lbl_middle, borderwidth=1, relief="sunken")
        self.lbl3_count.place(x=720, y=130)

        lbl4 = Label(self.frame_info, text="Tổng", font=self.font_lbl_middle)
        lbl4.place(x=600, y=165)
        self.lbl4_sum = Label(self.frame_info, text="0", font=self.font_lbl_middle, borderwidth=1, relief="sunken")
        self.lbl4_sum.place(x=720, y=165)

        # Button control

        self.btn_reset = tk.Button(self.frame_info, text='Reset', font=self.font_btn_big)
        self.btn_reset.configure(command=self.press_reset_btn, bg="red")
        self.btn_reset.place(x=1000, y=5)

        self.btn_exit = tk.Button(self.frame_info, text='Exit', font=self.font_btn_big)
        self.btn_exit.configure(command=self.press_exit_btn, bg="blue")
        self.btn_exit.place(x=1000, y=70)




    def on_button_click_2(self):
        # global Main_thread_running
        # Main_thread_running = False

        from utils.datasets import Camera_main
        # from Label_tools_origin import LabelTool
        global Camera_main_loop
        Camera_main_loop = False
        Camera_main(Camera_main_loop)
        self.vcap.release()
        self.master.destroy()

        # self.master.quit()
        print("OK")
        # self.process.join()
        # from Label_tools import LabelTool
        import time
        # time.sleep(10)

        # self.process.exit()

    # dpress_exit_btn

    def press_reset_btn(self):
        # from demoApp import stop_process
        self.ctr = 1
        print("ctr= ", self.ctr)
        global RESET_BUTTON
        RESET_BUTTON = True
        track_up(self.q_count_height, self.q_track_height, Camera_main_loop, RESET_BUTTON)
        print("RESET_BUTTON = ~RESET_BUTTON", RESET_BUTTON)
        RESET_BUTTON = False
        print("RESET_BUTTON = ~RESET_BUTTON", RESET_BUTTON)

    def press_exit_btn(self):
        # self.master.destroy()

        # Huy cho doi cac thread lien ket voi queue
        self.q_img_origin.cancel_join_thread()
        self.q1_img_track.cancel_join_thread()
        self.q2_count_error.cancel_join_thread()

        # Close queue
        # self.q_img_origin.close()
        # self.q1_img_track.close()
        # self.q2_count_error.close()

        # Giai phong bo nho queue
        del self.q_img_origin
        del self.q1_img_track
        del self.q2_count_error

        # Giai phong thread
        # self.Thread_update = None
        # self.process = None

        self.vcap.release()
        self.master.destroy()
        self.master.quit()
        sys.exit()

    def on_closing(self):
        # Cancel wait thread connect queue
        self.q_img_origin.cancel_join_thread()
        self.q1_img_track.cancel_join_thread()
        self.q2_count_error.cancel_join_thread()

        # Close queue
        # self.q_img_origin.close()
        # self.q1_img_track.close()
        # self.q2_count_error.close()

        # Giai phong bo nho queue
        del self.q_img_origin
        del self.q1_img_track
        del self.q2_count_error

        # Giai phong thread
        # self.Thread_update = None
        # self.process = None

        self.vcap.release()
        self.master.destroy()
        self.master.quit()
        sys.exit()


    def font_setup(self):
        self.font_frame = font.Font(family="Meiryo UI", size=15, weight="normal")
        self.font_btn_big = font.Font(family="Meiryo UI", size=20, weight="bold")
        self.font_btn_small = font.Font(family="Meiryo UI", size=15, weight="bold")

        self.font_lbl_bigger = font.Font(family="Meiryo UI", size=45, weight="bold")
        self.font_lbl_big = font.Font(family="Meiryo UI", size=30, weight="bold")
        self.font_lbl_big_sl = font.Font(family="Meiryo UI", size=20, weight="bold")
        self.font_lbl_middle = font.Font(family="Meiryo UI", size=15, weight="bold")
        self.font_lbl_small = font.Font(family="Meiryo UI", size=12, weight="normal")

# if __name__ == "__main__":
#     root = tk.Tk()
#     app = Application(master=root)  # Inherit
#     #global Main_thread_running
#     root.mainloop()
#     #label_tools()
#     #t._Thread__stop()
#     # if Main_thread_running is False:
#     #     root.destroy()
#     #     #if run_program == "n":
#     #     sys.exit()
#     #     # root.mainloop()

root = tk.Tk()
app = Application(master=root)  # Inherit
# global Main_thread_running
root.mainloop()