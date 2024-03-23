########################################################################################################################
#
#   Project:    OpenCV application for the design of graduation.
#   Author :    Yu.J.P
#   Time   :    2024/03/19 -
#
########################################################################################################################
import time
from threading import Lock, Thread
from time import sleep

flag = True
lock = Lock()
book_num = 100


def tar():
    global flag, lock, book_num
    while True:
        lock.acquire()
        # 线程任务逻辑
        if flag is False:
            break
        print("book={}, and add 1.".format(book_num))
        book_num += 1
        time.sleep(0.5)
        lock.release()
    lock.release()


if __name__ == "__main__":
    thread = Thread(target=tar)
    thread.start()
    print("\n3秒后线程会被杀死")
    sleep(3)
    flag = False
    print("线程已被杀死")


