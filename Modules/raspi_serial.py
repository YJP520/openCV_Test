########################################################################################################################
#
#   Project:    Raspberry application for the design of graduation.
#   Author :    Yu.J.P
#   Time   :    2024/03/24 -
#
########################################################################################################################

import serial
import time
from plug_in.Colors import Color


########################################################################################################################


class My_Serial:
    def __init__(self):
        """
        @Function: 构造函数 - 初始化配置
        @Author: Yu.J.P at 2024.3.25
        """
        pass
        # self.serial_0 = serial.Serial('/dev/ttyUSB0', 115200)  # 设置端口和波特率
        # self.serial_1 = serial.Serial('/dev/ttyAMA0', 115200)  # 设置端口和波特率

    def receive_test(self, serial_num):
        """
        @Function: 串口接收控制函数
        @Author: Yu.J.P at 2024.3.25
        @Description: 读取串口数据，返回有效串口数据
        :param serial_num: 串口序号，0或者1
        :return: 串口接收数据
        """
        if serial_num == 0:
            serial_in = self.serial_0
        else:  # serial_in == 1
            serial_in = self.serial_1
        while True:
            # 读30位
            receive_data = serial_in.read(30)
            if receive_data == '':
                # 无效数据等待
                time.sleep(0.02)  # 20ms
            else:
                # 有效数据返回
                return receive_data

    def receive_transmit(self):
        """
        @Function: 串口收发测试函数
        @Author: Yu.J.P at 2024.3.25
        :return: None
        """
        # 接收串口数据 使用串口1
        re_data = self.receive_test(1)
        # 串口发送数据
        self.serial_1.write(re_data)

    def instruct_date_capture(self):
        """
        @Function: 前台指令捕获函数
        @Author: Yu.J.P at 2024.3.25
        @Description: 将串口数据检验校正打包并返回
        :return: 校正后的数据
        """
        print(Color.carmine, "\r[INPUT-DATA]:", Color.white, end='')
        instruct = input()
        print("[STATE] - receive_date_capture. Data = ", instruct)
        return instruct

    def instruct_data_translate(self, instruct):
        """
        @Function: 串口数据转译函数
        @Author: Yu.J.P at 2024.3.25
        @Description: 将串口数据转译为对应指令并返回
        :return: robot control instructs
        """
        if instruct == 'go-front':      # 前进
            translate_data = '10001212'
        elif instruct == 'go-back':     # 后退
            translate_data = '10111212'
        elif instruct == 'go-left':     # 左转
            translate_data = '10011212'
        elif instruct == 'go-right':    # 右转
            translate_data = '10101212'
        elif instruct == 'go-stop':     # 停止
            translate_data = '10000000'

        elif instruct == 'hand-reboot':  # 机械臂待机
            translate_data = '01000001'
        elif instruct == 'hand-down':    # 机械臂下降
            translate_data = '01000010'
        elif instruct == 'hand-up':      # 机械臂抬升
            translate_data = '01000011'
        elif instruct == 'hand-close':   # 夹爪合拢
            translate_data = '01000100'
        elif instruct == 'hand-loose':   # 夹爪张开
            translate_data = '01000101'
        elif instruct == 'hand-start':    # 机械臂启动
            translate_data = '01000000'
        else:
            translate_data = '0'
        print("[STATE] - receive_data_translate. translate_data = ", translate_data)
        return translate_data

    def instruct_data_test(self):
        """
        @Function: 串口数据指令解译测试函数
        @Author: Yu.J.P at 2024.3.25
        @Description: 测试串口数据解译的准确性
        :return: None
        """
        print("[STATE] - receive_data_test test.")

        instruct = self.instruct_date_capture()
        translate_data = self.instruct_data_translate(instruct)
        print("[DATA] - translate_data is ", translate_data)
        pass


########################################################################################################################

if __name__ == '__main__':

    my_serial = My_Serial()
    # 指令转译测试
    while True:
        my_serial.instruct_data_test()
        print(Color.carmine, "\r[TEST] - IS CONTINUE ? [y/n]->", Color.white, end='')
        if 'n' == input():
            break

    # while True:
    #     # 串口收发测试
    #     my_serial.receive_transmit()
    #     time.sleep(0.01)  # 10ms










