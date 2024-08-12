#!/usr/bin/env python

import rospy
from tkinter import Tk, Label, StringVar
from xela_server_ros.msg import SensorFull  # 修改为你的实际包名和消息类型

# 全局变量
sensor_0_data = "Sensor 0 Data"
sensor_1_data = "Sensor 1 Data"


def callback(msg):
    global sensor_0_data, sensor_1_data

    # 解析传感器0数据
    sensor_0 = msg.sensors[0]
    sensor_0_data = (
            f"Sensor 0 Data:\n"
            f"Time: {sensor_0.time}\n"
            f"Model: {sensor_0.model}\n"
            f"Sensor Position: {sensor_0.sensor_pos}\n"
            f"Taxels:\n" +
            '\n'.join(
                [f"  Taxel {i}: x={taxel.x}, y={taxel.y}, z={taxel.z}" for i, taxel in enumerate(sensor_0.taxels)]) +
            '\nForces:\n' +
            '\n'.join(
                [f"  Force {i}: x={force.x}, y={force.y}, z={force.z}" for i, force in enumerate(sensor_0.forces)])
    )

    # 解析传感器1数据
    sensor_1 = msg.sensors[1]
    sensor_1_data = (
            f"Sensor 1 Data:\n"
            f"Time: {sensor_1.time}\n"
            f"Model: {sensor_1.model}\n"
            f"Sensor Position: {sensor_1.sensor_pos}\n"
            f"Taxels:\n" +
            '\n'.join(
                [f"  Taxel {i}: x={taxel.x}, y={taxel.y}, z={taxel.z}" for i, taxel in enumerate(sensor_1.taxels)]) +
            '\nForces:\n' +
            '\n'.join(
                [f"  Force {i}: x={force.x}, y={force.y}, z={force.z}" for i, force in enumerate(sensor_1.forces)])
    )


def update_labels():
    global sensor_0_data, sensor_1_data
    sensor_0_label.set(sensor_0_data)
    sensor_1_label.set(sensor_1_data)
    root.after(100, update_labels)  # 每100毫秒更新一次


def listener():
    global root, sensor_0_label, sensor_1_label

    rospy.init_node('data_listener', anonymous=True)
    rospy.Subscriber("/xServTopic", SensorFull, callback)

    # 创建 GUI
    root = Tk()
    root.title("Sensor Data Display")

    # 创建传感器数据标签
    sensor_0_label = StringVar()
    sensor_1_label = StringVar()

    label0 = Label(root, textvariable=sensor_0_label, justify="left", padx=10, pady=10)
    label0.pack(fill="both", expand=True)

    label1 = Label(root, textvariable=sensor_1_label, justify="left", padx=10, pady=10)
    label1.pack(fill="both", expand=True)

    update_labels()
    root.mainloop()


if __name__ == '__main__':
    try:
        listener()
    except rospy.ROSInterruptException:
        pass
