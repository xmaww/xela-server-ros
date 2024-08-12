#!/usr/bin/env python

import rospy
from xela_server_ros.msg import SensStream  # 导入正确的消息类型

# 定义一个全局变量用于存储接收到的数据
data = {}


# 话题回调函数
def callback(msg):
    global data_left,data_right
    try:
        # 解析消息
        data_left = msg.sensors[0]
        data_right = msg.sensors[1]
        rospy.loginfo("Received data: %s", data_right)
    except Exception as e:
        rospy.logerr("Error processing message: %s", e)


def listener():
    # 初始化ROS节点
    rospy.init_node('data_listener', anonymous=True)

    # 订阅话题
    rospy.Subscriber("/xServTopic", SensStream, callback)  # 使用正确的话题名称和消息类型

    # 保持节点运行直到被中断
    rospy.spin()


if __name__ == '__main__':
    try:
        listener()
    except rospy.ROSInterruptException:
        pass
