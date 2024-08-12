# import rospy
# from xela_server_ros.msg import SensStream
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.colors import Normalize
#
#
# # Callback function to process received messages
# def callback(data):
#     # Extract force data from the message
#     forces = data.sensors[0].forces  # Assuming you want to visualize forces from the first sensor
#     print("...../n")
#     # Convert forces to a numpy array
#     forces_array = np.array([f.z for f in forces]).reshape((4, 4))
#     print(forces_array)
#
#     # # Plotting
#     plt.imshow(forces_array, cmap='viridis', norm=Normalize())
#     plt.colorbar(label='Force (z-axis)')
#     plt.title('Combined Forces (4x4 Grid)')
#     plt.show()
#     plt.pause(1)
#     plt.close('all')  # Close all open figures
# def listener():
#     # Initialize the ROS node
#     rospy.init_node('force_visualizer', anonymous=True)
#
#     # Subscribe to the /xServTopic with SensStream message type
#     rospy.Subscriber('/xServTopic', SensStream, callback)
#
#     # Keep the node running
#     rospy.spin()
#
#
# if __name__ == '__main__':
#     listener()
#


# # !/usr/bin/env python
#
# import rospy
# from xela_server_ros.msg import SensStream
# import numpy as np
# import cv2
# import matplotlib.pyplot as plt
# import matplotlib.colors as mcolors
#
# # Data placeholder
# forces_array = np.zeros((4, 4), dtype=np.float32)
#
#
# def callback(data):
#     global forces_array
#     # Extract force data from the message
#     forces = data.sensors[0].forces  # Assuming you want to visualize forces from the first sensor
#
#     # Convert forces to a numpy array
#     forces_array = np.array([f.z for f in forces]).reshape((4, 4))
#
#     # Normalize to [0, 1] for colormap
#     normalized_array = cv2.normalize(forces_array, None, 0, 1, cv2.NORM_MINMAX)
#     image = np.uint8(normalized_array * 255)  # Scale to [0, 255]
#
#     # Create a custom colormap
#     cmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap', [(0, 'green'), (1, 'red')])
#     color_image = plt.cm.ScalarMappable(cmap=cmap).to_rgba(image, bytes=True)
#     color_image = color_image[:, :, :3]  # Remove alpha channel
#
#     # Resize the image for better visibility
#     scale_factor = 200  # Scale factor to increase the image size
#     resized_image = cv2.resize(color_image, (color_image.shape[1] * scale_factor, color_image.shape[0] * scale_factor),
#                                interpolation=cv2.INTER_NEAREST)
#
#     # Display the image using OpenCV
#     cv2.imshow('Force Array', resized_image)
#     cv2.waitKey(1)  # Wait for 1 ms for the window to update
#
#
# def listener():
#     # Initialize the ROS node
#     rospy.init_node('force_visualizer', anonymous=True)
#
#     # Subscribe to the /xServTopic with SensStream message type
#     rospy.Subscriber('/xServTopic', SensStream, callback)
#
#     # Keep the node running
#     rospy.spin()
#
#     # Clean up on exit
#     cv2.destroyAllWindows()
#
#
# if __name__ == '__main__':
#     listener()

# # !/usr/bin/env python
#
# import rospy
# from xela_server_ros.msg import SensStream
# import numpy as np
# import cv2
# import matplotlib.pyplot as plt
# import matplotlib.colors as mcolors
#
# # Data placeholder
# forces_array = np.zeros((4, 4), dtype=np.float32)
#
#
# def callback(data):
#     global forces_array
#     # Extract force data from the message
#     forces = data.sensors[0].forces  # Assuming you want to visualize forces from the first sensor
#
#     # Convert forces to a numpy array
#     forces_array = np.array([f.z for f in forces]).reshape((4, 4))
#     print(forces_array)
#     print("                                   /t")
#     # Set negative values to zero
#     forces_array = np.clip(forces_array, 0, None)  # Set all negative values to zero
#
#     # Normalize to [0, 1] by dividing by 5
#     normalized_array = np.clip(forces_array / 5.0, 0, 1)  # Ensure values are between 0 and 1
#     # Create a custom colormap
#     cmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap', ['green', 'red'], N=256)
#
#     # Apply the colormap to the image
#     color_image = cmap(normalized_array)  # Apply colormap to the normalized array
#     color_image = (color_image[:, :, :3] * 255).astype(np.uint8)  # Convert to 8-bit image
#
#     # Resize the image for better visibility
#     scale_factor = 50  # Scale factor to increase the image size
#     resized_image = cv2.resize(color_image, (color_image.shape[1] * scale_factor, color_image.shape[0] * scale_factor),
#                                interpolation=cv2.INTER_NEAREST)
#
#     # Display the image using OpenCV
#     cv2.imshow('Force Array', resized_image)
#     cv2.waitKey(1)  # Wait for 1 ms for the window to update
#
#
# def listener():
#     # Initialize the ROS node
#     rospy.init_node('force_visualizer', anonymous=True)
#
#     # Subscribe to the /xServTopic with SensStream message type
#     rospy.Subscriber('/xServTopic', SensStream, callback)
#
#     # Keep the node running
#     rospy.spin()
#
#     # Clean up on exit
#     cv2.destroyAllWindows()
#
#
# if __name__ == '__main__':
#     listener()


# # !/usr/bin/env python
#
# import rospy
# from xela_server_ros.msg import SensStream
# import numpy as np
# import cv2
# import matplotlib.colors as mcolors
#
# # Data placeholders
# forces_array_1 = np.zeros((4, 4), dtype=np.float32)
# forces_array_2 = np.zeros((4, 4), dtype=np.float32)
#
#
# def callback(data):
#     global forces_array_1, forces_array_2
#
#     # Extract force data from the message
#     forces_1 = data.sensors[0].forces  # Data from the first sensor
#     forces_2 = data.sensors[1].forces  # Data from the second sensor
#
#     # Convert forces to numpy arrays
#     forces_array_1 = np.array([f.z for f in forces_1]).reshape((4, 4))
#     forces_array_2 = np.array([f.z for f in forces_2]).reshape((4, 4))
#
#     # Set negative values to zero
#     forces_array_1 = np.clip(forces_array_1, 0, None)
#     forces_array_2 = np.clip(forces_array_2, 0, None)
#
#     # Normalize to [0, 1] by dividing by 5
#     normalized_array_1 = np.clip(forces_array_1 / 5.0, 0, 1)
#     normalized_array_2 = np.clip(forces_array_2 / 5.0, 0, 1)
#
#     # Create a custom colormap
#     cmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap', ['green', 'red'], N=256)
#
#     # Apply the colormap to the images
#     color_image_1 = cmap(normalized_array_1)
#     color_image_1 = (color_image_1[:, :, :3] * 255).astype(np.uint8)
#
#     color_image_2 = cmap(normalized_array_2)
#     color_image_2 = (color_image_2[:, :, :3] * 255).astype(np.uint8)
#
#     # Resize the images for better visibility
#     scale_factor = 200  # Scale factor to increase the image size
#     resized_image_1 = cv2.resize(color_image_1,
#                                  (color_image_1.shape[1] * scale_factor, color_image_1.shape[0] * scale_factor),
#                                  interpolation=cv2.INTER_NEAREST)
#     resized_image_2 = cv2.resize(color_image_2,
#                                  (color_image_2.shape[1] * scale_factor, color_image_2.shape[0] * scale_factor),
#                                  interpolation=cv2.INTER_NEAREST)
#
#     # Display the images using OpenCV
#     cv2.imshow('Force Array Sensor 1', resized_image_1)
#     cv2.imshow('Force Array Sensor 2', resized_image_2)
#     cv2.waitKey(1)  # Wait for 1 ms for the windows to update
#
#
# def listener():
#     # Initialize the ROS node
#     rospy.init_node('force_visualizer', anonymous=True)
#
#     # Subscribe to the /xServTopic with SensStream message type
#     rospy.Subscriber('/xServTopic', SensStream, callback)
#
#     # Keep the node running
#     rospy.spin()
#
#     # Clean up on exit
#     cv2.destroyAllWindows()
#
#
# if __name__ == '__main__':
#     listener()

# 可以实现接触中心的提取
import rospy
from xela_server_ros.msg import SensStream
import numpy as np
import cv2
from scipy.optimize import least_squares

import matplotlib.colors as mcolors
from sklearn.decomposition import PCA
# Data placeholders
from scipy.optimize import minimize

forces_array_1 = np.zeros((4, 4), dtype=np.float32)
forces_array_2 = np.zeros((4, 4), dtype=np.float32)

# Define taxel coordinates for a 4x4 grid
taxels_array_x = np.array([x for y in range(4) for x in range(4)])
taxels_array_x = taxels_array_x.reshape((4, 4))  # Reshape to (4, 4, 2)

taxels_array_y = np.array([y for y in range(4) for x in range(4)])
taxels_array_y = taxels_array_y.reshape((4, 4))  # Reshape to (4, 4, 2)
# print("111")
def compute_contact_center(forces_array, taxels_array_x,taxels_array_y):
    # Set negative values to zero
    forces_array = np.clip(forces_array, 0, None)

    # Compute total force
    total_force = np.sum(forces_array)

    # Compute contact position (center of pressure)
    if total_force == 0:
        contact_center = (0, 0)  # Avoid division by zero
    else:
        # Reshape forces_array for broadcasting
        forces_array_reshaped = forces_array.reshape((4, 4))

        # Compute the weighted sum
        weighted_sum_x = np.sum(forces_array_reshaped * taxels_array_x)
        contact_center_x = weighted_sum_x / total_force
        weighted_sum_y = np.sum(forces_array_reshaped * taxels_array_y)
        contact_center_y = weighted_sum_y / total_force
        contact_center = np.array([contact_center_x+0.5,contact_center_y+0.5])
    return contact_center
taxels_array_y = np.array([[x for x in range(4)] for _ in range(4)])
taxels_array_x = np.array([[y for x in range(4)] for y in range(4)])
# 计算的是最小二乘
# def weighted_line_fitting(cx, cy, f, taxels_x, taxels_y):
#     # Flatten arrays
#     taxels_x_flat = taxels_x.flatten()
#     taxels_y_flat = taxels_y.flatten()
#     forces_flat = f.flatten()
#
#     # Function to minimize
#     def residuals(params):
#         m, b = params
#         predicted_y = m * taxels_x_flat + b
#         distances = np.abs(predicted_y - taxels_y_flat)
#         weighted_residuals = distances * forces_flat  # Weight residuals by force
#         return weighted_residuals
#
#     # Initial guess for slope (m) and intercept (b)
#     initial_guess = [0, 0]
#
#     # Perform least squares fitting
#     result = least_squares(residuals, initial_guess)
#     m, b = result.x
#
#     return np.arctan(m)
def weighted_line_fitting(f,taxels_x, taxels_y):
    # Flatten arrays
    taxels_x_flat = taxels_x.flatten()
    taxels_y_flat = taxels_y.flatten()
    forces_flat = f.flatten()

    # Function to minimize for vertical lines

    # Function to minimize for non-vertical lines 用一般直线方程，通过限制迭代次数保证实时性的角度获取
    def non_vertical_line_residuals(params):
        theta, r = params
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        # 使用 NumPy 的向量化操作来计算距离
        distances = np.abs(taxels_x_flat * cos_theta + taxels_y_flat * sin_theta - r)
        weighted_residuals = distances * forces_flat
        return weighted_residuals

    # def weighted_residuals(params):
    #     m, b = params
    #     # 计算直线方程的点斜式
    #     predicted_y = m * taxels_x_flat + b
    #     distances = np.abs(predicted_y - taxels_y_flat)
    #     weighted_residuals = distances * forces_flat
    #     return weighted_residuals

    # Initial guess for slope (m) and intercept (b)
    initial_guess = [0, 0]
    result = least_squares(non_vertical_line_residuals, initial_guess,max_nfev=9)
    # result1 = least_squares(weighted_residuals, initial_guess)

    print("theta:",result.nfev)
    # print("m:",result1.nfev)
    theta, r = result.x
    # angle = np.arctan(theta)
    # print(theta)
    return theta


def draw_line_through_center(image, center, angle_radians, color=(0, 255, 0), thickness=0.1):
    """
    Draws a line on the image through the given center with a specified angle.

    Parameters:
    - image: The image on which to draw the line (numpy array).
    - center: Tuple of (cx, cy) specifying the center point.
    - angle_radians: The angle of the line in radians.
    - color: The color of the line (BGR format).
    - thickness: The thickness of the line.
    """
    # Get image dimensions
    height, width = image.shape[:2]

    # Convert angle to direction vector
    dx = np.cos(angle_radians)
    dy = np.sin(angle_radians)

    # Define line endpoints
    length = max(height, width)  # Length of the line

    # Convert center to integer coordinates
    cx, cy = int(center[0]), int(center[1])

    # Start and end points of the line
    x1 = int(cx - length * dx)
    y1 = int(cy - length * dy)
    x2 = int(cx + length * dx)
    y2 = int(cy + length * dy)

    # Draw the line
    cv2.line(image, (y1, x1), (y2, x2), color, thickness)
    return image
def callback(data):
    global forces_array_1, forces_array_2

    # Extract force data from the message
    forces_1 = data.sensors[0].forces  # Data from the first sensor
    forces_2 = data.sensors[1].forces  # Data from the second sensor
    # Convert forces to numpy arrays
    forces_array_1 = np.array([f.z for f in forces_1]).reshape((4, 4))
    forces_array_2 = np.array([f.z for f in forces_2]).reshape((4, 4))

    # Compute contact centers
    contact_center_1 = compute_contact_center(forces_array_1, taxels_array_x,taxels_array_y)
    contact_center_2 = compute_contact_center(forces_array_2, taxels_array_x,taxels_array_y)
    # contact_rad_1 = weighted_line_fitting(contact_center_1[0], contact_center_1[1], forces_array_1, taxels_array_x, taxels_array_y)
    # contact_rad_2 = weighted_line_fitting(contact_center_2[0], contact_center_2[1], forces_array_2, taxels_array_x, taxels_array_y)
    contact_rad_1 = weighted_line_fitting(forces_array_1, taxels_array_x, taxels_array_y)+np.pi/2
    contact_rad_2 = weighted_line_fitting(forces_array_2, taxels_array_x, taxels_array_y)+np.pi/2
    # print(contact_rad_1/3.14*180)
    # Normalize to [0, 1] by dividing by 5
    normalized_array_1 = np.clip(forces_array_1 / 5.0, 0, 1)
    normalized_array_2 = np.clip(forces_array_2 / 5.0, 0, 1)

    gray_image_1 = (255-normalized_array_1[:, :] * 255).astype(np.uint8)

    gray_image_2 = (255-normalized_array_2[:, :] * 255).astype(np.uint8)

    # Resize the images for better visibility
    scale_factor = 200  # Scale factor to increase the image size
    resized_image_1 = cv2.resize(gray_image_1,
                                 (gray_image_1.shape[1] * scale_factor, gray_image_1.shape[0] * scale_factor),
                                 interpolation=cv2.INTER_NEAREST)
    resized_image_2 = cv2.resize(gray_image_2,
                                 (gray_image_2.shape[1] * scale_factor, gray_image_2.shape[0] * scale_factor),
                                 interpolation=cv2.INTER_NEAREST)

    # Draw contact centers
    cv2.circle(resized_image_1, (int(contact_center_1[1] * scale_factor), int(contact_center_1[0] * scale_factor)), 10,
               128, -1)
    cv2.circle(resized_image_2, (int(contact_center_2[1] * scale_factor), int(contact_center_2[0] * scale_factor)), 10,
               128, -1)
    resized_image_1 = draw_line_through_center(resized_image_1, contact_center_1* scale_factor, contact_rad_1, 128,
                                            3)
    resized_image_2 = draw_line_through_center(resized_image_2, contact_center_2* scale_factor, contact_rad_2, 128,
                                            3)
    # Display the images using OpenCV
    cv2.imshow('Force Array Sensor 1', resized_image_1)
    cv2.imshow('Force Array Sensor 2', resized_image_2)
    cv2.waitKey(1)  # Wait for 1 ms for the windows to update


def listener():
    # Initialize the ROS node
    rospy.init_node('force_visualizer', anonymous=True)

    # Subscribe to the /xServTopic with SensStream message type
    rospy.Subscriber('/xServTopic', SensStream, callback)

    # Keep the node running
    rospy.spin()

    # Clean up on exit
    cv2.destroyAllWindows()


if __name__ == '__main__':
    listener()
