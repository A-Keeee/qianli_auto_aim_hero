#include <iostream>

#include <opencv2/opencv.hpp>
#include "cv_bridge/cv_bridge.h"
#include "image_transport/image_transport.hpp"

#include "rclcpp/rclcpp.hpp"

#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/convert.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "rm_msgs/msg/closed_loop.hpp"

#include "builtin_interfaces/msg/time.hpp"

#include "closed_loop.hpp"

namespace rm_closed_loop
{

class ClosedLoopNode : public rclcpp::Node
{
public:
    ClosedLoopNode(const rclcpp::NodeOptions & options);
    void parameters_init();
    void camera_info_callback(const sensor_msgs::msg::CameraInfo::SharedPtr msg);
    void trajectory_closed_loop_callback(const rm_msgs::msg::ClosedLoop::SharedPtr msg);
    void tracker_image_callback(const sensor_msgs::msg::Image::SharedPtr msg);
    void simulated_projectile_trajectory();
    void draw_armor_on_image(cv::Mat image, const geometry_msgs::msg::PoseStamped & armor_pose, std::string id, int armor_num, double r, double another_r, double dz, double yaw);
    std::vector<cv::Point3d> get_armor_3d_points(cv::Point3d & armor_center, std::vector<cv::Point3d> armor_world_points, double yaw);
    std::vector<cv::Point2d> get_image_points(const std::vector<cv::Point3d> & armor_3d_points);
    cv::Mat rotate(const cv::Mat& vec, const double& angle);

    double PITCH_;

    int64_t init_time_; //ms
    std::shared_ptr<FixedSizeMapQueue<int64_t, cv::Mat>> all_armor_images_;

    cv::Mat camera_matrix_;
    cv::Mat distortion_coefficients_;

    std::vector<cv::Point3d> big_armor_world_points_;
    std::vector<cv::Point3d> small_armor_world_points_;

    image_transport::Publisher closed_loop_result_image_pub_;

    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_sub_;
    rclcpp::Subscription<rm_msgs::msg::ClosedLoop>::SharedPtr trajectory_closed_loop_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr tracker_image_sub_;

    rclcpp::TimerBase::SharedPtr simulated_projectile_trajectory_timer_;

    std::shared_ptr<ClosedLoop> closed_loop_;

    //-----------------------------------------------------------------
    // tf2
    // Subscriber with tf2 message_filter
    std::shared_ptr<tf2_ros::Buffer> tf2_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf2_listener_;
    //-----------------------------------------------------------------
};

} // namespace rm_closed_loop