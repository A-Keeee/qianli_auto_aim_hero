#include <iostream>

#include "cv_bridge/cv_bridge.h"

#include "image_transport/image_transport.hpp"

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/header.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "ament_index_cpp/get_package_share_directory.hpp"

#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/convert.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include "angles/angles.h"

#include "builtin_interfaces/msg/time.hpp"

#include "rm_msgs/msg/armor.hpp"
#include "rm_msgs/msg/status.hpp"

#include "openvino_detector.hpp"
#include "pnp_solver.hpp"
#include "lights_detector.hpp"
#include "armor.hpp"
#include "projection_yaw.hpp"

namespace rm_armor_detector
{

class ArmorDetectorNode : public rclcpp::Node
{
public:
    ArmorDetectorNode(const rclcpp::NodeOptions & options);
    void parameters_init();
    std::unique_ptr<LightsDetector> initDetector();
    void camera_info_callback(const sensor_msgs::msg::CameraInfo::SharedPtr msg);
    void status_callback(const rm_msgs::msg::Status::SharedPtr msg);
    void image_callback(const sensor_msgs::msg::Image::SharedPtr msg);
    void test();
    double orientationToYaw(const geometry_msgs::msg::Quaternion & q);

    void robots_init();
    void get_robots(std::vector<DecisionArmor> &decision_armors, const std::vector<Armor> &armors);
    void allrobots_adjust(std::vector<DecisionArmor> &decision_armors);
    bool get_is_big_armor(const std::vector<cv::Point2f> &four_points);
    bool get_is_ignored(const Armor &armor, const std::vector<cv::Point2f> &four_points);
    DecisionArmor decide_armor_shoot(const std::vector<DecisionArmor> &decision_armors);

    bool is_rune_;
    bool is_openvino_;
    int is_debug_;
    int detect_color_;
    double MIN_BIG_ARMOR_RATIO_;

    int camera_width_;
    int camera_height_;
    cv::Mat camera_matrix_;
    cv::Mat distortion_coefficients_;
    cv::Point2f image_center_;

    double last_yaw_;

    std::vector<int> ignore_armors_;
    std::vector<DecisionArmor> decision_armors_;

    cv::Point2i last_decision_armor_image_point_;
    int last_decision_armor_id_;
    bool is_repeat;

    std::shared_ptr<OpenvinoDetector> openvino_detector_;
    std::shared_ptr<LightsDetector> lights_detector_;
    std::shared_ptr<PnpSolver> pnp_solver_;
    std::shared_ptr<ProjectionYaw> projection_yaw_;

    rclcpp::Publisher<rm_msgs::msg::Armor>::SharedPtr armor_pub_;

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_sub_;
    rclcpp::Subscription<rm_msgs::msg::Status>::SharedPtr status_sub_;

    //-----------------------------------------------------------------
    // tf2
    // Subscriber with tf2 message_filter
    std::shared_ptr<tf2_ros::Buffer> tf2_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf2_listener_;
    //-----------------------------------------------------------------

    //-------------------------------------------------------------------------------------------
    // debug
    void create_debug_publishers();
    void destroy_debug_publishers();
    void debug_deal(const cv::Mat &image, const std_msgs::msg::Header& image_header, const std::vector<Armor> &armors, const DecisionArmor &decision_armor);

    std::vector<cv::Point2f> pred_points_;
    
    image_transport::Publisher result_image_pub_;

    std::shared_ptr<rclcpp::ParameterEventHandler> debug_param_sub_;
    std::shared_ptr<rclcpp::ParameterCallbackHandle> debug_cb_handle_;

    //------------------------------------------------------------------------------
    std::chrono::steady_clock::time_point detector_start_;
    std::chrono::steady_clock::time_point detector_end_;
    int detector_fps_;
    int detector_now_fps_;
};

}  // namespace rm_armor_detector