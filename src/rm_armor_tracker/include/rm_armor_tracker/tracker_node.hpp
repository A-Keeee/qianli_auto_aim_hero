#include <iostream>

#include "opencv2/opencv.hpp"
#include "cv_bridge/cv_bridge.h"

#include "image_transport/image_transport.hpp"

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/point_stamped.hpp"
#include "geometry_msgs/msg/vector3.hpp"

#include <message_filters/subscriber.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/create_timer_ros.h>
#include <tf2_ros/message_filter.h>
#include <tf2_ros/transform_listener.h>

#include "builtin_interfaces/msg/time.hpp"

#include "rm_msgs/msg/armor.hpp"
#include "rm_msgs/msg/target.hpp"

#include "tracker.hpp"

namespace rm_armor_tracker
{

using tf2_filter = tf2_ros::MessageFilter<rm_msgs::msg::Armor>;

struct DebugParam
{
    int draw_count;
    int image_width;
    int image_height;
    double last_armor_x;
    double last_armor_y;
    double last_armor_x_v;
    double last_armor_y_v;
    double last_car_x;
    double last_car_y;
    double last_car_x_v;
    double last_car_y_v;
    double last_yaw;
    double last_yaw_v;
};

class ArmorTrackerNode : public rclcpp::Node
{
public:
    ArmorTrackerNode(const rclcpp::NodeOptions &options);
    void parameters_init();
    void armorCallback(const rm_msgs::msg::Armor::SharedPtr armor_msg);

    rclcpp::Time last_time_;
    double dt_;
    double lost_time_thres_;

    std::shared_ptr<Tracker> tracker_;

    rclcpp::Publisher<rm_msgs::msg::Target>::SharedPtr target_pub_;

    //-----------------------------------------------------------------
    // time
    std::chrono::steady_clock::time_point tracker_start_;
    std::chrono::steady_clock::time_point tracker_end_;
    int tracker_fps_;
    int tracker_now_fps_;

    //-----------------------------------------------------------------
    // tf2
    // Subscriber with tf2 message_filter
    std::string target_frame_;
    std::shared_ptr<tf2_ros::Buffer> tf2_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf2_listener_;
    message_filters::Subscriber<rm_msgs::msg::Armor> armor_sub_;
    std::shared_ptr<tf2_filter> tf2_filter_;
    //-----------------------------------------------------------------

    //-----------------------------------------------------------------
    // debug
    void create_debug_publishers();
    void destroy_debug_publishers();
    void ResultImageCallback(const sensor_msgs::msg::Image::SharedPtr msg);
    void debug_deal();

    int is_debug_;
    cv::Mat debug_image_;
    std::shared_ptr<DebugParam> debug_param_;

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr detector_result_image_sub_;

    image_transport::Publisher tracker_result_image_pub_;

    std::shared_ptr<rclcpp::ParameterEventHandler> debug_param_sub_;
    std::shared_ptr<rclcpp::ParameterCallbackHandle> debug_cb_handle_;
};

} // namespace rm_armor_tracker