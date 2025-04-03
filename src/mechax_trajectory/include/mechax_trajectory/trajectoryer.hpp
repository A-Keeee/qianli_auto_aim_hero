#include <iostream>
#include <cmath>
#include <algorithm>
#include "rclcpp/rclcpp.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include <tf2_ros/transform_broadcaster.h>

#include <geometry_msgs/msg/point_stamped.hpp>

#include <message_filters/subscriber.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/create_timer_ros.h>
#include <tf2_ros/message_filter.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <std_msgs/msg/float64.hpp>

#include "rm_msgs/msg/receive_serial.hpp"
#include "rm_msgs/msg/send_serial.hpp"
#include "rm_msgs/msg/target.hpp"
#include "rm_msgs/msg/bias.hpp"
#include "rm_msgs/msg/closed_loop.hpp"

namespace rm_trajectory
{

using namespace std;
using tf2_filter = tf2_ros::MessageFilter<rm_msgs::msg::Target>;

const float g  = 9.8;
#define ARMOR_NUM_BALANCE 2;
#define ARMOR_NUM_OUTPOST 3;

struct result
{
    float x;
    float y;
    float z;
    float yaw;
    float pitch;
    float distance;
};

class Trajectoryer : public::rclcpp::Node
{
public:
    // hanshu
    Trajectoryer(const rclcpp::NodeOptions & options);

    void parameters_init();

    int no_resistance_model(const float &object_x,const float &object_y,const float &object_z,const float &v0);

    int single_resistance_model_two(const float &object_x,const float &object_y,const float &object_z,const float &v0,const float &randa);

    int two_resistance_model(const float &object_x,const float &object_y,const float &object_z,const float &v0,const float &randa);

    bool is_solvable(const float &object_x,const float &object_y,const float &object_z,const float &v0,float &alpha);

    int solve_trajectory();

    void test();

    void targetCallback(const rm_msgs::msg::Target msg);

    void angle_callback(const rm_msgs::msg::ReceiveSerial msg);

    void power_rune_callback(geometry_msgs::msg::PointStamped msg);

    // parameters
    //------------------
    float v0; // m/s
    float angle_pitch;
    float angle_yaw;
    float distance;
    float fly_t; // m
    //------------------
    float now_pitch;
    float now_yaw;
    //------------------
    //------------------
    int armor_num;
    // 敌方云台中心在我方云台中心坐标系下的坐标
    float yaw;
    float v_yaw;
    float armor_ros_x;
    float car_ros_x;
    float armor_vx;
    float car_vx;
    float armor_ros_y;
    float car_ros_y;
    float armor_vy;
    float car_vy;
    float armor_ros_z;
    float armor_vz;
    float r_1;
    float r_2;
    float dz;
    bool is_tracking;
    bool is_can_hit;
    string id;
    rclcpp::Time armor_time;
    //------------------
    int latency_count;
    float all_latency;
    //------------------
    float motor_speed;
    float motor_bias_time;
    float serial_bias_time;
    float latency_bias_time;
    rm_msgs::msg::Bias bias_time_msg;
    //------------------
    rm_msgs::msg::ClosedLoop closed_loop_msg_;
    //------------------
    float randa;
    bool is_hero;
    double max_yaw_diff;
    bool is_repeat;
    //------------------
    // Subsciption
    //------------------
    rclcpp::Subscription<rm_msgs::msg::ReceiveSerial>::SharedPtr angle_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PointStamped>::SharedPtr power_rune_sub_;
    //------------------
    // Publisher
    //------------------
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr maker_pub_;
    rclcpp::Publisher<rm_msgs::msg::Bias>::SharedPtr bias_time_pub_;
    rclcpp::Publisher<rm_msgs::msg::SendSerial>::SharedPtr result_pub_;
    rclcpp::Publisher<rm_msgs::msg::ClosedLoop>::SharedPtr closed_loop_pub_;
    //------------------
    //timer
    //------------------
    rclcpp::TimerBase::SharedPtr timer_;
    //------------------
    //tf2
    // Subscriber with tf2 message_filter
    std::string target_frame_;
    std::shared_ptr<tf2_ros::Buffer> tf2_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf2_listener_;
    message_filters::Subscriber<rm_msgs::msg::Target> target_sub_;
    std::shared_ptr<tf2_filter> tf2_filter_;
};

} // namespace rm_trajectory