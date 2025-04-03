#include <iostream>

#include <opencv2/opencv.hpp>

#include <message_filters/subscriber.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/create_timer_ros.h>
#include <tf2_ros/message_filter.h>
#include <tf2_ros/transform_listener.h>

#include "geometry_msgs/msg/pose_stamped.hpp"

#include "fixed_size_map_queue.hpp"

// 2024.10.11初步思考：
// 统一采用odom坐标系进行弹丸坐标的计算 -- 因为odom坐标系是固定的，而其他坐标系随云台而发生变化
// 将每一帧的图像保存下来，用于后续的闭环检测，主要存储时间戳信息，并定义一个初始时间，用于计算时间差，减少时间长度 // 时间ms单位
// 使用自定义map队列，用于存储图像信息，当图像数量超过一定数量时，删除最早的图像信息，将新的图像信息插入队列，保存一定时间内的图像信息，减少内存占用，加快检索
// 主要解算信息：装甲板在图像时间戳下的坐标（目标）| 发射点对于图像时间戳下打击的装甲板的坐标（起点），发射时间，发射速度，发射角度，飞行时间 -- 用于识别的弹丸的模拟弹道解算
// 对于每一帧图片，制作模拟弹丸轨迹的可视化，定时间处理（timer处理）or其他时间处理方法？超过飞行时间的弹丸轨迹不再显示
// 如何闭环？
// 难点：只能获得弹丸在二维图像的像素坐标，而不能获得弹丸在三维空间的坐标，如何将二维坐标转换为三维坐标？ --》关键：弹丸小！！！
// 比对弹丸在二维图像中的坐标和模拟弹道此时弹丸位置坐标的接近程度来判断符合哪一条模拟弹道？ --》 如果能准确匹配的话，就可以再通过模拟弹道和实际弹丸的误差来进行闭环检测

// 参数定义
const double g = 9.81; // 重力加速度, m/s^2
const double rho = 1.225; // 空气密度, kg/m^3
const double Cd = 0.47; // 阻力系数
const double d_small = 42.5 / 1000; // 弹丸直径, m
const double A = CV_PI * pow(d_small / 2, 2); // 横截面积, m^2
const double m_small = 3.2 / 1000; // 物体质量, kg

struct Looper
{
    cv::Mat image_;
    double image_time_;
    double shoot_time_;
    double v0; // 弹丸发射时的加速度
    double theta_; // 弹丸发射时的角度
    double fly_t_; // 弹丸飞行时间
    double accumulated_time_; // 累计时间
    geometry_msgs::msg::PoseStamped odom_projectile_pose_;
    geometry_msgs::msg::PoseStamped odom_armor_pose_;
    std::map<double, cv::Point2d> projectile_image_points_;
};

class ClosedLoop
{
public:
    ClosedLoop();

    void parameters_init();

    void update_tf2_buffer(const std::shared_ptr<tf2_ros::Buffer>& tf2_buffer, const std::shared_ptr<tf2_ros::TransformListener>& tf2_listener);

    void add_projectiles_messages(const double& image_time, const cv::Mat& image);

    void update_projectiles_messages(Looper& looper);

    geometry_msgs::msg::PoseStamped get_projectile_pose(const geometry_msgs::msg::PoseStamped& odom_projectile_pose, const geometry_msgs::msg::PoseStamped& odom_armor_pose, const double& time, const double& v0, const double& theta);

    std::shared_ptr<FixedSizeMapQueue<int64_t, Looper>> all_projectiles_messages_; // 时间顺序存储的弹丸信息Looper

    double dt_; // 时间步长, s

    //-----------------------------------------------------------------
    // tf2
    // Subscriber with tf2 message_filter
    std::shared_ptr<tf2_ros::Buffer> tf2_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf2_listener_;
    //-----------------------------------------------------------------
};