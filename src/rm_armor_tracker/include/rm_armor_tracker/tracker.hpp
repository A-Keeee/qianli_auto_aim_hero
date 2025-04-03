#include <iostream>

#include <opencv2/opencv.hpp>

#include "angles/angles.h"
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/convert.h>

#include <rclcpp/logger.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include "rm_msgs/msg/armor.hpp"

#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/point_stamped.hpp"
#include "geometry_msgs/msg/quaternion.hpp"

#include "ekf.hpp"
#include "armor_ekf.hpp"

struct TrackerArmor
{
    int armor_id;
    int armor_num;
    int detect_count;
    int lost_count;
    std::string armor_name;
    std::string status;
    geometry_msgs::msg::PoseStamped armor_pose;
};

// armor_id: 0, 1, 2, 3, 4, 5, 6, 7, 8
// status: "LOST", "DETECTING","TRACKING", "LOSTING"

class Tracker
{
public:
    Tracker();
    Tracker(double max_match_distance, double max_match_yaw_diff);
    void tracker_init(const rm_msgs::msg::Armor::SharedPtr armor_msg);
    void EKF_init(const rm_msgs::msg::Armor::SharedPtr &armor_msg);
    void ArmorEKF_init(const rm_msgs::msg::Armor::SharedPtr &armor_msg);
    void tracker_update(const rm_msgs::msg::Armor::SharedPtr armor_msg);
    void handleArmorJump(const TrackerArmor &armor);
    double orientationToYaw(const geometry_msgs::msg::Quaternion & q);
    cv::Point3d get_armor_position(const cv::Mat &state);

    double max_match_distance_;
    double max_match_yaw_diff_;

    int tracking_thres_;
    int lost_thres_;

    double dz_;
    double another_r_;
    double last_yaw_;

    cv::Mat tracking_state_; // 通过EKF预测后的追踪状态
    cv::Mat target_state_; // 通过EKF更新后的目标状态
    
    cv::Mat armor_tracking_state_; // 通过EKF预测后的追踪状态
    cv::Mat armor_target_state_; // 通过EKF更新后的目标状态

    std::shared_ptr<TrackerArmor> tracker_armor_; // 追踪的装甲板信息,包含真实装甲板信息和追踪状态
    std::shared_ptr<ArmorEKF> armor_ekf_;
    std::shared_ptr<EKF> ekf_;
};