#include "tracker.hpp"

Tracker::Tracker()
{

}

Tracker::Tracker(double max_match_distance, double max_match_yaw_diff)
{
    std::cout << "Tracker constructed" << std::endl;
    this->max_match_distance_ = max_match_distance;
    this->max_match_yaw_diff_ = max_match_yaw_diff;
    this->tracking_thres_ = 0;
    this->lost_thres_ = 0;
    this->dz_ = 0.0;
    this->another_r_ = 0.0;
    this->last_yaw_ = 0.0;
    this->tracking_state_ = cv::Mat::zeros(9, 1, CV_64F);
    this->target_state_ = cv::Mat::zeros(9, 1, CV_64F);
    this->armor_tracking_state_ = cv::Mat::zeros(4, 1, CV_64F);
    this->armor_target_state_ = cv::Mat::zeros(4, 1, CV_64F);
    tracker_armor_ = std::make_shared<TrackerArmor>();
    tracker_armor_->armor_id = -1;
    tracker_armor_->armor_num = 0;
    tracker_armor_->detect_count = 0;
    tracker_armor_->lost_count = 0;
    tracker_armor_->armor_name = "";
    tracker_armor_->status = "LOST";
    
}

void Tracker::tracker_init(const rm_msgs::msg::Armor::SharedPtr armor_msg)
{
    tracker_armor_->armor_id = armor_msg->id;
    tracker_armor_->detect_count = 0;
    tracker_armor_->lost_count = 0;
    if(armor_msg->id == 0)
    {
        tracker_armor_->armor_num = 4;
        tracker_armor_->armor_name = "sentry-4";
    }
    else if(armor_msg->id >= 1 && armor_msg->id <= 5)
    {
        if(armor_msg->type == "BIG" && armor_msg->id == 1)
        {
            tracker_armor_->armor_num = 4;
            tracker_armor_->armor_name = "robot" + std::to_string(armor_msg->id) + "-4";
        }
        else if(armor_msg->type == "SMALL")
        {
            tracker_armor_->armor_num = 4;
            tracker_armor_->armor_name = "robot" + std::to_string(armor_msg->id) + "-4";
        }
    }
    else if(armor_msg->id == 6)
    {
        tracker_armor_->armor_num = 3;
        tracker_armor_->armor_name = "outpost-3";
    }
    else if(armor_msg->id == 7)
    {
        tracker_armor_->armor_num = 1;
        tracker_armor_->armor_name = "base-1";
    }
    else if(armor_msg->id == 8)
    {
        tracker_armor_->armor_num = 1;
        tracker_armor_->armor_name = "base_big-1";
    }
    tracker_armor_->status = "DETECTING";
    tracker_armor_->armor_pose.pose.position = armor_msg->pose.position;
    tracker_armor_->armor_pose.pose.orientation = armor_msg->pose.orientation;

    EKF_init(armor_msg);
    ArmorEKF_init(armor_msg);
    std::cout << "Tracker initialized!" << std::endl;
}

void Tracker::EKF_init(const rm_msgs::msg::Armor::SharedPtr &armor_msg)
{
    double xa = armor_msg->pose.position.x;
    double ya = armor_msg->pose.position.y;
    double za = armor_msg->pose.position.z;
    last_yaw_ = 0;
    double yaw = orientationToYaw(armor_msg->pose.orientation);
    
    cv::Mat state = (cv::Mat_<double>(9,1) << 0, 0, 0, 0, 0, 0, 0, 0, 0);
    double r = 0.26;
    double xc = xa + r * sin(yaw);
    double yc = ya - r * cos(yaw);
    dz_ = 0.0;
    another_r_ = r;
    state.at<double>(0,0) = xc;
    state.at<double>(2,0) = yc;
    state.at<double>(4,0) = za;
    state.at<double>(6,0) = yaw;
    state.at<double>(8,0) = r;

    ekf_->set_state(state);
}

void Tracker::ArmorEKF_init(const rm_msgs::msg::Armor::SharedPtr &armor_msg)
{
    double xa = armor_msg->pose.position.x;
    double ya = armor_msg->pose.position.y;

    cv::Mat state = (cv::Mat_<double>(4,1) << 0, 0, 0, 0);

    state.at<double>(0,0) = xa;
    state.at<double>(2,0) = ya;

    armor_ekf_->set_state(state);
}

void Tracker::tracker_update(const rm_msgs::msg::Armor::SharedPtr armor_msg)
{
    ekf_->EKF_predict();
    tracking_state_ = ekf_->state_;

    armor_ekf_->EKF_predict();
    armor_tracking_state_ = armor_ekf_->state_;

    bool is_matched = false;

    if(armor_msg->color != 2)
    {
        // Find the closest armor with the same id
        rm_msgs::msg::Armor same_id_armor;
        cv::Point3d same_id_armor_position;
        same_id_armor_position = get_armor_position(tracking_state_);
        double position_diff = cv::norm(same_id_armor_position - cv::Point3d(armor_msg->pose.position.x, armor_msg->pose.position.y, armor_msg->pose.position.z));
        double yaw_diff = abs(orientationToYaw(armor_msg->pose.orientation) - tracking_state_.at<double>(6,0));
        if(position_diff < max_match_distance_ && yaw_diff < max_match_yaw_diff_)
        {
            is_matched = true;
            tracker_armor_->armor_pose.pose.position = armor_msg->pose.position;
            tracker_armor_->armor_pose.pose.orientation = armor_msg->pose.orientation;
            auto armor_pose_position = tracker_armor_->armor_pose.pose.position;
            auto armor_pose_orientation_yaw = orientationToYaw(tracker_armor_->armor_pose.pose.orientation);
            cv::Mat Z = (cv::Mat_<double>(4,1) << armor_pose_position.x, armor_pose_position.y, armor_pose_position.z, armor_pose_orientation_yaw);
            ekf_->EKF_update(Z);
            target_state_ = ekf_->state_;
            /*
            位置信息：
            target_state(0): X 坐标位置
            target_state(1): X 方向速度
            target_state(2): Y 坐标位置
            target_state(3): Y 方向速度
            target_state(4): Z 坐标位置
            target_state(5): Z 方向速度
            偏航角和角速度：
            target_state(6): 目标的偏航角（yaw）
            target_state(7): 目标的角速度（yaw）
            半径信息：
            target_state(8): 目标的半径
            */

            cv::Mat Z_armor = (cv::Mat_<double>(2,1) << armor_pose_position.x, armor_pose_position.y);
            armor_ekf_->EKF_update(Z_armor);
            armor_target_state_ = armor_ekf_->state_;
            /*
            位置信息：
            armor_target_state(0): X 坐标位置
            armor_target_state(1): X 方向速度
            armor_target_state(2): Y 坐标位置
            armor_target_state(3): Y 方向速度
            */
        }
        else if(yaw_diff > max_match_yaw_diff_)
        {
            // Matched armor not found, but there is only one armor with the same id
            // and yaw has jumped, take this case as the target is spinning and armor jumped
            // 如果有且仅有一个具有相同ID的装甲板（same_id_armors_count == 1），
            // 并且偏航角差异大于预定义的最大匹配偏航角差异（max_match_yaw_diff_），
            // 则认为目标正在旋转且装甲板发生了跳变。
            handleArmorJump(*tracker_armor_);
        }
        else
        {
            // No matched armor found
            RCLCPP_WARN(rclcpp::get_logger("armor_tracker"), "No matched armor found!");
        }

    }

        // Prevent radius from spreading
        if (target_state_.at<double>(8,0) < 0.12) {
            target_state_.at<double>(8,0) = 0.12;
            ekf_->set_state(target_state_);
        } else if (target_state_.at<double>(8,0) > 0.4) {
            target_state_.at<double>(8,0) = 0.4;
            ekf_->set_state(target_state_);
        }

        // Tracking state machine
        if(tracker_armor_->status == "DETECTING")
        {
            if(is_matched)
            {
                tracker_armor_->detect_count++;
                if(tracker_armor_->detect_count > tracking_thres_)
                {
                    tracker_armor_->detect_count = 0;
                    tracker_armor_->status = "TRACKING";
                }
            }
            else
            {
                tracker_armor_->detect_count = 0;
                tracker_armor_->status = "LOST";
                tracker_armor_->armor_id = -1;
                tracker_armor_->armor_num = 0;
                tracker_armor_->armor_name = "";
            }
        }
        else if(tracker_armor_->status == "TRACKING")
        {
            if(is_matched == false)
            {
                tracker_armor_->lost_count++;
                tracker_armor_->status = "LOSTING";
            }
        }
        else if(tracker_armor_->status == "LOSTING")
        {
            if(is_matched == false)
            {
                tracker_armor_->lost_count++;
                if(tracker_armor_->lost_count > lost_thres_)
                {
                    tracker_armor_->lost_count = 0;
                    tracker_armor_->status = "LOST";
                    tracker_armor_->armor_id = -1;
                    tracker_armor_->armor_num = 0;
                    tracker_armor_->armor_name = "";
                }
            }
            else
            {
                tracker_armor_->lost_count = 0;
                tracker_armor_->status = "TRACKING";
            }
        }
}

void Tracker::handleArmorJump(const TrackerArmor &armor)
{
    // armor为这一次检测到的真实装甲板信息
    // target_state_未更新，为上一次的装甲板状态
    // tracking_state_为根据上一次装甲板信息得到的预测状态
    double yaw = orientationToYaw(armor.armor_pose.pose.orientation);
    target_state_.at<double>(6,0) = yaw;
    // Only 4 armors has 2 radius and height
    if(armor.armor_num == 4)
    {
        dz_ = target_state_.at<double>(4,0) - armor.armor_pose.pose.position.z;
        target_state_.at<double>(4,0) = armor.armor_pose.pose.position.z;
        std::swap(target_state_.at<double>(8,0), another_r_);
    }
    RCLCPP_WARN(rclcpp::get_logger("armor_tracker_node"), "Armor jump!");

    // 切换装甲板，重新初始化ArmorEKF
    double xa = armor.armor_pose.pose.position.x;
    double ya = armor.armor_pose.pose.position.y;

    cv::Mat state = (cv::Mat_<double>(4,1) << xa, 0, ya, 0);
    armor_ekf_->set_state(state);

    // If position difference is larger than max_match_distance_,
    // take this case as the ekf diverged, reset the state
    cv::Point3d current_armor_position = cv::Point3d(armor.armor_pose.pose.position.x, armor.armor_pose.pose.position.y, armor.armor_pose.pose.position.z);
    cv::Point3d target_armor_position = get_armor_position(target_state_);
    double distance_position_diff = cv::norm(current_armor_position - target_armor_position);
    if(distance_position_diff > max_match_distance_)
    {
        RCLCPP_WARN(rclcpp::get_logger("armor_tracker_node"), "EKF diverged!");
        double r = target_state_.at<double>(8,0);
        target_state_.at<double>(0,0) = armor.armor_pose.pose.position.x + r * sin(yaw);
        target_state_.at<double>(1,0) = 0.0;
        target_state_.at<double>(2,0) = armor.armor_pose.pose.position.y - r * cos(yaw);
        target_state_.at<double>(3,0) = 0.0;
        target_state_.at<double>(4,0) = armor.armor_pose.pose.position.z;
        target_state_.at<double>(5,0) = 0.0;
        RCLCPP_ERROR(rclcpp::get_logger("armor_tracker"), "Reset State!");
    }

    ekf_->set_state(target_state_);
}


// 用于将给定的四元数表示的姿态（朝向）转换为偏航角（yaw）
double Tracker::orientationToYaw(const geometry_msgs::msg::Quaternion & q)
{
  // Get armor yaw
  tf2::Quaternion tf_q;
  tf2::fromMsg(q, tf_q);
  double roll, pitch, yaw;
  tf2::Matrix3x3(tf_q).getRPY(roll, pitch, yaw);
  // Make yaw change continuous (-pi~pi to -inf~inf)
  // 将当前计算得到的偏航角 yaw 与上一次记录的偏航角 last_yaw_ 进行比较，
  // 使用 angles::shortest_angular_distance 函数计算两者之间的最短角度差，
  // 从而确保偏航角变化在连续范围内。这一步旨在解决角度从 -pi 到 pi 的跳变问题，使角度变化连续。
  yaw = last_yaw_ + angles::shortest_angular_distance(last_yaw_, yaw);
  last_yaw_ = yaw;
  return yaw;
}

cv::Point3d Tracker::get_armor_position(const cv::Mat &state)
{
    double xc = state.at<double>(0,0);
    double yc = state.at<double>(2,0);
    double za = state.at<double>(4,0);
    double yaw = state.at<double>(6,0);
    double r = state.at<double>(8,0);

    double xa = xc - r * sin(yaw);
    double ya = yc + r * cos(yaw);
    return cv::Point3d(xa, ya, za);
}