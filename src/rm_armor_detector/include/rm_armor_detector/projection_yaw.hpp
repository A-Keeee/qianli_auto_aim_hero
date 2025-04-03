#include "opencv2/opencv.hpp"

#include "armor.hpp"

#include <iostream>

#include <message_filters/subscriber.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/create_timer_ros.h>
#include <tf2_ros/message_filter.h>
#include <tf2_ros/transform_listener.h>

#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/point_stamped.hpp"

#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"

class ProjectionYaw
{
public:
    ProjectionYaw();
    void parameters_init();
    void set_matrix(const cv::Mat& camera_matrix, const cv::Mat& dist_coeffs);
    void update_tf2_buffer(const std::shared_ptr<tf2_ros::Buffer>& tf2_buffer, const std::shared_ptr<tf2_ros::TransformListener>& tf2_listener);
    double get_yaw(const DecisionArmor& decision_armor);
    double update(const DecisionArmor& decision_armor,const double& pred_yaw);
    double get_cost(const std::vector<cv::Point2f>& pred_points, const std::vector<cv::Point2f>& points,const double& pred_yaw);
    double get_abs_angle(const cv::Point2f& first_point, const cv::Point2f& second_point);
    double update_pred_yaw(const DecisionArmor& decision_armor,double left_yaw, double right_yaw,const int& ITERATIONS_NUM);
    std::vector<cv::Point3f> get_armor_points(const DecisionArmor& decision_armor,const double& pitch,const double& pred_yaw);
    std::vector<cv::Point2f> get_pred_points(const DecisionArmor& decision_armor,const double& pitch,const double& pred_yaw);
    cv::Mat rotate(const cv::Mat& vec, const double& angle);
    double reduced_angle(const double& x);

    double PITCH_;
    int ITERATIONS_NUM_;
    int DETECTOR_ERROR_PIXEL_BY_SLOPE;
    std::vector<cv::Point3f> big_armor_world_points_;
    std::vector<cv::Point3f> small_armor_world_points_;

    cv::Mat camera_matrix_;
    cv::Mat dist_coeffs_;

    std::shared_ptr<tf2_ros::Buffer> tf2_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf2_listener_;
};