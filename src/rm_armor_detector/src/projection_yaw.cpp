#include "projection_yaw.hpp"

ProjectionYaw::ProjectionYaw()
{
    std::cout << "ProjectionYaw constructed!" << std::endl;
    parameters_init();
}

void ProjectionYaw::parameters_init()
{
    PITCH_ = 15.0 * CV_PI / 180.0;
    ITERATIONS_NUM_ = 12;
    DETECTOR_ERROR_PIXEL_BY_SLOPE = 2;

    // 1  2
    // 4  3
    big_armor_world_points_ = {
        cv::Point3f(0.115, -0.0265, 0.),
        cv::Point3f(-0.115, -0.0265, 0.),
        cv::Point3f(-0.115, 0.0265, 0.),
        cv::Point3f(0.115, 0.0265, 0.)
    };

    small_armor_world_points_ = {
        cv::Point3f(0.068, -0.0275, 0.),
        cv::Point3f(-0.068, -0.0275, 0.),
        cv::Point3f(-0.068, 0.0275, 0.),
        cv::Point3f(0.068, 0.0275, 0.)
    };
}

void ProjectionYaw::set_matrix(const cv::Mat& camera_matrix, const cv::Mat& dist_coeffs)
{
    camera_matrix_ = camera_matrix;
    dist_coeffs_ = dist_coeffs;
}

void ProjectionYaw::update_tf2_buffer(const std::shared_ptr<tf2_ros::Buffer>& tf2_buffer, const std::shared_ptr<tf2_ros::TransformListener>& tf2_listener)
{
    tf2_buffer_ = tf2_buffer;
    tf2_listener_ = tf2_listener;
}

double ProjectionYaw::get_yaw(
    const DecisionArmor& decision_armor
)
{
    double pred_yaw = 0.0;
    pred_yaw = update_pred_yaw(decision_armor, -CV_PI / 2, CV_PI / 2, ITERATIONS_NUM_);
    //return reduced_angle(pred_yaw);
    return reduced_angle(pred_yaw);
}

double ProjectionYaw::update(
    const DecisionArmor& decision_armor,
    const double& pred_yaw
)
{
    std::vector<cv::Point2f> pred_points = get_pred_points(decision_armor, PITCH_, pred_yaw);
    std::vector<cv::Point2f> points = decision_armor.four_points;
    double cost = get_cost(pred_points, points, pred_yaw);
    return cost;
}

double ProjectionYaw::get_cost(const std::vector<cv::Point2f>& pred_points, 
                                const std::vector<cv::Point2f>& points,const double& pred_yaw)
{
    std::size_t size = pred_points.size();
    double cost = 0.0;
    for(std::size_t i = 0; i < size; i++)
    {
        std::size_t p = (i + 1u) % size;
        cv::Point2f pred_point_standard = pred_points[p] - pred_points[i];
        cv::Point2f point_standard = points[p] - points[i];
        double pixel_dis = (0.5 * (cv::norm(pred_points[i] - points[i]) + cv::norm(pred_points[p] - points[p]))
                            + std::fabs(cv::norm(pred_point_standard) - cv::norm(point_standard))) / cv::norm(pred_point_standard);
        double angular_dis = cv::norm(pred_point_standard) * get_abs_angle(pred_point_standard, point_standard) / cv::norm(pred_point_standard);
        double cost_i = std::pow(pixel_dis * std::sin(pred_yaw), 2) + std::pow(angular_dis * std::cos(pred_yaw), 2) * DETECTOR_ERROR_PIXEL_BY_SLOPE;
        cost += std::sqrt(cost_i);
    }
    return cost;
}

double ProjectionYaw::get_abs_angle(const cv::Point2f& first_point, const cv::Point2f& second_point)
{
    if(cv::norm(first_point) == 0 || cv::norm(second_point) == 0)
    {
        std::cout << "Error: The norm of the point is zero!" << std::endl;
        return 0;
    }
    double cos_theta = (first_point.x * second_point.x + first_point.y * second_point.y) / (cv::norm(first_point) * cv::norm(second_point));
    return std::acos(cos_theta);
}

// 利用三分法计算极小值，不断更新预测的偏航角
double ProjectionYaw::update_pred_yaw(const DecisionArmor& decision_armor,double left_yaw, double right_yaw,const int& ITERATIONS_NUM)
{
    double phi = (std::sqrt(5.0) - 1.0) / 2.0;
    int choice = -1;
    double left_cost = 0.0;
    double right_cost = 0.0;
    for(int i = 0; i < ITERATIONS_NUM; i++)
    {
        double ml = left_yaw + (right_yaw - left_yaw) * (1 - phi);
        double mr = left_yaw + (right_yaw - left_yaw) * phi;
        if(choice != 0)
        {
            left_cost = update(decision_armor, ml);
        }
        if(choice != 1)
        {
            right_cost = update(decision_armor, mr);
        }
        //std::cout << "left_cost: " << left_cost << " right_cost: " << right_cost << std::endl;
        //std::cout << "ml: " << ml << " mr: " << mr << std::endl;
        if(left_cost < right_cost)
        {
            right_yaw = mr;
            right_cost = left_cost;
            choice = 1;
        }
        else
        {
            left_yaw = ml;
            left_cost = right_cost;
            choice = 0;
        }
    }
    //std::cout << "left_yaw: " << left_yaw << " right_yaw: " << right_yaw << std::endl;
    //std::cout << "result: " << ((left_yaw + right_yaw) / 2.0) * 57.3f << std::endl;
    return (left_yaw + right_yaw) / 2;
}

std::vector<cv::Point3f> ProjectionYaw::get_armor_points(
    const DecisionArmor& decision_armor,
    const double& pitch,
    const double& pred_yaw
){
    std::vector<cv::Point3f> armor_world_points;
    if(decision_armor.is_big_armor == true)
    {
        armor_world_points = big_armor_world_points_;
    }
    else if(decision_armor.is_big_armor == false)
    {
        armor_world_points = small_armor_world_points_;
    }

    geometry_msgs::msg::PoseStamped ros_armor_center;
    try {
        // Use the latest available transform instead of a specific timestamp
        geometry_msgs::msg::TransformStamped transformStamped = tf2_buffer_->lookupTransform(
            "horizontal_camera_link", "camera_optical_frame",
            decision_armor.header.stamp); // Use the latest available transform

        // 将装甲板坐标变换到horizontal平面
        ros_armor_center.pose.position.x = decision_armor.pose.position.x;
        ros_armor_center.pose.position.y = decision_armor.pose.position.y;
        ros_armor_center.pose.position.z = decision_armor.pose.position.z;
        ros_armor_center.header.frame_id = "camera_optical_frame";
        ros_armor_center.header.stamp = decision_armor.header.stamp;
        
        tf2::doTransform(ros_armor_center, ros_armor_center, transformStamped);
    } catch (tf2::TransformException &ex) {
        std::cout << "Could not transform armor center: " << ex.what() << std::endl;
    }

    cv::Point3f armor_center = cv::Point3f(ros_armor_center.pose.position.x, ros_armor_center.pose.position.y, ros_armor_center.pose.position.z);

    // x z y
    cv::Mat radius_vec = (cv::Mat_<double>(2, 1) << 0, 1);
    radius_vec = rotate(radius_vec, pred_yaw);
    cv::Mat xz_2_vec = rotate(radius_vec, - CV_PI / 2);
    cv::Mat xz_vec = (cv::Mat_<double>(3, 1) << xz_2_vec.at<double>(0), xz_2_vec.at<double>(1), 0);
    cv::Mat y_vec = (cv::Mat_<double>(3, 1) << radius_vec.at<double>(0) * std::sin(pitch), radius_vec.at<double>(1) * std::sin(pitch)
                                                ,  std::cos(pitch));
    std::vector<cv::Point3f> armor_points;
    for(size_t i = 0; i < armor_world_points.size(); i++)
    {
        cv::Mat xz_trans = armor_world_points[i].x * xz_vec;
        // xzy 转为 xyz
        cv::Point3f xz_trans_point = cv::Point3f(xz_trans.at<double>(0,0),xz_trans.at<double>(2,0),xz_trans.at<double>(1,0));
        cv::Mat y_trans = armor_world_points[i].y * y_vec;
        // xzy 转为 xyz
        cv::Point3f y_trans_point = cv::Point3f(y_trans.at<double>(0,0),y_trans.at<double>(2,0),y_trans.at<double>(1,0));
        armor_points.push_back(armor_center + xz_trans_point + y_trans_point);
    }
    return armor_points;
}

std::vector<cv::Point2f> ProjectionYaw::get_pred_points(
    const DecisionArmor& decision_armor,
    const double& pitch,
    const double& pred_yaw
)
{
    std::vector<cv::Point3f> armor_points = get_armor_points(decision_armor, pitch, pred_yaw);
    std::vector<cv::Point2f> pred_points;
    
    std::vector<cv::Point3f> camera_points;
    for(size_t i = 0; i < armor_points.size(); i++)
    {
        geometry_msgs::msg::PointStamped transform_point;
        try{
            geometry_msgs::msg::TransformStamped transformStamped = tf2_buffer_->lookupTransform(
                "camera_optical_frame", "horizontal_camera_link",
                decision_armor.header.stamp); // Use the latest available transform

            transform_point.point.x = armor_points[i].x;
            transform_point.point.y = armor_points[i].y;
            transform_point.point.z = armor_points[i].z;
            transform_point.header.frame_id = "horizontal_camera_link";
            transform_point.header.stamp = decision_armor.header.stamp;
            
            tf2::doTransform(transform_point, transform_point, transformStamped);
        } catch (tf2::TransformException &ex) {
            std::cout << "Could not transform armor center: " << ex.what() << std::endl;
        }
        cv::Point3f armor_point = cv::Point3f(transform_point.point.x, transform_point.point.y, transform_point.point.z);
        camera_points.push_back(armor_point);
    }
    // 直接从相机坐标系转到像素坐标系，rvec和tvec都是0
    cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64F);
    cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64F);
    cv::projectPoints(camera_points, rvec, tvec, camera_matrix_, dist_coeffs_, pred_points);
    return pred_points;
}

// 将二维向量逆时针旋转角度angle
// vec : 2 * 1的矩阵
// angle : 旋转角度 弧度制
cv::Mat ProjectionYaw::rotate(const cv::Mat& vec, const double& angle)
{
    cv::Mat rotate_mat = (cv::Mat_<double>(2, 2) << std::cos(angle), -std::sin(angle), std::sin(angle), std::cos(angle));
    return rotate_mat * vec;
}

// 限制到 -pi ~ pi
double ProjectionYaw::reduced_angle(const double& x) 
{
    return std::atan2(std::sin(x), std::cos(x));
}