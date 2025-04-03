#include "closed_loop_node.hpp"

namespace rm_closed_loop
{

ClosedLoopNode::ClosedLoopNode(const rclcpp::NodeOptions & options) : Node("closed_loop_node", options)
{
    RCLCPP_INFO(this->get_logger(), "Starting ClosedLoopNode!");
    parameters_init();
    //test();
}

void ClosedLoopNode::parameters_init()
{
    RCLCPP_INFO(this->get_logger(), "Begin to init ClosedLoopNode parameters!");

    init_time_ = this->now().nanoseconds() / 1000000;

    PITCH_ = 15.0 * CV_PI / 180.0;

    size_t capacity = 20;
    all_armor_images_ = std::make_shared<FixedSizeMapQueue<int64_t, cv::Mat>>();
    all_armor_images_->set_capacity(capacity);

    closed_loop_ = std::make_shared<ClosedLoop>();
    closed_loop_->all_projectiles_messages_->set_capacity(capacity);

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

    // Subscriber with tf2 message_filter
    // tf2 relevant
    tf2_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    // Create the timer interface before call to waitForTransform,
    // to avoid a tf2_ros::CreateTimerInterfaceException exception
    auto timer_interface = std::make_shared<tf2_ros::CreateTimerROS>(
        this->get_node_base_interface(), this->get_node_timers_interface());
    tf2_buffer_->setCreateTimerInterface(timer_interface);
    tf2_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf2_buffer_);

    // Pub
    closed_loop_result_image_pub_ = image_transport::create_publisher(this, "closed_loop/result_image");

    // Sub
    camera_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
        "camera_info", rclcpp::SensorDataQoS(), std::bind(&ClosedLoopNode::camera_info_callback, this, std::placeholders::_1));

    trajectory_closed_loop_sub_ = this->create_subscription<rm_msgs::msg::ClosedLoop>(
        "/trajectory/closed_loop", 10, std::bind(&ClosedLoopNode::trajectory_closed_loop_callback, this, std::placeholders::_1));

    tracker_image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
        "/image_raw", rclcpp::SensorDataQoS(), std::bind(&ClosedLoopNode::tracker_image_callback, this, std::placeholders::_1));

    //simulated_projectile_trajectory_timer_ = this->create_wall_timer(std::chrono::milliseconds(2), std::bind(&ClosedLoopNode::simulated_projectile_trajectory, this));

    RCLCPP_INFO(this->get_logger(), "Finished init ClosedLoopNode parameters successfully!");
}

void ClosedLoopNode::camera_info_callback(const sensor_msgs::msg::CameraInfo::SharedPtr msg)
{
    camera_matrix_ = cv::Mat(3, 3, CV_64F, (void *)msg->k.data()).clone();
    distortion_coefficients_ = cv::Mat(1, 5, CV_64F, (void *)msg->d.data()).clone();

    std::cout << "ClosedLoopNode camera_matrix: " << camera_matrix_ << std::endl;
    std::cout << "ClosedLoopNode distortion_coefficients: " << distortion_coefficients_ << std::endl;

    camera_info_sub_.reset();
}

void ClosedLoopNode::trajectory_closed_loop_callback(const rm_msgs::msg::ClosedLoop::SharedPtr msg)
{
    geometry_msgs::msg::PoseStamped now_odom_armor_pose;
    now_odom_armor_pose.pose = msg->now_armor_pose;
    try
    {
        geometry_msgs::msg::TransformStamped transformStamped = tf2_buffer_->lookupTransform(
            "camera_optical_frame", "odom", 
            msg->image_header.stamp);

        tf2::doTransform(msg->now_pose, msg->now_pose, transformStamped);

        tf2::doTransform(msg->now_armor_pose, msg->now_armor_pose, transformStamped);
    }
    catch (tf2::TransformException &ex)
    {
        RCLCPP_ERROR(this->get_logger(), "Transform error: %s", ex.what());
        return;
    }

    builtin_interfaces::msg::Time draw_image_stamp = msg->image_header.stamp;
    int64_t draw_image_time = draw_image_stamp.sec * 1000LL + draw_image_stamp.nanosec / 1000000LL;
    int64_t draw_bias_time;
    if(draw_image_time > init_time_)
    {
        draw_bias_time = draw_image_time - init_time_; //ms
        bool is_contain = all_armor_images_->contains(draw_bias_time);
        if(!is_contain)
        {
            //std::cout << "draw_bias_time: " << draw_bias_time << " is not in all_armor_images_!" << std::endl;
            return;
        }
        cv::Mat draw_image = all_armor_images_->get(draw_bias_time);
        if(draw_image.empty())
        {
            //std::cout << "draw_image is empty!" << std::endl;
            return;
        }

        //------------------------------------------------------------------------------------------------
        // 模拟弹道轨迹
        Looper looper;
        looper.image_time_ = draw_bias_time;
        builtin_interfaces::msg::Time shoot_image_stamp = msg->shoot_header.stamp;
        int64_t shoot_image_time = shoot_image_stamp.sec * 1000LL + shoot_image_stamp.nanosec / 1000000LL;
        looper.shoot_time_ = shoot_image_time - init_time_;
        looper.v0 = msg->v0;
        looper.theta_ = msg->theta;
        looper.fly_t_ = msg->fly_t;
        looper.accumulated_time_ = 0;
        // 得到发射点坐标--odom坐标系下
        geometry_msgs::msg::PoseStamped odom_projectile_pose; //-- shoot坐标系下
        odom_projectile_pose.pose.position.x = 0;
        odom_projectile_pose.pose.position.y = 0;
        odom_projectile_pose.pose.position.z = 0;
        odom_projectile_pose.pose.orientation.x = 0;
        odom_projectile_pose.pose.orientation.y = 0;
        odom_projectile_pose.pose.orientation.z = 0;
        odom_projectile_pose.pose.orientation.w = 1;
        try
        {
            geometry_msgs::msg::TransformStamped transformStamped = tf2_buffer_->lookupTransform(
                "odom", "shoot", 
                tf2::TimePointZero);

            tf2::doTransform(odom_projectile_pose, odom_projectile_pose, transformStamped);
        }
        catch (tf2::TransformException &ex)
        {
            RCLCPP_ERROR(this->get_logger(), "Transform error: %s", ex.what());
            return;
        }
        looper.odom_projectile_pose_ = odom_projectile_pose;
        // 得到目标点坐标--odom坐标系下
        looper.odom_armor_pose_.pose = now_odom_armor_pose.pose;
        looper.odom_armor_pose_.header = msg->image_header;
        // 得到弹丸在图像时间戳下的像素坐标
        double add_time = 0.002;
        while(looper.accumulated_time_ < looper.fly_t_ && rclcpp::ok())
        {
            //std::cout << "looper.odom_projectile_pose_: " << looper.odom_projectile_pose_.pose.position.x << " " << looper.odom_projectile_pose_.pose.position.y << " " << looper.odom_projectile_pose_.pose.position.z << std::endl;
            //std::cout << "looper.odom_armor_pose_: " << looper.odom_armor_pose_.pose.position.x << " " << looper.odom_armor_pose_.pose.position.y << " " << looper.odom_armor_pose_.pose.position.z << std::endl;
            geometry_msgs::msg::PoseStamped projectile_pose = closed_loop_->get_projectile_pose(looper.odom_projectile_pose_, looper.odom_armor_pose_, looper.accumulated_time_, looper.v0, looper.theta_);
            //std::cout << "projectile_pose: " << projectile_pose.pose.position.x << " " << projectile_pose.pose.position.y << " " << projectile_pose.pose.position.z << std::endl;
            try
            {
                geometry_msgs::msg::TransformStamped transformStamped = tf2_buffer_->lookupTransform(
                    "camera_optical_frame", "odom", 
                    tf2::TimePointZero);

                tf2::doTransform(projectile_pose, projectile_pose, transformStamped);
            }
            catch (tf2::TransformException &ex)
            {
                RCLCPP_ERROR(this->get_logger(), "Transform error: %s", ex.what());
                return;
            }
            //std::cout << "camera _- projectile_pose: " << projectile_pose.pose.position.x << " " << projectile_pose.pose.position.y << " " << projectile_pose.pose.position.z << std::endl;
            cv::Point3d cv_projectile_point = cv::Point3d(projectile_pose.pose.position.x, projectile_pose.pose.position.y, projectile_pose.pose.position.z);
            std::vector<cv::Point2d> projectile_image_points = get_image_points({cv_projectile_point});
            //std::cout << "projectile_image_points: " << projectile_image_points[0] << std::endl;
            //std::cout << "looper.accumulated_time_: " << looper.accumulated_time_ << std::endl;
            //std::cout << "looper.fly_t_: " << looper.fly_t_ << std::endl;
            looper.projectile_image_points_[looper.shoot_time_ + looper.accumulated_time_] = projectile_image_points[0];
            looper.accumulated_time_ += add_time;
        }
        closed_loop_->update_projectiles_messages(looper);
        //------------------------------------------------------------------------------------------------

        //------------------------------------------------------------------------------------------------
        // 绘制全车装甲板
        geometry_msgs::msg::PoseStamped now_pose;
        now_pose.pose = msg->now_pose;
        now_pose.header = msg->image_header;
        // 处理的地方msg->c_to_a_pitch可能要个负值，因为pitch转动方向和用rotate函数的方向相反
        // 先前在给camera_optical_frame到odom的变换时，pitch的值为-180 / CV_PI- yaw, 那么这里应该把pitch取负值 -- 需测试 2024.10.9
        // 测试结果：取负值
        //draw_armor_on_image(draw_image, now_pose, msg->id, msg->armor_num, msg->r, msg->another_r, msg->dz, msg->c_to_a_pitch);
        draw_armor_on_image(draw_image, now_pose, msg->id, msg->armor_num, msg->r, msg->another_r, msg->dz, - msg->c_to_a_pitch);
        // 绘制当前装甲板
        
        // 提取四元数的各个分量
        /*double w = now_pose.pose.orientation.w;
        double x = now_pose.pose.orientation.x;
        double y = now_pose.pose.orientation.y;
        double z = now_pose.pose.orientation.z;

        // 计算欧拉角

        double sinp = 2 * (w * y - z * x);
        double pitch;
        if (std::abs(sinp) >= 1)
            pitch = std::copysign(M_PI / 2, sinp); // 使用90度或-90度
        else
            pitch = std::asin(sinp);

        double sinr_cosp = 2 * (w * x + y * z);
        double cosr_cosp = 1 - 2 * (x * x + y * y);
        double roll = std::atan2(sinr_cosp, cosr_cosp);

        double siny_cosp = 2 * (w * z + x * y);
        double cosy_cosp = 1 - 2 * (y * y + z * z);
        double yaw = std::atan2(siny_cosp, cosy_cosp);*/

        cv::Point3d now_armor_center = cv::Point3d(msg->now_armor_pose.position.x, msg->now_armor_pose.position.y, msg->now_armor_pose.position.z);
        cv::Point2d now_armor_center_image_point = get_image_points({now_armor_center})[0];
        // 与上同理
        // 处理的地方msg->c_to_a_pitch可能要个负值，因为pitch转动方向和用rotate函数的方向相反
        // 先前在给camera_optical_frame到odom的变换时，pitch的值为- CV_PI - yaw, 那么这里应该把pitch取负值 -- 需测试 2024.10.9
        // 测试结果：取负值
        std::vector<cv::Point3d> now_armor_3d_points = get_armor_3d_points(now_armor_center, small_armor_world_points_, - msg->c_to_a_pitch);
        std::vector<cv::Point2d> now_armor_image_points = get_image_points(now_armor_3d_points);
        cv::line(draw_image, now_armor_image_points[0], now_armor_image_points[1], cv::Scalar(255, 0, 0), 2);
        cv::line(draw_image, now_armor_image_points[1], now_armor_image_points[2], cv::Scalar(255, 0, 0), 2);
        cv::line(draw_image, now_armor_image_points[2], now_armor_image_points[3], cv::Scalar(255, 0, 0), 2);
        cv::line(draw_image, now_armor_image_points[3], now_armor_image_points[0], cv::Scalar(255, 0, 0), 2);
        cv::circle(draw_image, now_armor_center_image_point, 10, cv::Scalar(255, 0, 0), -1);

        // 画上当前时刻的时间戳 ms
        std::string time_str = std::to_string(draw_bias_time) + "ms";
        cv::putText(draw_image, time_str, cv::Point(500, 10), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 3);
        // 画上当前时刻的完整时间戳 ms
        std::string full_time_str = std::to_string(draw_image_time) + "ms";
        cv::putText(draw_image, full_time_str, cv::Point(500, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 3);
        all_armor_images_->change(draw_bias_time, draw_image);

        // 绘制弹丸模拟轨迹
        for(std::map<double, cv::Point2d>::iterator iter = looper.projectile_image_points_.begin(); iter != looper.projectile_image_points_.end(); iter++)
        {
            cv::circle(draw_image, iter->second, 10, cv::Scalar(0, 0, 255), -1);
            // 在圆圈上画上图片时间戳
            std::string time_str = std::to_string(iter->first) + "ms";
            cv::putText(draw_image, time_str, iter->second + cv::Point2d(0, -50), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 3);
        }

        cv_bridge::CvImage cv_image;
        cv_image.image = draw_image;
        cv_image.encoding = "bgr8";
        cv_image.header = msg->image_header;
        sensor_msgs::msg::Image draw_image_msg = *(cv_image.toImageMsg());
        closed_loop_result_image_pub_.publish(draw_image_msg);
        //------------------------------------------------------------------------------------------------
    }
}

void ClosedLoopNode::tracker_image_callback(const sensor_msgs::msg::Image::SharedPtr msg)
{
    cv_bridge::CvImagePtr cv_ptr;
    try
    {
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    }
    catch (cv_bridge::Exception &e)
    {
        RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
        return;
    }
    
    builtin_interfaces::msg::Time image_stamp = msg->header.stamp;
    int64_t image_time = image_stamp.sec * 1000LL + image_stamp.nanosec / 1000000LL;
    int64_t bias_time;
    if(image_time > init_time_)
    {
        bias_time = image_time - init_time_; //ms
        all_armor_images_->insert(bias_time, cv_ptr->image);
        closed_loop_->add_projectiles_messages(bias_time, cv_ptr->image);
    }

    if(all_armor_images_->get_size() == 0)
    {
        return;
    }
}

void ClosedLoopNode::simulated_projectile_trajectory()
{
    if(closed_loop_->all_projectiles_messages_->get_size() == 0)
    {
        return;
    }
    if(closed_loop_->all_projectiles_messages_->get_begin().second.fly_t_ <= 0)
    {
        if(closed_loop_->all_projectiles_messages_->get_begin().second.accumulated_time_ > 0)
        {
            //std::cout << "The projectile trajectory is over -- pop pop pop!" << std::endl;
            closed_loop_->all_projectiles_messages_->pop();
        }
        return;
    }

    Looper looper = closed_loop_->all_projectiles_messages_->get_begin().second;
    looper.accumulated_time_ += 0.002;
    looper.fly_t_ -= 0.002;
    geometry_msgs::msg::PoseStamped projectile_pose; // -- odom坐标系下
    projectile_pose = closed_loop_->get_projectile_pose(looper.odom_projectile_pose_, looper.odom_armor_pose_, looper.accumulated_time_, looper.v0, looper.theta_);
    try
    {
        geometry_msgs::msg::TransformStamped transformStamped = tf2_buffer_->lookupTransform(
            "camera_optical_frame", "odom",
            tf2::TimePointZero);

        tf2::doTransform(projectile_pose, projectile_pose, transformStamped);
    }
    catch (tf2::TransformException &ex)
    {
        RCLCPP_ERROR(this->get_logger(), "Transform error: %s", ex.what());
        return;
    }
    closed_loop_->all_projectiles_messages_->change(looper.image_time_, looper);
    cv::Point3d cv_projectile_pose = cv::Point3d(projectile_pose.pose.position.x, projectile_pose.pose.position.y, projectile_pose.pose.position.z);
    std::vector<cv::Point2d> cv_projectile_image_points = get_image_points({cv_projectile_pose});
    cv::Mat draw_image = all_armor_images_->get(looper.image_time_);
    cv::circle(draw_image, cv_projectile_image_points[0], 10, cv::Scalar(0, 0, 255), -1);
    // 在圆圈上画上图片时间戳
    std::string time_str = std::to_string(looper.image_time_) + "ms";
    cv::putText(draw_image, time_str, cv_projectile_image_points[0] + cv::Point2d(0, -50), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 3);
    cv_bridge::CvImage cv_image;
    cv_image.image = draw_image;
    cv_image.encoding = "bgr8";
    cv_image.header.stamp = this->now();
    sensor_msgs::msg::Image draw_image_msg = *(cv_image.toImageMsg());
    closed_loop_result_image_pub_.publish(draw_image_msg);
}

void ClosedLoopNode::draw_armor_on_image(cv::Mat image, const geometry_msgs::msg::PoseStamped & armor_pose, std::string id, int armor_num, double r, double another_r, double dz, double yaw)
{
    cv::Point3d car_center = cv::Point3d(armor_pose.pose.position.x, armor_pose.pose.position.y, armor_pose.pose.position.z);
    std::vector<cv::Point3d> armor_3d_points;
    std::vector<cv::Point2d> armor_image_points;
    std::vector<std::vector<cv::Point2d>> all_armor_image_points;
    if(armor_num == 2)
    {
        for(size_t i = 0; i < 2; i++)
        {
            double x = car_center.x - r * std::sin(i * CV_PI + yaw);
            double z = car_center.z + r * std::cos(i * CV_PI + yaw);
            cv::Point3d armor_center = cv::Point3d(x, car_center.y, z); 
            armor_3d_points = get_armor_3d_points(armor_center, big_armor_world_points_, i * CV_PI + yaw);
            armor_image_points = get_image_points(armor_3d_points);
            all_armor_image_points.push_back(armor_image_points);
        }
    }
    else if(armor_num == 3)
    {
        for(size_t i = 0; i < 3; i++)
        {
            double x = car_center.x - r * std::sin(i * CV_PI * 2 / 3 + yaw);
            double z = car_center.z + r * std::cos(i * CV_PI * 2 / 3 + yaw);
            cv::Point3d armor_center = cv::Point3d(x, car_center.y, z); 
            armor_3d_points = get_armor_3d_points(armor_center, small_armor_world_points_, i * CV_PI * 2 / 3 + yaw);
            armor_image_points = get_image_points(armor_3d_points);
            all_armor_image_points.push_back(armor_image_points);
        }
    }
    else if(armor_num == 4 && id != "robot1-4")
    {
        for(size_t i = 0; i < 4; i++)
        {
            double x = 0.0;
            double z = 0.0;
            cv::Point3d armor_center;
            // 测试是否cos，sin计算方法有误，经过计算，应该为x-r*sin(i * CV_PI / 2 + yaw)，z+r*cos(i * CV_PI / 2 + yaw) ，--需测试 2024.10.9
            // 测试结果：确定计算方法有误，应该为x-r*sin(i * CV_PI / 2 + yaw)，z+r*cos(i * CV_PI / 2 + yaw)，同理修改了tracker里面追踪器的算法
            if(i == 0 || i == 2)
            {
                /*x = car_center.x + r * std::cos(i * CV_PI / 2 + yaw);
                z = car_center.z + r * std::sin(i * CV_PI / 2 + yaw);*/
                x = car_center.x - r * std::sin(i * CV_PI / 2 + yaw);
                z = car_center.z + r * std::cos(i * CV_PI / 2 + yaw);
                armor_center = cv::Point3d(x, car_center.y, z); 
            }
            else if(i == 1 || i == 3)
            {
                /*x = car_center.x + another_r * std::cos(i * CV_PI / 2 + yaw);
                z = car_center.z + another_r * std::sin(i * CV_PI / 2 + yaw);*/
                x = car_center.x - another_r * std::sin(i * CV_PI / 2 + yaw);
                z = car_center.z + another_r * std::cos(i * CV_PI / 2 + yaw);
                armor_center = cv::Point3d(x, car_center.y + dz, z); 
            }
            armor_3d_points = get_armor_3d_points(armor_center, small_armor_world_points_, i * CV_PI / 2 + yaw);
            armor_image_points = get_image_points(armor_3d_points);
            all_armor_image_points.push_back(armor_image_points);
        }
    }
    else if(armor_num == 4 && id == "robot1-4")
    {
        for(size_t i = 0; i < 4; i++)
        {
            double x = 0.0;
            double z = 0.0;
            cv::Point3d armor_center;
            if(i == 0 || i == 2)
            {
                x = car_center.x - r * std::sin(i * CV_PI / 2 + yaw);
                z = car_center.z + r * std::cos(i * CV_PI / 2 + yaw);
                armor_center = cv::Point3d(x, car_center.y + dz, z); 
            }
            else if(i == 1 || i == 3)
            {
                x = car_center.x - another_r * std::sin(i * CV_PI / 2 + yaw);
                z = car_center.z + another_r * std::cos(i * CV_PI / 2 + yaw);
                armor_center = cv::Point3d(x, car_center.y + dz, z); 
            }
            armor_3d_points = get_armor_3d_points(armor_center, big_armor_world_points_, i * CV_PI / 2 + yaw);
            armor_image_points = get_image_points(armor_3d_points);
            all_armor_image_points.push_back(armor_image_points);
        }
    }
    cv::Point2d car_center_image_point = get_image_points({car_center})[0];

    // 将一个装甲板的四个点连接起来
    for(size_t i = 0; i < all_armor_image_points.size(); i++)
    {
        cv::Scalar color = cv::Scalar(i * 255 / all_armor_image_points.size(), i * 255 / all_armor_image_points.size(), i * 255 / all_armor_image_points.size());
        for(size_t j = 0; j < all_armor_image_points[i].size(); j++)
        {
            cv::line(image, all_armor_image_points[i][j], all_armor_image_points[i][(j + 1) % 4], color, 2);
        }
    }
    // 标出车中心
    cv::circle(image, car_center_image_point, 10, cv::Scalar(0, 0, 255), -1);
    // 将车中心与装甲板中心连接起来
    for(size_t i = 0; i < all_armor_image_points.size(); i++)
    {
        double armor_center_image_x = (all_armor_image_points[i][0].x + all_armor_image_points[i][2].x) / 2;
        double armor_center_image_y = (all_armor_image_points[i][0].y + all_armor_image_points[i][2].y) / 2;
        cv::Point2d armor_center_image_point = cv::Point2d(armor_center_image_x, armor_center_image_y);
        cv::line(image, car_center_image_point, armor_center_image_point, cv::Scalar(0, 255, 0), 2);
    }
    // 将前一个装甲板的两个点与后一个装甲板的两个点连接起来
    for(size_t i = 0; i < all_armor_image_points.size() ; i++)
    {
        cv::line(image, all_armor_image_points[i][1], all_armor_image_points[(i+1) % armor_image_points.size()][0], cv::Scalar(0, 255, 0), 2);
        cv::line(image, all_armor_image_points[i][2], all_armor_image_points[(i+1) % armor_image_points.size()][3], cv::Scalar(0, 255, 0), 2);
    }
}

std::vector<cv::Point3d> ClosedLoopNode::get_armor_3d_points(cv::Point3d & armor_center, std::vector<cv::Point3d> armor_world_points, double yaw)
{
    cv::Mat radius_vec = (cv::Mat_<double>(2, 1) << 0, 1);
    radius_vec = rotate(radius_vec, yaw);
    cv::Mat xz_2_vec = rotate(radius_vec, - CV_PI / 2);
    cv::Mat xz_vec = (cv::Mat_<double>(3, 1) << xz_2_vec.at<double>(0), xz_2_vec.at<double>(1), 0);
    cv::Mat y_vec = (cv::Mat_<double>(3, 1) << radius_vec.at<double>(0) * std::sin(PITCH_), radius_vec.at<double>(1) * std::sin(PITCH_)
                                                ,  std::cos(PITCH_));
    std::vector<cv::Point3d> armor_points;
    for(size_t i = 0; i < armor_world_points.size(); i++)
    {
        cv::Mat xz_trans = armor_world_points[i].x * xz_vec;
        // xzy 转为 xyz
        cv::Point3d xz_trans_point = cv::Point3f(xz_trans.at<double>(0,0),xz_trans.at<double>(2,0),xz_trans.at<double>(1,0));
        cv::Mat y_trans = armor_world_points[i].y * y_vec;
        // xzy 转为 xyz
        cv::Point3d y_trans_point = cv::Point3f(y_trans.at<double>(0,0),y_trans.at<double>(2,0),y_trans.at<double>(1,0));
        armor_points.push_back(armor_center + xz_trans_point + y_trans_point);
    }
    return armor_points;
}

std::vector<cv::Point2d> ClosedLoopNode::get_image_points(const std::vector<cv::Point3d> & armor_3d_points)
{
    std::vector<cv::Point2d> armor_image_points;
    cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64F);
    cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64F);
    cv::projectPoints(armor_3d_points, rvec, tvec, camera_matrix_, distortion_coefficients_, armor_image_points);
    return armor_image_points;
}

// 将二维向量逆时针旋转角度angle
// vec : 2 * 1的矩阵
// angle : 旋转角度 弧度制
cv::Mat ClosedLoopNode::rotate(const cv::Mat& vec, const double& angle)
{
    cv::Mat rotate_mat = (cv::Mat_<double>(2, 2) << std::cos(angle), -std::sin(angle), std::sin(angle), std::cos(angle));
    return rotate_mat * vec;
}

} // namespace rm_closed_loop

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(rm_closed_loop::ClosedLoopNode)