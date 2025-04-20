#include "tracker_node.hpp"

namespace rm_armor_tracker
{

ArmorTrackerNode::ArmorTrackerNode(const rclcpp::NodeOptions &options) : Node("armor_tracker_node", options)
{
    RCLCPP_INFO(this->get_logger(), "Starting ArmorTrackerNode!");
    parameters_init();
    //test();
}

void ArmorTrackerNode::parameters_init()
{
    this->declare_parameter("tracker_debug", 0);
    is_debug_ = this->get_parameter("tracker_debug").as_int();
    std::cout << "is_tracker_debug: " << is_debug_ << std::endl;
    if(is_debug_ == true)
    {
        create_debug_publishers();
    }
    debug_param_sub_ = std::make_shared<rclcpp::ParameterEventHandler>(this);
    debug_cb_handle_ =
    debug_param_sub_->add_parameter_callback("tracker_debug", [this](const rclcpp::Parameter & p) {
      is_debug_ = p.as_bool();
      is_debug_ ? create_debug_publishers() : destroy_debug_publishers();
    });

    // tracker
    this->declare_parameter("max_match_distance", 0.5);
    double max_match_distance = this->get_parameter("max_match_distance").as_double();
    this->declare_parameter("max_match_yaw_diff", 1.0);
    double max_match_yaw_diff = this->get_parameter("max_match_yaw_diff").as_double();
    tracker_ = std::make_shared<Tracker>(max_match_distance, max_match_yaw_diff);
    tracker_->ekf_ = std::make_shared<EKF>();
    tracker_->armor_ekf_ = std::make_shared<ArmorEKF>();

    this->declare_parameter("lost_time_thres", 0.3);
    lost_time_thres_ = this->get_parameter("lost_time_thres").as_double();

    this->declare_parameter("tracking_time_thres", 5);
    tracker_->tracking_thres_ = this->get_parameter("tracking_time_thres").as_int();

    std::cout << "max_match_distance: " << max_match_distance << std::endl;
    std::cout << "max_match_yaw_diff: " << max_match_yaw_diff << std::endl;
    std::cout << "lost_time_thres: " << lost_time_thres_ << std::endl;
    std::cout << "tracking_time_thres: " << tracker_->tracking_thres_ << std::endl;

    // time
    last_time_ = this->now();

    //-----------------------------------------------------------------
    // ekf

    this->declare_parameter("q_xyz", 20.0);
    this->declare_parameter("q_yaw", 200.0);
    this->declare_parameter("q_r", 800.0);
    this->declare_parameter("r_xyz", 0.002);
    this->declare_parameter("r_yaw", 0.005);

    tracker_->ekf_->q_xyz_ = this->get_parameter("q_xyz").as_double();
    tracker_->ekf_->q_yaw_ = this->get_parameter("q_yaw").as_double();
    tracker_->ekf_->q_r_ = this->get_parameter("q_r").as_double();
    tracker_->ekf_->r_xyz_ = this->get_parameter("r_xyz").as_double();
    tracker_->ekf_->r_yaw_ = this->get_parameter("r_yaw").as_double();

    std::cout << "q_xyz: " << tracker_->ekf_->q_xyz_ << std::endl;
    std::cout << "q_yaw: " << tracker_->ekf_->q_yaw_ << std::endl;
    std::cout << "q_r: " << tracker_->ekf_->q_r_ << std::endl;
    std::cout << "r_xyz: " << tracker_->ekf_->r_xyz_ << std::endl;
    std::cout << "r_yaw: " << tracker_->ekf_->r_yaw_ << std::endl;

    // armor ekf
    this->declare_parameter("a_q_xyz", 20.0);
    this->declare_parameter("a_r_xyz", 0.002);

    tracker_->armor_ekf_->q_xyz_ = this->get_parameter("a_q_xyz").as_double();
    tracker_->armor_ekf_->r_xyz_ = this->get_parameter("a_r_xyz").as_double();

    std::cout << "a_q_xyz: " << tracker_->armor_ekf_->q_xyz_ << std::endl;
    std::cout << "a_r_xyz: " << tracker_->armor_ekf_->r_xyz_ << std::endl;

    // Subscriber with tf2 message_filter
    // tf2 relevant
    tf2_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    // Create the timer interface before call to waitForTransform,
    // to avoid a tf2_ros::CreateTimerInterfaceException exception
    auto timer_interface = std::make_shared<tf2_ros::CreateTimerROS>(
        this->get_node_base_interface(), this->get_node_timers_interface());
    tf2_buffer_->setCreateTimerInterface(timer_interface);
    tf2_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf2_buffer_);
    // subscriber and filter
    armor_sub_.subscribe(this, "/detector/armor", rmw_qos_profile_sensor_data);
    target_frame_ = this->declare_parameter("target_frame", "odom");
    tf2_filter_ = std::make_shared<tf2_filter>(
        armor_sub_, *tf2_buffer_, target_frame_, 10, this->get_node_logging_interface(),
        this->get_node_clock_interface(), std::chrono::duration<int>(1));
    // Register a callback with tf2_ros::MessageFilter to be called when transforms are available
    tf2_filter_->registerCallback(&ArmorTrackerNode::armorCallback, this);

    // pub
    target_pub_ = this->create_publisher<rm_msgs::msg::Target>("/tracker/target", 10);

    tracker_start_ = std::chrono::steady_clock::now();
    tracker_end_ = std::chrono::steady_clock::now();
    tracker_fps_ = 0;
    tracker_now_fps_ = 0;

}

void ArmorTrackerNode::armorCallback(const rm_msgs::msg::Armor::SharedPtr armor_msg)
{
    tracker_end_ = std::chrono::steady_clock::now();

    std::chrono::duration<double> tracker_diff = tracker_end_ - tracker_start_;

    if(tracker_diff.count() >= 1)
    {
        std::cout << tracker_diff.count() << "s and tracker receive fps: " << tracker_fps_<< std::endl;
        tracker_now_fps_ = tracker_fps_;
        tracker_start_ = std::chrono::steady_clock::now();
        tracker_fps_ = 0;
    }

    tracker_fps_++;

    geometry_msgs::msg::PoseStamped ps;
    ps.header = armor_msg->header;
    ps.pose = armor_msg->pose;
    try
    {
        armor_msg->pose = tf2_buffer_->transform(ps, target_frame_).pose;
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
        return;
    }
    
    if(armor_msg->color != 2)
    {
        if(armor_msg->id != tracker_->tracker_armor_->armor_id)
        {
            tracker_->tracker_init(armor_msg);
        }
        else
        {
            rclcpp::Time current_time = armor_msg->header.stamp;
            dt_ = (current_time - last_time_).seconds();
            last_time_ = current_time;
            tracker_->ekf_->dt_ = dt_;
            tracker_->armor_ekf_->dt_ = dt_;
            tracker_->lost_thres_ = static_cast<int>(lost_time_thres_ / dt_);
            tracker_->tracker_update(armor_msg);
        }
    }
    else
    {
        if(tracker_->tracker_armor_->status != "LOST")
        {
            rclcpp::Time current_time = armor_msg->header.stamp;
            dt_ = (current_time - last_time_).seconds();
            last_time_ = current_time;
            tracker_->ekf_->dt_ = dt_;
            tracker_->armor_ekf_->dt_ = dt_;
            tracker_->lost_thres_ = static_cast<int>(lost_time_thres_ / dt_);
            tracker_->tracker_update(armor_msg);
        }
    }

    // publish target
    rm_msgs::msg::Target target_msg;
    target_msg.header.stamp = armor_msg->header.stamp;
    target_msg.header.frame_id = target_frame_;
    if(tracker_->tracker_armor_->status == "TRACKING")
    {
        target_msg.tracking = true;
        target_msg.id = tracker_->tracker_armor_->armor_name;
        target_msg.armor_num = tracker_->tracker_armor_->armor_num;
        target_msg.armor_position.position.x = tracker_->armor_target_state_.at<double>(0,0);
        target_msg.armor_position.position.y = tracker_->armor_target_state_.at<double>(2,0);
        target_msg.armor_velocity.x = tracker_->armor_target_state_.at<double>(1,0);
        target_msg.armor_velocity.y = tracker_->armor_target_state_.at<double>(3,0);
        target_msg.car_position.position.x = tracker_->target_state_.at<double>(0,0);
        target_msg.car_position.position.y = tracker_->target_state_.at<double>(2,0);
        target_msg.car_position.position.z = tracker_->target_state_.at<double>(4,0);
        target_msg.car_velocity.x = tracker_->target_state_.at<double>(1,0);
        target_msg.car_velocity.y = tracker_->target_state_.at<double>(3,0);
        target_msg.car_velocity.z = tracker_->target_state_.at<double>(5,0);
        target_msg.yaw = tracker_->target_state_.at<double>(6,0);
        target_msg.v_yaw = tracker_->target_state_.at<double>(7,0);
        target_msg.radius_1 = tracker_->target_state_.at<double>(8,0);
        target_msg.radius_2 = tracker_->another_r_;
        target_msg.dz = tracker_->dz_;

        target_msg.c_to_a_pitch = armor_msg->c_to_a_pitch;

        target_msg.is_repeat = armor_msg->is_repeat;

        target_pub_->publish(target_msg);
    }
    else
    {
        target_msg.tracking = false;
        target_msg.id = "";
        target_msg.armor_num = 0;
        target_msg.armor_position.position.x = 0;
        target_msg.armor_position.position.y = 0;
        target_msg.armor_velocity.x = 0;
        target_msg.armor_velocity.y = 0;
        target_msg.armor_position.position.x = 0;
        target_msg.armor_position.position.y = 0;
        target_msg.armor_position.position.z = 0;
        target_msg.car_velocity.x = 0;
        target_msg.car_velocity.y = 0;
        target_msg.car_velocity.z = 0;
        target_msg.yaw = 0;
        target_msg.v_yaw = 0;
        target_msg.radius_1 = 0;
        target_msg.radius_2 = 0;
        target_msg.dz = 0;
        target_msg.c_to_a_pitch = 0;
        target_pub_->publish(target_msg);
    }

    if(is_debug_ == true)
    {
        debug_deal();
    }
}

//--------------------------------------------------------------------------------------------------
// debug
void ArmorTrackerNode::create_debug_publishers()
{
    // param
    debug_param_ = std::make_shared<DebugParam>();
    debug_param_->draw_count = 0;
    debug_param_->image_width = 1000;
    debug_param_->image_height = 1000;

    double width = debug_param_->image_width;
    double height = debug_param_->image_height;
    debug_param_->last_armor_x = width / 2;
    debug_param_->last_armor_y = height / 2;
    debug_param_->last_armor_x_v = width / 2;
    debug_param_->last_armor_y_v = height / 2;
    debug_param_->last_car_x = width / 2;
    debug_param_->last_car_y = height / 2;
    debug_param_->last_car_x_v = width / 2;
    debug_param_->last_car_y_v = height / 2;
    debug_param_->last_yaw = width / 2;
    debug_param_->last_yaw_v = width / 2;

    cv::namedWindow("EKF Simulation", cv::WINDOW_AUTOSIZE);
    debug_image_ = cv::Mat::zeros(width, height, CV_8UC3);

    // sub
    detector_result_image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
        "/detector/result_image", 10, std::bind(&ArmorTrackerNode::ResultImageCallback, this, std::placeholders::_1));
    
    // pub
    tracker_result_image_pub_ = image_transport::create_publisher(this, "/tracker/result_image");
}

void ArmorTrackerNode::destroy_debug_publishers()
{
    // param
    debug_param_ = nullptr;

    // sub
    detector_result_image_sub_.reset();

    // pub
    tracker_result_image_pub_.shutdown();
}

void ArmorTrackerNode::ResultImageCallback(const sensor_msgs::msg::Image::SharedPtr msg)
{
    if(is_debug_ == true)
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
        cv::Mat image = cv_ptr->image;
        cv::putText(image, tracker_->tracker_armor_->status, cv::Point(10, 200), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 3);
        cv::putText(image, tracker_->tracker_armor_->armor_name, cv::Point(10, 150), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 3);
        cv::putText(image, std::to_string(tracker_->tracker_armor_->lost_count), cv::Point(10, 250), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 3);
        // 绘制时间戳
        builtin_interfaces::msg::Time image_stamp = msg->header.stamp;
        int64_t image_time = image_stamp.sec * 1000LL + image_stamp.nanosec / 1000000LL;
        cv::putText(image, "tracker image_stamp: " + std::to_string(image_time) + "ms", cv::Point(500, 70), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 3);
        cv_bridge::CvImage cv_image;
        cv_image.image = image;
        cv_image.encoding = "bgr8";
        cv_image.header = msg->header;
        tracker_result_image_pub_.publish(cv_image.toImageMsg());
    }
}

void ArmorTrackerNode::debug_deal()
{
    if(tracker_->tracker_armor_->status == "TRACKING")
    {
        //chansform
        double armor_x = tracker_->armor_target_state_.at<double>(0,0) * 50 + 500;
        double armor_y = - tracker_->armor_target_state_.at<double>(2,0) * 50 + 500;
        double armor_x_v = tracker_->armor_target_state_.at<double>(1,0) * 50;
        double armor_y_v = - tracker_->armor_target_state_.at<double>(3,0) * 50;

        double car_x = tracker_->target_state_.at<double>(0,0) * 50 + 500;
        double car_y = - tracker_->target_state_.at<double>(2,0) * 50 + 500;
        double car_x_v = tracker_->target_state_.at<double>(1,0) * 50;
        double car_y_v = - tracker_->target_state_.at<double>(3,0) * 50;
        double yaw = tracker_->target_state_.at<double>(6,0);
        double yaw_v = tracker_->target_state_.at<double>(7,0);
        double r = tracker_->target_state_.at<double>(8,0);

        double armor_car_x = (tracker_->target_state_.at<double>(0,0) - cos(yaw) * r) * 50 + 500;
        double armor_car_y = - (tracker_->target_state_.at<double>(2,0) - sin(yaw) * r) * 50 + 500;

        // 绘画标准
        // BGR
        // 白色：速度，由一条直线表示 || 中心位置
        // 红色：车辆位置，连线轨迹
        // 蓝色：装甲板位置， 连线轨迹
        // 绿色：由yaw和半径得到的装甲板位置
        cv::circle(debug_image_, cv::Point(debug_param_->image_width, debug_param_->image_height), 2, cv::Scalar(255, 255, 255), -1);

        cv::line(debug_image_, cv::Point(debug_param_->last_car_x, debug_param_->last_car_y), cv::Point(car_x, car_y), cv::Scalar(0, 0, 255), 2);
        cv::line(debug_image_, cv::Point(car_x, car_y), cv::Point(armor_car_x, armor_car_y), cv::Scalar(0, 255, 0), 2);
        if(tracker_->tracker_armor_->armor_num == 4)
        {
            cv::line(debug_image_, cv::Point((tracker_->target_state_.at<double>(0,0) - cos(yaw + M_PI / 2) * r) * 50 + 500, - (tracker_->target_state_.at<double>(2,0) - sin(yaw + M_PI / 2) * r) * 50 + 500), cv::Point(car_x, car_y), cv::Scalar(0, 255, 0), 2);
            cv::line(debug_image_, cv::Point((tracker_->target_state_.at<double>(0,0) - cos(yaw + M_PI) * r) * 50 + 500, - (tracker_->target_state_.at<double>(2,0) - sin(yaw + M_PI) * r) * 50 + 500), cv::Point(car_x, car_y), cv::Scalar(0, 255, 0), 2);
            cv::line(debug_image_, cv::Point((tracker_->target_state_.at<double>(0,0) - cos(yaw + M_PI / 2 * 3) * r) * 50 + 500, - (tracker_->target_state_.at<double>(2,0) - sin(yaw + M_PI / 2 * 3) * r) * 50 + 500), cv::Point(car_x, car_y), cv::Scalar(0, 255, 0), 2);
        }
        else if(tracker_->tracker_armor_->armor_num == 3)
        {
            cv::line(debug_image_, cv::Point((tracker_->target_state_.at<double>(0,0) - cos(yaw + M_PI / 3 * 2) * r) * 50 + 500, - (tracker_->target_state_.at<double>(2,0) - sin(yaw + M_PI / 3 * 2) * r) * 50 + 500), cv::Point(car_x, car_y), cv::Scalar(0, 255, 0), 2);
            cv::line(debug_image_, cv::Point((tracker_->target_state_.at<double>(0,0) - cos(yaw + M_PI / 3 * 4) * r) * 50 + 500, - (tracker_->target_state_.at<double>(2,0) - sin(yaw + M_PI / 3 * 4) * r) * 50 + 500), cv::Point(car_x, car_y), cv::Scalar(0, 255, 0), 2);
        }
        else if(tracker_->tracker_armor_->armor_num == 2)
        {
            cv::line(debug_image_, cv::Point((tracker_->target_state_.at<double>(0,0) - cos(yaw + M_PI) * r) * 50 + 500, - (tracker_->target_state_.at<double>(2,0) - sin(yaw + M_PI) * r) * 50 + 500), cv::Point(car_x, car_y), cv::Scalar(0, 255, 0), 2);
        }
        cv::circle(debug_image_, cv::Point(armor_x, armor_y), 2, cv::Scalar(0, 255, 0), -1);

        cv::line(debug_image_, cv::Point(car_x, car_y), cv::Point(car_x + car_x_v, car_y), cv::Scalar(255, 255, 255), 2);
        cv::line(debug_image_, cv::Point(car_x, car_y), cv::Point(car_x, car_y + car_y_v), cv::Scalar(255, 255, 255), 2);

        cv::line(debug_image_, cv::Point(debug_param_->last_armor_x, debug_param_->last_armor_y), cv::Point(armor_x, armor_y), cv::Scalar(255, 0, 0), 2);

        cv::line(debug_image_, cv::Point(armor_x, armor_y), cv::Point(armor_x + armor_x_v, armor_y), cv::Scalar(255, 255, 255), 2);
        cv::line(debug_image_, cv::Point(armor_x, armor_y), cv::Point(armor_x, armor_y + armor_y_v), cv::Scalar(255, 255, 255), 2);

        debug_param_->last_armor_x = armor_x;
        debug_param_->last_armor_y = armor_y;
        debug_param_->last_armor_x_v = armor_x_v;
        debug_param_->last_armor_y_v = armor_y_v;
        debug_param_->last_car_x = car_x;
        debug_param_->last_car_y = car_y;
        debug_param_->last_car_x_v = car_x_v;
        debug_param_->last_car_y_v = car_y_v;
        debug_param_->last_yaw = yaw;
        debug_param_->last_yaw_v = yaw_v;

        cv::imshow("EKF Simulation", debug_image_);
        cv::waitKey(1);

        debug_param_->draw_count++;
        if(debug_param_->draw_count > 1)
        {
            debug_image_ = cv::Mat::zeros(debug_param_->image_width, debug_param_->image_height, CV_8UC3);
            debug_param_->draw_count = 0;
        }
    }
}
//--------------------------------------------------------------------------------------------------


}  // namespace rm_armor_tracker

#include "rclcpp_components/register_node_macro.hpp"

RCLCPP_COMPONENTS_REGISTER_NODE(rm_armor_tracker::ArmorTrackerNode)