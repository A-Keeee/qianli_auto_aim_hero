#include "detector_node.hpp"

namespace rm_armor_detector
{

ArmorDetectorNode::ArmorDetectorNode(const rclcpp::NodeOptions & options) : Node("armor_detector_node", options)
{
    RCLCPP_INFO(this->get_logger(), "Starting ArmorDetectorNode!");
    robots_init();
    parameters_init();
    //test();
}

void ArmorDetectorNode::parameters_init()
{
    RCLCPP_INFO(this->get_logger(), "Begin to init parameters!");
    this->declare_parameter("detector_debug", 0);
    is_debug_ = this->get_parameter("detector_debug").as_int();
    std::cout << "is_detector_debug: " << is_debug_ << std::endl;
    if(is_debug_ == true)
    {
        create_debug_publishers();
    }
    // Debug param change moniter
    debug_param_sub_ = std::make_shared<rclcpp::ParameterEventHandler>(this);
    debug_cb_handle_ =
    debug_param_sub_->add_parameter_callback("detector_debug", [this](const rclcpp::Parameter & p) {
      is_debug_ = p.as_bool();
      is_debug_ ? create_debug_publishers() : destroy_debug_publishers();
    });

    #ifdef USE_CUDA_DETCTOR
        RCLCPP_INFO(this->get_logger(), "Cuda detect mode!");
        auto pkg_path = ament_index_cpp::get_package_share_directory("rm_armor_detector");
        DL_INIT_PARAM params;
        params.rectConfidenceThreshold = 0.1;
        params.iouThreshold = 0.5;
        params.modelPath = pkg_path + "/model/four_points_armor/armor.onnx";
        params.imgSize = { 640, 640 };
        params.cudaEnable = true;
        cuda_detector_ = std::make_shared<CudaDetector>();
        char* ret = cuda_detector_->CreateSession(params);
        if (ret != RET_OK)
        {
            RCLCPP_ERROR(this->get_logger(), "CreateSession failed: %s", ret);
        }
    #elif defined(USE_OPENVINO_DETCTOR)
        RCLCPP_INFO(this->get_logger(), "Openvino detect mode!");
        auto pkg_path = ament_index_cpp::get_package_share_directory("rm_armor_detector");
        auto model_path = pkg_path + "/model/four_points_armor/armor.onnx";
        openvino_detector_ = std::make_shared<OpenvinoDetector>();
        openvino_detector_->set_onnx_model(model_path, "GPU");
    #else
        RCLCPP_INFO(this->get_logger(), "number detect mode!");
        lights_detector_ = initDetector();
    #endif

    this->declare_parameter("MIN_BIG_ARMOR_RATIO", 3.2);
    MIN_BIG_ARMOR_RATIO_ = this->get_parameter("MIN_BIG_ARMOR_RATIO").as_double();
    std::cout << "MIN_BIG_ARMOR_RATIO: " << MIN_BIG_ARMOR_RATIO_ << std::endl;

    this->declare_parameter("detect_color", 2);
    detect_color_ = this->get_parameter("detect_color").as_int();
    std::cout << "detect_color: " << detect_color_ << std::endl;

    this->declare_parameter("priority_armors", std::vector<int>());
    auto priority_armors_long = this->get_parameter("priority_armors").as_integer_array();
    std::vector<int> priority_armors;
    std::transform(priority_armors_long.begin(), priority_armors_long.end(), std::back_inserter(priority_armors),
                    [](long int val) { return static_cast<int>(val); });
    if(priority_armors.size() != decision_armors_.size())
    {
        RCLCPP_ERROR(this->get_logger(), "priority_armors size is != decision_armors_ size !");
        return;
    }
    for(size_t i = 0;i<decision_armors_.size();i++)
    {
        decision_armors_[i].priority = priority_armors[i];
    }
    for(size_t i = 0;i<decision_armors_.size();i++)
    {
        std::cout << "priority_armors: " << decision_armors_[i].priority << std::endl;
    }

    this->declare_parameter("ignore_armors", std::vector<int>());
    auto ignore_armors_long = this->get_parameter("ignore_armors").as_integer_array();
    ignore_armors_.clear();
    std::transform(ignore_armors_long.begin(), ignore_armors_long.end(), std::back_inserter(ignore_armors_),
                    [](long int val) { return static_cast<int>(val); });
    for(size_t i = 0;i<ignore_armors_.size();i++)
    {
        std::cout << "ignore_armors: " << ignore_armors_[i] << std::endl;
    }

    armor_pub_ = this->create_publisher<rm_msgs::msg::Armor>("/detector/armor", 10);

    camera_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
        "/camera_info", rclcpp::SensorDataQoS(), std::bind(&ArmorDetectorNode::camera_info_callback, this, std::placeholders::_1));

    is_rune_ = true; // 默认为打符模式，才能刷新接受者接受图片消息, 即开始默认自己不能接受图片消息
    status_sub_ = this->create_subscription<rm_msgs::msg::Status>(
        "/status", rclcpp::SensorDataQoS(), std::bind(&ArmorDetectorNode::status_callback, this, std::placeholders::_1));

    detector_start_ = std::chrono::steady_clock::now();
    detector_end_ = std::chrono::steady_clock::now();
    detector_fps_ = 0;
    detector_now_fps_ = 0;

    last_decision_armor_image_point_ = cv::Point2d(0,0);
    last_decision_armor_id_ = -1;

    // Subscriber with tf2 message_filter
    // tf2 relevant
    tf2_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    // Create the timer interface before call to waitForTransform,
    // to avoid a tf2_ros::CreateTimerInterfaceException exception
    auto timer_interface = std::make_shared<tf2_ros::CreateTimerROS>(
        this->get_node_base_interface(), this->get_node_timers_interface());
    tf2_buffer_->setCreateTimerInterface(timer_interface);
    tf2_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf2_buffer_);

    RCLCPP_INFO(this->get_logger(), "Finished init parameters successfully!");
}

std::unique_ptr<LightsDetector> ArmorDetectorNode::initDetector()
{
  rcl_interfaces::msg::ParameterDescriptor param_desc;
  param_desc.integer_range.resize(1);
  param_desc.integer_range[0].step = 1;
  param_desc.integer_range[0].from_value = 0;
  param_desc.integer_range[0].to_value = 255;
  int binary_thres = declare_parameter("binary_thres", 160, param_desc);

  LightsDetector::LightParams l_params;
    l_params.min_ratio = declare_parameter("light.min_ratio", 0.1);
    l_params.max_ratio = declare_parameter("light.max_ratio", 0.4);
    l_params.max_angle = declare_parameter("light.max_angle", 40.0);

  LightsDetector::ArmorParams a_params;
    a_params.min_light_ratio = declare_parameter("armor.min_light_ratio", 0.7);
    a_params.min_small_center_distance = declare_parameter("armor.min_small_center_distance", 0.8);
    a_params.max_small_center_distance = declare_parameter("armor.max_small_center_distance", 3.2);
    a_params.min_large_center_distance = declare_parameter("armor.min_large_center_distance", 3.2);
    a_params.max_large_center_distance = declare_parameter("armor.max_large_center_distance", 5.5);
    a_params.max_angle = declare_parameter("armor.max_angle", 35.0);

    if(detect_color_ == 0)
    {
        detect_color_ = 1;
    }
    else if(detect_color_ == 1)
    {
        detect_color_ = 0;
    }

  auto detector = std::make_unique<LightsDetector>(binary_thres, detect_color_, l_params, a_params);

  // Init classifier
  auto pkg_path = ament_index_cpp::get_package_share_directory("rm_armor_detector");
  auto model_path = pkg_path + "/model/light_armor/mlp.onnx";
  auto label_path = pkg_path + "/model/light_armor/label.txt";
  double threshold = this->declare_parameter("classifier_threshold", 0.7);
  std::vector<std::string> ignore_classes =
    this->declare_parameter("ignore_classes", std::vector<std::string>{"negative"});
  detector->classifier =
    std::make_unique<NumberClassifier>(model_path, label_path, threshold, ignore_classes);

  return detector;
}

void ArmorDetectorNode::create_debug_publishers()
{
    result_image_pub_ = image_transport::create_publisher(this, "/detector/result_image");
}

void ArmorDetectorNode::destroy_debug_publishers()
{
    result_image_pub_.shutdown();
}

void ArmorDetectorNode::debug_deal(const cv::Mat &image, const std_msgs::msg::Header& image_header, const std::vector<Armor> &armors, const DecisionArmor &decision_armor)
{
    if(is_debug_ == true)
    {
        cv::Mat debug_image = image.clone();
        // 绘制识别到的装甲板框框
        for(size_t i =0;i<armors.size();i++)
        {
            // Convert vector<cv::Point2f> to vector<vector<cv::Point>>
            std::vector<std::vector<cv::Point>> pts(1);
            for (const auto& p : armors[i].four_points) {
                pts[0].emplace_back(cv::Point(p.x, p.y));
            }
            if(armors[i].id == decision_armor.id)
            {
                if(pred_points_.size() == 4)
                {
                    cv::putText(debug_image, "1", cv::Point(pred_points_[0].x, pred_points_[0].y), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 3);
                    cv::putText(debug_image, "2", cv::Point(pred_points_[1].x, pred_points_[1].y), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 3);
                    cv::putText(debug_image, "3", cv::Point(pred_points_[2].x, pred_points_[2].y), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 3);
                    cv::putText(debug_image, "4", cv::Point(pred_points_[3].x, pred_points_[3].y), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 3);
                }
                cv::putText(debug_image, "1", cv::Point(armors[i].four_points[0].x, armors[i].four_points[0].y), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 3);
                cv::putText(debug_image, "2", cv::Point(armors[i].four_points[1].x, armors[i].four_points[1].y), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 3);
                cv::putText(debug_image, "3", cv::Point(armors[i].four_points[2].x, armors[i].four_points[2].y), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 3);
                cv::putText(debug_image, "4", cv::Point(armors[i].four_points[3].x, armors[i].four_points[3].y), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 3);
                cv::polylines(debug_image, pts, true, cv::Scalar(0, 0, 255), 2);
                cv::Point armor_center = cv::Point((armors[i].four_points[0].x + armors[i].four_points[2].x) / 2, (armors[i].four_points[0].y + armors[i].four_points[2].y) / 2);
                //绘制距离
                cv::putText(debug_image, std::to_string(decision_armor.distance), armor_center, cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 3);
                //绘制yaw值
                double yaw = orientationToYaw(decision_armor.pose.orientation);
                cv::putText(debug_image, std::to_string(yaw * 57.3f), armor_center + cv::Point(0, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 3);
                double pred_yaw = decision_armor.yaw;
                cv::putText(debug_image, std::to_string(pred_yaw * 57.3f), armor_center + cv::Point(0, 60), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 3);
            }
            else
            {
                cv::polylines(debug_image, pts, true, cv::Scalar(0, 255, 0), 2);
            }
            cv::putText(debug_image, armors[i].name, cv::Point(armors[i].rect.x, armors[i].rect.y), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 3);
        }
        // 绘制图片大小以及图片中心
        cv::putText(debug_image, "width: " + std::to_string(debug_image.cols) + " height: " + std::to_string(debug_image.rows), cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 3);
        cv::circle(debug_image, image_center_, 3, cv::Scalar(0, 255, 0), -1);
        // 绘制fps
        cv::putText(debug_image, "detector fps: " + std::to_string(detector_now_fps_), cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 3);
        // 绘制时间戳
        builtin_interfaces::msg::Time image_stamp = image_header.stamp;
        int64_t image_time = image_stamp.sec * 1000LL + image_stamp.nanosec / 1000000LL;
        cv::putText(debug_image, "detector image_stamp: " + std::to_string(image_time) + "ms", cv::Point(500, 50), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 3);
        sensor_msgs::msg::Image debug_image_msg = *(cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", debug_image).toImageMsg());
        debug_image_msg.header = image_header;
        result_image_pub_.publish(debug_image_msg);
    }
}

void ArmorDetectorNode::camera_info_callback(const sensor_msgs::msg::CameraInfo::SharedPtr msg)
{
    RCLCPP_INFO(this->get_logger(), "Begin to receive camera info!");
    camera_width_ = msg->width;
    camera_height_ = msg->height;
    camera_matrix_ = cv::Mat(3, 3, CV_64F, (void *)msg->k.data()).clone();
    distortion_coefficients_ = cv::Mat(1, 5, CV_64F, (void *)msg->d.data()).clone();
    image_center_ = cv::Point2f(camera_width_ / 2, camera_height_ / 2);
    
    std::cout << "camera_width: " << camera_width_ << std::endl;
    std::cout << "camera_height: " << camera_height_ << std::endl;
    std::cout << "camera_matrix: " << camera_matrix_ << std::endl;
    std::cout << "distortion_coefficients: " << distortion_coefficients_ << std::endl;
    std::cout << "image_center: " << image_center_ << std::endl;

    pnp_solver_ = std::make_shared<PnpSolver>();
    pnp_solver_->set_matrix(camera_width_, camera_height_, camera_matrix_, distortion_coefficients_);

    projection_yaw_ = std::make_shared<ProjectionYaw>();
    projection_yaw_->set_matrix(camera_matrix_, distortion_coefficients_);

    camera_info_sub_.reset();
    RCLCPP_INFO(this->get_logger(), "Finished receive camera info successfully!");
}

void ArmorDetectorNode::status_callback(const rm_msgs::msg::Status::SharedPtr msg)
{
    if(msg->is_rune == is_rune_)
    {
        return;
    }
    is_rune_ = msg->is_rune;
    if(is_rune_ == true)
    {
        image_sub_.reset();
    }
    else
    {
        image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/image_raw", rclcpp::SensorDataQoS(), std::bind(&ArmorDetectorNode::image_callback, this, std::placeholders::_1));

        detector_start_ = std::chrono::steady_clock::now();
    }
}

void ArmorDetectorNode::image_callback(const sensor_msgs::msg::Image::SharedPtr msg)
{
    detector_end_ = std::chrono::steady_clock::now();

    std::chrono::duration<double> detector_diff = detector_end_ - detector_start_;

    if(detector_diff.count() >= 1)
    {
        std::cout << detector_diff.count() << "s and detector receive fps: " << detector_fps_<< std::endl;
        detector_now_fps_ = detector_fps_;
        detector_start_ = std::chrono::steady_clock::now();
        detector_fps_ = 0;
    }

    detector_fps_++;

    // 1. Convert ROS image message to OpenCV image
    cv::Mat image = cv_bridge::toCvCopy(msg, "bgr8")->image;
    // 2. Detect armors
    #ifdef USE_CUDA_DETCTOR
        if(cuda_detector_ != nullptr)
        {
            cuda_detector_->infer(image, detect_color_);
            // 3. Updata robots
            get_robots(decision_armors_, cuda_detector_->armors_);
            allrobots_adjust(decision_armors_);
        }
        else
        {
            RCLCPP_ERROR(this->get_logger(), "Cuda detector is nullptr!");
        }
    #elif defined(USE_OPENVINO_DETCTOR)
        if(openvino_detector_ != nullptr)
        {
            openvino_detector_->infer(image, detect_color_);
            // 3. Updata robots
            get_robots(decision_armors_, openvino_detector_->armors_);
            allrobots_adjust(decision_armors_);
        }
        else
        {
            RCLCPP_ERROR(this->get_logger(), "Openvino detector is nullptr!");
        }
    #else
        if(lights_detector_ != nullptr)
        {
            std::vector<Armor> armors;
            lights_detector_->detect(image);
            for(size_t i = 0;i<lights_detector_->armors_.size();i++)
            {
                Armor armor;
                armor.id = lights_detector_->armors_[i].id;
                armor.color = lights_detector_->armors_[i].color;
                armor.number_score = lights_detector_->armors_[i].number_score;
                armor.color_score = lights_detector_->armors_[i].color_score;
                armor.name = lights_detector_->armors_[i].name;
                armor.four_points = lights_detector_->armors_[i].four_points;
                armor.rect = lights_detector_->armors_[i].rect;
                armors.push_back(armor);
            }
            // 3. Updata robots
            get_robots(decision_armors_, armors);
            allrobots_adjust(decision_armors_);
        }
        else
        {
            RCLCPP_ERROR(this->get_logger(), "Light detector is nullptr!");
        }
    #endif

    // 4. Decide which armor to shoot
    DecisionArmor decision_armor;
    decision_armor.color = 2;
    decision_armor = decide_armor_shoot(decision_armors_);

    if(decision_armor.color != 2)
    {
        // 5. Solve PnP
        if(pnp_solver_ != nullptr && projection_yaw_ != nullptr)
        {
            cv::Mat rvec, tvec;
            bool success = pnp_solver_->solve_pnp(decision_armor.four_points, rvec, tvec, decision_armor.is_big_armor);
            if(success == true)
            {
                decision_armor.distance = std::sqrt(tvec.at<double>(0) * tvec.at<double>(0) + tvec.at<double>(1) * tvec.at<double>(1) + tvec.at<double>(2) * tvec.at<double>(2));

                rm_msgs::msg::Armor armor_msg;
                armor_msg.header = msg->header;

                // 采用传统pnp方法计算yaw
                // rvec to 3x3 rotation matrix
                cv::Mat rotation_matrix;
                cv::Rodrigues(rvec, rotation_matrix);
                // rotation matrix to quaternion
                tf2::Matrix3x3 tf2_rotation_matrix(
                rotation_matrix.at<double>(0, 0), rotation_matrix.at<double>(0, 1),
                rotation_matrix.at<double>(0, 2), rotation_matrix.at<double>(1, 0),
                rotation_matrix.at<double>(1, 1), rotation_matrix.at<double>(1, 2),
                rotation_matrix.at<double>(2, 0), rotation_matrix.at<double>(2, 1),
                rotation_matrix.at<double>(2, 2));
                tf2::Quaternion tf2_q;
                tf2_rotation_matrix.getRotation(tf2_q);
                armor_msg.pose.orientation = tf2::toMsg(tf2_q);

                // tvec to translation
                armor_msg.pose.position.x = tvec.at<double>(0);
                armor_msg.pose.position.y = tvec.at<double>(1);
                armor_msg.pose.position.z = tvec.at<double>(2);

                decision_armor.header = msg->header;

                decision_armor.pose = armor_msg.pose;

                projection_yaw_->update_tf2_buffer(tf2_buffer_, tf2_listener_);
                double yaw = projection_yaw_->get_yaw(decision_armor);
                pred_points_ = projection_yaw_->get_pred_points(decision_armor, projection_yaw_->PITCH_, yaw);
                decision_armor.yaw = yaw;
                
                // 采用投影方法计算yaw
                double roll = 15.0 * CV_PI / 180.0;
                double pitch = - CV_PI  - yaw; // 假设yaw已经定义
                double trans_yaw = 0.0; // 这里的yaw是绕z轴的旋转
                tf2::Quaternion q_roll, q_pitch, q_yaw;
                // 设置四元数值
                q_roll.setRPY(roll, 0.0, 0.0);
                q_pitch.setRPY(0.0, pitch, 0.0);
                q_yaw.setRPY(0.0, 0.0, trans_yaw);
                // 按照特定的顺序组合四元数
                tf2::Quaternion q_combined = q_yaw * q_pitch * q_roll;
                armor_msg.pose.orientation = tf2::toMsg(q_combined); // 先饶z轴yaw，再绕y轴pitch，最后绕x轴roll

                armor_msg.c_to_a_pitch = pitch;

                armor_msg.id = decision_armor.id;
                armor_msg.color = decision_armor.color;
                if(armor_msg.id == 0)
                {
                    armor_msg.name = "sentry";
                }
                else if(armor_msg.id >= 1 && armor_msg.id <= 5)
                {
                    armor_msg.name = std::to_string(armor_msg.id);
                }
                else if(armor_msg.id == 6)
                {
                    armor_msg.name = "outpost";
                }
                else if(armor_msg.id == 7)
                {
                    armor_msg.name = "base";
                }
                else if(armor_msg.id == 8)
                {
                    armor_msg.name = "base_big";
                }
                decision_armor.is_big_armor == true ? armor_msg.type = "BIG" : armor_msg.type = "SMALL";
                armor_msg.distance_to_image_center = decision_armor.distance_to_image_center;
                armor_msg.is_repeat = is_repeat;
                armor_pub_->publish(armor_msg);
            }
            else
            {
                RCLCPP_ERROR(this->get_logger(), "Pnp solver failed!");
            }
        }
        else
        {
            RCLCPP_ERROR(this->get_logger(), "Pnp solver or projection solver is nullptr!");
        }
    }
    else // 未识别到有效装甲板，也需要发布消息，刷新tracker
    {
        rm_msgs::msg::Armor armor_msg;
        armor_msg.header = msg->header;
        armor_msg.id = 100;
        armor_msg.color = 2;
        armor_msg.name = "none";
        armor_msg.type = "none";
        armor_msg.distance_to_image_center = 0;
        armor_msg.c_to_a_pitch = 0;
        armor_pub_->publish(armor_msg);
    }
    if(is_debug_ == true)
    {
        #ifdef USE_CUDA_DETCTOR
            debug_deal(image, msg->header, cuda_detector_->armors_, decision_armor);
        #elif defined(USE_OPENVINO_DETCTOR)
            debug_deal(image, msg->header, openvino_detector_->armors_, decision_armor);
        #else
            std::vector<Armor> armors;
            lights_detector_->detect(image);
            for(size_t i = 0;i<lights_detector_->armors_.size();i++)
            {
                Armor armor;
                armor.id = lights_detector_->armors_[i].id;
                armor.color = lights_detector_->armors_[i].color;
                armor.number_score = lights_detector_->armors_[i].number_score;
                armor.color_score = lights_detector_->armors_[i].color_score;
                armor.name = lights_detector_->armors_[i].name;
                armor.four_points = lights_detector_->armors_[i].four_points;
                armor.rect = lights_detector_->armors_[i].rect;
                armors.push_back(armor);
            }
            debug_deal(image, msg->header, armors, decision_armor);
        #endif
    }
}

void ArmorDetectorNode::test()
{
    std::string model_path = "/home/zyicome/zyb/qianli_auto_aim/src/rm_armor_detector/model/four_points_armor/armor.onnx";
    //std::string model_path = "/home/zyicome/zyb/qianli_auto_aim/src/rm_armor_detector/model/inference_armor/armor.onnx";
    #ifdef USE_OPENVINO_DETCTOR
        OpenvinoDetector detector;
        detector.set_onnx_model(model_path, "CPU");
        cv::Mat input = cv::imread("/home/zyicome/zyb/pictures/armors/images/8.jpg");
        detector.infer(input, detect_color_);
        if(detector.armors_.size() > 0)
        {
            for(size_t i =0;i<detector.armors_.size();i++)
            {
                // Convert vector<cv::Point2f> to vector<vector<cv::Point>>
                std::vector<std::vector<cv::Point>> pts(1);
                for (const auto& p : detector.armors_[i].four_points) {
                    pts[0].emplace_back(cv::Point(p.x, p.y));
                }
                cv::polylines(input, pts, true, cv::Scalar(0, 255, 0), 2);
                cv::putText(input, detector.armors_[i].name, cv::Point(detector.armors_[i].rect.x, detector.armors_[i].rect.y), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 3);
            }
            cv::imshow("result", input);
            cv::waitKey(0);
        }
        else
        {
            std::cout << "No armor detected!" << std::endl;
        }
    #endif
}

// 用于将给定的四元数表示的姿态（朝向）转换为偏航角（yaw）
double ArmorDetectorNode::orientationToYaw(const geometry_msgs::msg::Quaternion & q)
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

//-----------------------------------------------------------
void ArmorDetectorNode::robots_init()
{
    RCLCPP_INFO(this->get_logger(), "Begin to init robots!");
    DecisionArmor decision_armor;
    decision_armor.is_big_armor = false;
    decision_armor.is_ignored = false;
    decision_armor.is_continue = false;
    decision_armor.id = 0;
    decision_armor.color = 2; //默认为灰色，即无效装甲板
    decision_armor.priority = 100;
    decision_armor.distance = 0;
    decision_armor.distance_to_image_center = 0;
    decision_armor.yaw = 0;
    decision_armor.four_points = {cv::Point(0, 0), cv::Point(0, 0), cv::Point(0, 0), cv::Point(0, 0)};
    decision_armor.pose.position.x = 0;
    decision_armor.pose.position.y = 0;
    decision_armor.pose.position.z = 0;
    decision_armor.pose.orientation.x = 0;
    decision_armor.pose.orientation.y = 0;
    decision_armor.pose.orientation.z = 0;
    for(int i = 0;i < 9; i++)
    {
        decision_armors_.push_back(decision_armor);
        decision_armor.id++;
    }
    decision_armors_[1].is_big_armor = true; // 英雄必定为大装甲板
    RCLCPP_INFO(this->get_logger(), "Finished init robots successfully!");
}

void ArmorDetectorNode::get_robots(std::vector<DecisionArmor> &decision_armors, const std::vector<Armor> &armors)
{
    bool is_big_armor;
    int armor_id;
    int armor_color;
    std::vector<cv::Point2f> armor_four_points;

    for(size_t i = 0; i<armors.size();i++)
    {
        armor_id = armors[i].id;
        armor_color = armors[i].color;
        armor_four_points = armors[i].four_points;
        cv::Point armor_center = cv::Point((armor_four_points[0].x + armor_four_points[2].x) / 2, (armor_four_points[0].y + armor_four_points[2].y) / 2);
        is_big_armor = get_is_big_armor(armor_four_points);
        if(get_is_ignored(armors[i], armor_four_points) == true)
        {
            continue;
        }
        if(decision_armors[armor_id].is_continue == true) // 重复装甲板出现
        {
            std::cout << "111" << std::endl;
            if(last_decision_armor_id_ == decision_armors[armor_id].id) // 并且有上一次的记录
            {
                // 比较谁离上一次的记录装甲板更相近，则选择哪个装甲板
                cv::Point now_decision_armor_center = cv::Point((decision_armors[armor_id].four_points[0].x + decision_armors[armor_id].four_points[2].x) / 2, (decision_armors[armor_id].four_points[0].y + decision_armors[armor_id].four_points[2].y) / 2);
                double distance_one = cv::norm(now_decision_armor_center - last_decision_armor_image_point_);
                double distance_two = cv::norm(armor_center - last_decision_armor_image_point_);
                std::cout << "distance_one: " << distance_one << std::endl;
                std::cout << "distance_two: " << distance_two << std::endl;
                if(distance_one < distance_two) // 如果原来的就比较好，就跳过
                {
                    continue;
                }
            }
        }
        decision_armors[armor_id].is_big_armor = is_big_armor;
        decision_armors[armor_id].is_ignored = false;
        decision_armors[armor_id].is_continue = true;
        decision_armors[armor_id].id = armor_id;
        decision_armors[armor_id].color = armor_color;
        decision_armors[armor_id].distance_to_image_center = pnp_solver_->get_distance_armor_center_to_image_center(armor_center);
        decision_armors[armor_id].four_points = armor_four_points;
    }
}

void ArmorDetectorNode::allrobots_adjust(std::vector<DecisionArmor> &decision_armors)
{
    for(size_t i =0;i<decision_armors.size();i++)
    {
        if(decision_armors[i].is_continue == false)
        {
            decision_armors[i].color = 2; //默认为灰色，即无效装甲板
            decision_armors[i].distance = 0;
            decision_armors[i].distance_to_image_center = 0;
            decision_armors[i].four_points = {cv::Point(0, 0), cv::Point(0, 0), cv::Point(0, 0), cv::Point(0, 0)};
        }

        decision_armors[i].is_continue = false;
    }
}

bool ArmorDetectorNode::get_is_big_armor(const std::vector<cv::Point2f> &four_points)
{
    double width = cv::norm(four_points[0] - four_points[1]);
    double height = cv::norm(four_points[1] - four_points[2]);
    double ratio = width / height;
    if(ratio > MIN_BIG_ARMOR_RATIO_)
    {
        return true;
    }
    else
    {
        return false;
    }
}

bool ArmorDetectorNode::get_is_ignored(const Armor &armor, const std::vector<cv::Point2f> &four_points)
{
    // 忽略类别判断
    if(ignore_armors_.size() != 0)
    {
        for(size_t i = 0; i<ignore_armors_.size(); i++)
        {
            if(armor.id == ignore_armors_[i])
            {
                return true;
            }
        }
    }
    double width = cv::norm(four_points[0] - four_points[1]);
    if(width == 0)
    {
        return true;
    }
    return false;
}

DecisionArmor ArmorDetectorNode::decide_armor_shoot(const std::vector<DecisionArmor> &decision_armors)
{
    DecisionArmor decision_armor;
    int min_priority = 100;
    int min_distance_to_image_center = 10000;
    int decision_distance_id = 0;
    int decision_priority_id = 0;
    is_repeat = false;
    int armor_number[9] = {0};
    std::vector<int> decision_ids;
    for(size_t i = 0;i<decision_armors.size();i++)
    {
        if(decision_armors[i].color != 2) // 表明该装甲板有效
        {
            if(decision_armors[i].priority < min_priority)
            {
                min_priority = decision_armors[i].priority;
                decision_priority_id = i;
            }
            else if(decision_armors[i].priority == min_priority && min_priority != 100)
            {
                decision_ids.push_back(decision_priority_id); //多插入不影响
                decision_ids.push_back(i);
            }
            if(decision_armors[i].distance_to_image_center < min_distance_to_image_center)
            {
                min_distance_to_image_center = decision_armors[i].distance_to_image_center;
                decision_distance_id = i;
            }
            armor_number[decision_armors[i].id]++;
            if(armor_number[decision_armors[i].id] >= 2 && last_decision_armor_id_ == decision_armors[i].id)
            {
                is_repeat = true;
            }
        }
    }
    if(decision_ids.size() != 0)
    {
        min_distance_to_image_center = 10000;
        decision_distance_id = 0;
        for(size_t i = 0;i<decision_ids.size();i++)
        {
            if(decision_armors[decision_ids[i]].distance_to_image_center < min_distance_to_image_center)
            {
                min_distance_to_image_center = decision_armors[decision_ids[i]].distance_to_image_center;
                decision_distance_id = decision_ids[i];
            }
        }
        decision_armor = decision_armors[decision_distance_id];
    }
    else if(min_priority != 100)
    {
        decision_armor = decision_armors[decision_priority_id];
    }
    else
    {
        decision_armor = decision_armors[decision_distance_id];
    }
    last_decision_armor_id_ = decision_armor.id;
    cv::Point now_decision_armor_center = cv::Point((decision_armor.four_points[0].x + decision_armor.four_points[2].x) / 2, (decision_armor.four_points[0].y + decision_armor.four_points[2].y) / 2);
    last_decision_armor_image_point_ = now_decision_armor_center;
    return decision_armor;
}

}  // namespace rm_armor_detector

#include "rclcpp_components/register_node_macro.hpp"

RCLCPP_COMPONENTS_REGISTER_NODE(rm_armor_detector::ArmorDetectorNode)