#include <iostream>

#include "opencv2/opencv.hpp"

#include "openvino/openvino.hpp"

#include "armor.hpp"

class OpenvinoDetector
{
public:
    OpenvinoDetector();
    void set_onnx_model(const std::string &model_path, const std::string &device);
    void infer(const cv::Mat &input, int detect_color);
    cv::Mat letterbox(const cv::Mat &input);
    double sigmoid(double x);

    ov::Core core_;
    ov::CompiledModel compiled_model_;
    ov::InferRequest infer_request_;

    const float IMAGE_WIDTH_ = 640;
    const float IMAGE_HEIGHT_ = 640;
    const float CONFIDENCE_THRESHOLD_ = 0.8;
    const float SCORE_THRESHOLD_ = 0.9;
    const float COLOR_THRESHOLD_ = 0.9;
    const float NMS_THRESHOLD_ = 0.5;
    const std::vector<std::string> class_names_ = {"sentry", "1", "2", "3", "4", "5", "outpost", "base", "base_big"};
    std::vector<Armor> armors_;
};