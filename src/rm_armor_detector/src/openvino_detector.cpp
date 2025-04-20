#include "openvino_detector.hpp"

OpenvinoDetector::OpenvinoDetector()
{
    std::cout << "OpenvinoDetector constructed!" << std::endl;
}

void OpenvinoDetector::set_onnx_model(const std::string &model_path, const std::string &device)
{
     // -------- Step 1. Initialize OpenVINO Runtime Core -------
    core_ = ov::Core();
    // -------- Step 2. Read a model --------
    std::shared_ptr<ov::Model> model;
    model = core_.read_model(model_path);
    if(model == nullptr)
    {
        std::cerr << "Model not loaded, check the model_path!" << std::endl;
        return;
    }
    // -------- Step 3. Compile a model --------
    compiled_model_ = core_.compile_model(model, device);
    // -------- Step 4. Create an Infer Request --------
    infer_request_ = compiled_model_.create_infer_request();
    std::cout << "Openvino onnx model loaded successfully!" << std::endl;
}

void OpenvinoDetector::infer(const cv::Mat &input, int detect_color)
{
    cv::Rect roiRect(input.cols / 2 - 300, input.rows / 2 - 300, 600, 600);
    cv::Mat resized_img = input(roiRect);
    armors_.clear();
    // -------- Step 5. Prepare input --------
    cv::Mat letterbox_img = letterbox(resized_img);
    cv::Mat blob;
    cv::dnn::blobFromImage(letterbox_img, blob, 1.0 / 255.0, cv::Size(IMAGE_WIDTH_, IMAGE_HEIGHT_), cv::Scalar(), true, false);

    // 图片缩放比例，因为输出的框是由缩放后的图片得到的，所以要根据缩放比例还原回输入的图片的大小
    float x_scale = letterbox_img.cols / IMAGE_WIDTH_;
    float y_scale = letterbox_img.rows / IMAGE_HEIGHT_;

    auto input_port = compiled_model_.input();
    ov::Tensor input_tensor(input_port.get_element_type(), input_port.get_shape(), blob.ptr(0));

    // -------- Step 6. Do inference --------
    infer_request_.set_input_tensor(input_tensor);
    infer_request_.infer();

    // -------- Step 7. Process output --------
    // 获取输出的张量，得根据实际情况修改，主要根据训练时的网络结构来确定
    // yolov5 has an output of shape (batchSize, 25200, 85) (box[x,y,w,h] + confidence[c] + Num classes)
    // yolov8 has an output of shape (batchSize, 84,  8400) (box[x,y,w,h] + Num classes)
    auto output0 = infer_request_.get_output_tensor(0);
    cv::Mat output_buffer(output0.get_shape()[1], output0.get_shape()[2], CV_32F, output0.data());

    std::vector<int> class_number_ids;
    std::vector<float> class_number_scores;
    std::vector<int> class_color_ids;
    std::vector<float> class_color_scores;
    std::vector<cv::Rect> boxes;
    std::vector<std::vector<cv::Point2f>> four_points_vec;

    if(output_buffer.rows < output_buffer.cols) // yolov8 == output0.get_shape()[1] < output0.get_shape()[2]
    {
        for(int i = 0; i<output_buffer.cols; i++)
        {
            cv::Mat class_scores_mat = output_buffer.col(i).rowRange(4,16);
            cv::transpose(class_scores_mat, class_scores_mat);
            cv::Mat boxes_mat = output_buffer.col(i).rowRange(0,4);
            cv::transpose(boxes_mat, boxes_mat);
            cv::Point class_id_point;
            double maxClassScore;
            cv::minMaxLoc(class_scores_mat, 0, &maxClassScore, 0, &class_id_point);
            if(maxClassScore > SCORE_THRESHOLD_)
            {
                int class_number_id;
                int class_color_id;
                std::cout << "class_id_point.x: " << class_id_point.x << std::endl;
                if(class_id_point.x < 6 && detect_color == 0)
                {
                    class_number_id = class_id_point.x + 1;
                    class_color_id = 0;
                }
                else if((class_id_point.x < 12 || class_id_point.x >= 7) && detect_color == 1)
                {
                    class_number_id = class_id_point.x - 5;
                    class_color_id = 1;
                }
                else if(class_id_point.x == 6 && detect_color == 0)
                {
                    class_number_id = 0;
                    class_color_id = 0;
                }
                else if(class_id_point.x == 12 && detect_color == 1)
                {
                    class_number_id = 0;
                    class_color_id = 1;
                }
                else
                {
                    continue;
                }
                float class_number_score = maxClassScore;
                float class_color_score = maxClassScore;
                float cx = boxes_mat.at<float>(0,0);
                float cy = boxes_mat.at<float>(0,1);
                float w = boxes_mat.at<float>(0,2);
                float h = boxes_mat.at<float>(0,3);
                int left = int((cx - w / 2) * x_scale);
                int top = int((cy - h / 2) * y_scale);
                int width = int(w * x_scale);
                int height = int(h * y_scale);
                cv::Rect rect(left, top, width, height);
                class_number_ids.push_back(class_number_id);
                class_number_scores.push_back(class_number_score);
                class_color_ids.push_back(class_color_id);
                class_color_scores.push_back(class_color_score);
                boxes.push_back(rect);
                std::vector<cv::Point2f> four_point;
                four_point.push_back(cv::Point(left, top));
                four_point.push_back(cv::Point(left + width, top));
                four_point.push_back(cv::Point(left + width, top + height));
                four_point.push_back(cv::Point(left, top + height));
                four_points_vec.push_back(four_point);
            }
        }
    }
    else if(output_buffer.rows > output_buffer.cols) // yolov5 == output0.get_shape()[1] > output0.get_shape()[2]
    {
        for(int i = 0; i<output_buffer.rows; i++)
        {
            if(output_buffer.at<float>(i, 8) < CONFIDENCE_THRESHOLD_)
            {
                continue;
            }
            cv::Mat xyxyxyxy_boxes_mat = output_buffer.row(i).colRange(0,8);
            cv::Mat color_scores_mat = output_buffer.row(i).colRange(9,13);
            cv::Mat number_scores_mat = output_buffer.row(i).colRange(13,22);
            cv::Point number_id_point;
            cv::Point color_id_point;
            double maxNumberScore;
            double maxColorScore;
            cv::minMaxLoc(number_scores_mat, 0, &maxNumberScore, 0, &number_id_point);
            cv::minMaxLoc(color_scores_mat, 0, &maxColorScore, 0, &color_id_point);
            maxNumberScore = sigmoid(maxNumberScore);
            maxColorScore = sigmoid(maxColorScore);
            if(maxNumberScore > SCORE_THRESHOLD_ && maxColorScore > COLOR_THRESHOLD_)
            {
                int class_number = number_id_point.x;
                float class_number_score = maxNumberScore;
                int class_color = color_id_point.x;
                if(class_color == 0 && detect_color == 0) // blue
                {
                }
                else if(class_color == 1 && detect_color == 1) // red
                {
                }
                else
                {
                    continue;
                }
                float class_color_score = maxColorScore;
                float x1 = xyxyxyxy_boxes_mat.at<float>(0,0) * x_scale;
                float y1 = xyxyxyxy_boxes_mat.at<float>(0,1) * y_scale;
                float x2 = xyxyxyxy_boxes_mat.at<float>(0,2) * x_scale;
                float y2 = xyxyxyxy_boxes_mat.at<float>(0,3) * y_scale;
                float x3 = xyxyxyxy_boxes_mat.at<float>(0,4) * x_scale;
                float y3 = xyxyxyxy_boxes_mat.at<float>(0,5) * y_scale;
                float x4 = xyxyxyxy_boxes_mat.at<float>(0,6) * x_scale; 
                float y4 = xyxyxyxy_boxes_mat.at<float>(0,7) * y_scale;
                class_number_ids.push_back(class_number);
                class_number_scores.push_back(class_number_score);
                class_color_ids.push_back(class_color);
                class_color_scores.push_back(class_color_score);
                cv::Rect rect(x1, y1, x3 - x1, y3 - y1);
                boxes.push_back(rect);
                std::vector<cv::Point2f> four_point;
                four_point.push_back(cv::Point(x1, y1));
                four_point.push_back(cv::Point(x4, y4));
                four_point.push_back(cv::Point(x3, y3));
                four_point.push_back(cv::Point(x2, y2));
                four_points_vec.push_back(four_point);
            }
        }
    }

    // -------- Step 8. NMSBoxes process --------
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, class_number_scores, SCORE_THRESHOLD_, NMS_THRESHOLD_, indices);
    for(size_t i = 0; i < indices.size(); i++)
    {
        int idx = indices[i];
        cv::Rect box = boxes[idx];
        int class_number_id = class_number_ids[idx];
        int class_color_id = class_color_ids[idx];
        float number_score = class_number_scores[idx];
        float color_score = class_color_scores[idx];
        std::string class_name;
        if(class_color_id == 0)
        {
            class_name = class_names_[class_number_id] + " blue--" + std::to_string(number_score) + "--" + std::to_string(color_score);
        }
        else if(class_color_id == 1)
        {
            class_name = class_names_[class_number_id] + " red--" + std::to_string(number_score) + "--" + std::to_string(color_score);
        }
        std::vector<cv::Point2f> four_points = four_points_vec[idx];
        Armor armor;
        armor.id = class_number_id;
        armor.color = class_color_id;
        armor.number_score = number_score;
        armor.color_score = color_score;
        armor.name = class_name;
        box.x = input.cols / 2 - 300 + box.x;
        box.y = input.rows / 2 - 300 + box.y;
        armor.rect = box;
        four_points[0].x = input.cols / 2 - 300 + four_points[0].x;
        four_points[0].y = input.rows / 2 - 300 + four_points[0].y;
        four_points[1].x = input.cols / 2 - 300 + four_points[1].x;
        four_points[1].y = input.rows / 2 - 300 + four_points[1].y;
        four_points[2].x = input.cols / 2 - 300 + four_points[2].x;
        four_points[2].y = input.rows / 2 - 300 + four_points[2].y;
        four_points[3].x = input.cols / 2 - 300 + four_points[3].x;
        four_points[3].y = input.rows / 2 - 300 + four_points[3].y;
        armor.four_points = four_points;
        armors_.push_back(armor);
    }
}

cv::Mat OpenvinoDetector::letterbox(const cv::Mat &input)
{
    int col = input.cols;
    int row = input.rows;
    int _max = std::max(col, row);
    cv::Mat result = cv::Mat::zeros(_max, _max, CV_8UC3);
    input.copyTo(result(cv::Rect(0, 0, col, row)));
    return result;
}

double OpenvinoDetector::sigmoid(double x) 
{
    if(x>0)
        return 1.0 / (1.0 + exp(-x));
    else
        return exp(x) / (1.0 + exp(x));
}