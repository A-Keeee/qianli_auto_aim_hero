#include "cuda_detector.hpp"
#include <regex>

#define benchmark
#define min(a,b)            (((a) < (b)) ? (a) : (b))

cv::Mat letterbox(const cv::Mat &input)
{
    int col = input.cols;
    int row = input.rows;
    int _max = std::max(col, row);
    cv::Mat result = cv::Mat::zeros(_max, _max, CV_8UC3);
    input.copyTo(result(cv::Rect(0, 0, col, row)));
    return result;
}

double sigmoid(double x) 
{
    if(x>0)
        return 1.0 / (1.0 + exp(-x));
    else
        return exp(x) / (1.0 + exp(x));
}

CudaDetector::CudaDetector()
{
    std::cout << "CudaDetector constructed." << std::endl;
}

char* CudaDetector::CreateSession(DL_INIT_PARAM& iParams)
{
    char* Ret = RET_OK;
    std::regex pattern("[\u4e00-\u9fa5]");
    bool result = std::regex_search(iParams.modelPath, pattern);
    if (result)
    {
        Ret = "[YOLO_V8]:Your model path is error.Change your model path without chinese characters.";
        std::cout << Ret << std::endl;
        return Ret;
    }
    try
    {
        rectConfidenceThreshold = iParams.rectConfidenceThreshold;
        iouThreshold = iParams.iouThreshold;
        imgSize = iParams.imgSize;
        cudaEnable = iParams.cudaEnable;
        env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "Yolo");
        Ort::SessionOptions sessionOption;
        if (iParams.cudaEnable)
        {
            OrtCUDAProviderOptions cudaOption;
            cudaOption.device_id = 0;
            sessionOption.AppendExecutionProvider_CUDA(cudaOption);
        }
        sessionOption.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        sessionOption.SetIntraOpNumThreads(iParams.intraOpNumThreads);
        sessionOption.SetLogSeverityLevel(iParams.logSeverityLevel);

#ifdef _WIN32
        int ModelPathSize = MultiByteToWideChar(CP_UTF8, 0, iParams.modelPath.c_str(), static_cast<int>(iParams.modelPath.length()), nullptr, 0);
        wchar_t* wide_cstr = new wchar_t[ModelPathSize + 1];
        MultiByteToWideChar(CP_UTF8, 0, iParams.modelPath.c_str(), static_cast<int>(iParams.modelPath.length()), wide_cstr, ModelPathSize);
        wide_cstr[ModelPathSize] = L'\0';
        const wchar_t* modelPath = wide_cstr;
#else
        const char* modelPath = iParams.modelPath.c_str();
#endif // _WIN32

        session = new Ort::Session(env, modelPath, sessionOption);
        Ort::AllocatorWithDefaultOptions allocator;
        size_t inputNodesNum = session->GetInputCount();
        for (size_t i = 0; i < inputNodesNum; i++)
        {
            Ort::AllocatedStringPtr input_node_name = session->GetInputNameAllocated(i, allocator);
            char* temp_buf = new char[50];
            strcpy(temp_buf, input_node_name.get());
            inputNodeNames.push_back(temp_buf);
        }
        size_t OutputNodesNum = session->GetOutputCount();
        for (size_t i = 0; i < OutputNodesNum; i++)
        {
            Ort::AllocatedStringPtr output_node_name = session->GetOutputNameAllocated(i, allocator);
            char* temp_buf = new char[10];
            strcpy(temp_buf, output_node_name.get());
            outputNodeNames.push_back(temp_buf);
        }
        options = Ort::RunOptions{ nullptr };
        //WarmUpSession();
        return RET_OK;
    }
    catch (const std::exception& e)
    {
        const char* str1 = "[YOLO_V8]:";
        const char* str2 = e.what();
        std::string result = std::string(str1) + std::string(str2);
        char* merged = new char[result.length() + 1];
        std::strcpy(merged, result.c_str());
        std::cout << merged << std::endl;
        delete[] merged;
        return "[YOLO_V8]:Create session failed.";
    }

}

void CudaDetector::infer(const cv::Mat &input, int detect_color)
{
    cv::Mat blob_img;
    cv::Mat letterbox_img = letterbox(input);
    cv::dnn::blobFromImage(letterbox_img, blob_img, 1.0 / 255.0, cv::Size(IMAGE_WIDTH_, IMAGE_HEIGHT_), cv::Scalar(), true, false);
    // 获取 blob 的数据指针
	float* blob_data = reinterpret_cast<float*>(blob_img.data);
    // 计算 blob 的大小
	size_t blob_size = blob_img.total() * blob_img.channels();
	float x_scale = (float)letterbox_img.cols / IMAGE_WIDTH_;
	float y_scale = (float)letterbox_img.rows / IMAGE_HEIGHT_;
    std::vector<int64_t> inputNodeDims = { 1, 3, imgSize.at(0), imgSize.at(1) };
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU), blob_data, blob_size,
        inputNodeDims.data(), inputNodeDims.size());
    
    //处理推理数据begin
    auto outputTensor = session->Run(options, inputNodeNames.data(), &inputTensor, 1, outputNodeNames.data(),outputNodeNames.size());
    float* pdata = outputTensor.front().GetTensorMutableData<float>();
	cv::Mat output_buffer(outputTensor.front().GetTypeInfo().GetTensorTypeAndShapeInfo().GetShape().at(1)
                                        , outputTensor.front().GetTypeInfo().GetTensorTypeAndShapeInfo().GetShape().at(2), CV_32FC1, pdata);

    std::vector<int> class_number_ids;
    std::vector<float> class_number_scores;
    std::vector<int> class_color_ids;
    std::vector<float> class_color_scores;
    std::vector<cv::Rect> boxes;
    std::vector<std::vector<cv::Point2f>> four_points_vec;

    for(int i = 1;i<output_buffer.rows;i++)
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
        armor.rect = box;
        armor.four_points = four_points;
        armors_.push_back(armor);
    }
}
