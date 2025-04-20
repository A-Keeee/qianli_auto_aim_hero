#pragma once

#define    USE_CUDA
#define    RET_OK nullptr

#ifdef _WIN32
#include <Windows.h>
#include <direct.h>
#include <io.h>
#endif

#include <string>
#include <vector>
#include <cstdio>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include "opencv2/imgproc/imgproc_c.h"
#include <opencv2/imgproc/types_c.h>
#include "onnxruntime_cxx_api.h"

#include "armor.hpp"

#ifdef USE_CUDA
#include <cuda_fp16.h>
#endif

typedef struct _DL_INIT_PARAM
{
    std::string modelPath;
    std::vector<int> imgSize = { 640, 640 };
    float rectConfidenceThreshold = 0.6;
    float iouThreshold = 0.5;
    bool cudaEnable = true;
    int logSeverityLevel = 3;
    int intraOpNumThreads = 1;
} DL_INIT_PARAM;


class CudaDetector
{
public:
    CudaDetector();

    char* CreateSession(DL_INIT_PARAM& iParams);

    void infer(const cv::Mat &input, int detect_color);

    Ort::Env env;
    Ort::Session* session;
    bool cudaEnable;
    Ort::RunOptions options;
    std::vector<const char*> inputNodeNames;
    std::vector<const char*> outputNodeNames;

    std::vector<int> imgSize;
    float rectConfidenceThreshold;
    float iouThreshold;
    float resizeScales;//letterbox scale

    const float IMAGE_WIDTH_ = 640;
    const float IMAGE_HEIGHT_ = 640;
    const float CONFIDENCE_THRESHOLD_ = 0.8;
    const float SCORE_THRESHOLD_ = 0.9;
    const float COLOR_THRESHOLD_ = 0.9;
    const float NMS_THRESHOLD_ = 0.5;
    const std::vector<std::string> class_names_ = {"sentry", "1", "2", "3", "4", "5", "outpost", "base", "base_big"};
    std::vector<Armor> armors_;
};