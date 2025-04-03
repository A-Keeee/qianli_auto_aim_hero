#pragma once
#include <iostream>

#include "opencv2/opencv.hpp"

#include "rm_armor_detector/number_classifier.hpp"
#include "rm_armor_detector/armor.hpp"

class LightsDetector
{
public:
    struct LightParams
    {
        // width / height
        double min_ratio;
        double max_ratio;
        // vertical angle
        double max_angle;
    };

    struct ArmorParams
    {
        double min_light_ratio;
        // light pairs distance
        double min_small_center_distance;
        double max_small_center_distance;
        double min_large_center_distance;
        double max_large_center_distance;
        // horizontal angle
        double max_angle;
    };

    LightsDetector(const int & bin_thres, const int & color, const LightParams & l, const ArmorParams & a);

    std::vector<LightArmor> detect(const cv::Mat & input);

    cv::Mat preprocessImage(const cv::Mat & input);
    std::vector<Light> findLights(const cv::Mat & rbg_img, const cv::Mat & binary_img);
    std::vector<LightArmor> matchLights(const std::vector<Light> & lights);

    // For debug usage
    cv::Mat getAllNumbersImage();
    void drawResults(cv::Mat & img);

    void translateArmorToLightArmor(std::vector<LightArmor> & light_armors);

    int binary_thres;
    int detect_color;
    LightParams l;
    ArmorParams a;

    std::unique_ptr<NumberClassifier> classifier;

    // Debug msgs
    cv::Mat binary_img;

    bool isLight(const Light & possible_light);
    bool containLight(
        const Light & light_1, const Light & light_2, const std::vector<Light> & lights);
    ArmorType isArmor(const Light & light_1, const Light & light_2);

    std::vector<Light> lights_;
    std::vector<LightArmor> armors_;
};