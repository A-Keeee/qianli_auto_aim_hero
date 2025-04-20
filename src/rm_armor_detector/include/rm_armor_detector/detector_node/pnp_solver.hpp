#include <iostream>
#include <vector>

#include <opencv2/opencv.hpp>

class PnpSolver
{
public:
    PnpSolver();
    void init();
    void set_matrix(const int &image_width, const int &image_height, cv::Mat &camera_matrix, const cv::Mat &dist_coeffs);
    bool solve_pnp(const std::vector<cv::Point2f> &points, cv::Mat &rvec, cv::Mat &tvec, bool is_big_armor);
    double get_distance_armor_center_to_image_center(const cv::Point2f &armor_center);

    // 单位：m
    const double SMALL_ARMOR_WIDTH_ = 0.135;
    const double SMALL_ARMOR_LIGHT_HEIGHT_ = 0.055;
    const double BIG_ARMOR_WIDTH_ = 0.225;
    const double BIG_ARMOR_LIGHT_HEIGHT_ = 0.055;

    int image_width_;
    int image_height_;
    cv::Mat camera_matrix_;
    cv::Mat dist_coeffs_;
    cv::Point2f image_center_;

    std::vector<cv::Point3f> small_armor_points_3d_;
    std::vector<cv::Point3f> big_armor_points_3d_;
};