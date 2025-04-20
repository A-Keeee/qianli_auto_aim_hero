#include "pnp_solver.hpp"

PnpSolver::PnpSolver()
{
    init();
    std::cout << "PnpSolver constructed!" << std::endl;
}

void PnpSolver::init()
{
    // 单位为m
    double small_armor_half_width = SMALL_ARMOR_WIDTH_ / 2;
    double small_armor_half_height = SMALL_ARMOR_LIGHT_HEIGHT_ / 2;
    double big_armor_half_width = BIG_ARMOR_WIDTH_ / 2;
    double big_armor_half_height = BIG_ARMOR_LIGHT_HEIGHT_ / 2;

    // 装甲板实际四个点的坐标，以中心为原点，朝装甲板向外为x轴正方向，单位为m， 左灯条上顶点为起始点，顺时针
    small_armor_points_3d_.push_back(cv::Point3f(0, -small_armor_half_width, small_armor_half_height));
    small_armor_points_3d_.push_back(cv::Point3f(0, small_armor_half_width, small_armor_half_height));
    small_armor_points_3d_.push_back(cv::Point3f(0, small_armor_half_width, -small_armor_half_height));
    small_armor_points_3d_.push_back(cv::Point3f(0, -small_armor_half_width, -small_armor_half_height));

    big_armor_points_3d_.push_back(cv::Point3f(0, -big_armor_half_width, big_armor_half_height));
    big_armor_points_3d_.push_back(cv::Point3f(0, big_armor_half_width, big_armor_half_height));
    big_armor_points_3d_.push_back(cv::Point3f(0, big_armor_half_width, -big_armor_half_height));
    big_armor_points_3d_.push_back(cv::Point3f(0, -big_armor_half_width, -big_armor_half_height));

    // 中点法
    /*small_armor_points_3d_.push_back(cv::Point3f(0, 0, small_armor_half_height));
    small_armor_points_3d_.push_back(cv::Point3f(0, small_armor_half_width, 0));
    small_armor_points_3d_.push_back(cv::Point3f(0, 0, - small_armor_half_height));
    small_armor_points_3d_.push_back(cv::Point3f(0, -small_armor_half_width, 0));

    big_armor_points_3d_.push_back(cv::Point3f(0, 0, big_armor_half_height));
    big_armor_points_3d_.push_back(cv::Point3f(0, big_armor_half_width, 0));
    big_armor_points_3d_.push_back(cv::Point3f(0, 0, -big_armor_half_height));
    big_armor_points_3d_.push_back(cv::Point3f(0, -big_armor_half_width, 0));*/

    std::cout << "PnpSolver init successfully!" << std::endl;
}

void PnpSolver::set_matrix(const int &image_width, const int &image_height, cv::Mat &camera_matrix, const cv::Mat &dist_coeffs)
{
    this->image_width_ = image_width;
    this->image_height_ = image_height;
    this->image_center_ = cv::Point2f(image_width / 2, image_height / 2);
    this->camera_matrix_ = camera_matrix;
    this->dist_coeffs_ = dist_coeffs;
    std::cout << "PnpSolver set matrix successfully!" << std::endl;
}

bool PnpSolver::solve_pnp(const std::vector<cv::Point2f> &points, cv::Mat &rvec, cv::Mat &tvec, bool is_big_armor)
{
    // 模型准确度有待测试
    //cv::Mat inliers;
    //cv::solvePnPRansac(points, armor_points_3d_, camera_matrix_, dist_coeffs_, rvec, tvec, true, 100, 8.0, 0.99, inliers);
    /*std::vector<cv::Point2f> half_points;
    half_points.push_back(cv::Point2f((points[0].x + points[1].x) / 2,(points[0].y + points[1].y) / 2));
    half_points.push_back(cv::Point2f((points[1].x + points[2].x) / 2,(points[1].y + points[2].y) / 2));
    half_points.push_back(cv::Point2f((points[2].x + points[3].x) / 2,(points[2].y + points[3].y) / 2));
    half_points.push_back(cv::Point2f((points[3].x + points[0].x) / 2,(points[3].y + points[0].y) / 2));*/

    std::vector<cv::Point3f> objectPoints = is_big_armor ? big_armor_points_3d_ : small_armor_points_3d_;
    cv::solvePnP(objectPoints, points, camera_matrix_, dist_coeffs_, rvec, tvec, false, cv::SOLVEPNP_IPPE);
    return true;
}

double PnpSolver::get_distance_armor_center_to_image_center(const cv::Point2f &armor_center)
{
    // 单位：像素
    double distance_to_image_center = cv::norm(armor_center - image_center_);
    return distance_to_image_center;
}
