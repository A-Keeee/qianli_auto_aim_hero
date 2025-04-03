#include <iostream>

#include "opencv2/opencv.hpp"

class EKF
{
public:
    EKF();
    void Q_update();
    void R_update(const cv::Mat &z);
    void set_state(cv::Mat &state);
    void motion_model(cv::Mat &state);
    cv::Mat observation_model(cv::Mat &state);
    cv::Mat jacob_f();
    cv::Mat jacob_h(const cv::Mat &state);
    void EKF_predict();
    void EKF_update(const cv::Mat &z);

    double dt_;

    double q_xyz_;
    double q_yaw_;
    double q_r_;

    double r_xyz_;
    double r_yaw_;

    cv::Mat state_;
    cv::Mat P_;
    cv::Mat Q_;
    cv::Mat R_;
};