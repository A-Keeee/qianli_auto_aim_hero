#include <iostream>

#include <opencv2/opencv.hpp>

class ArmorEKF
{
public:
    ArmorEKF();
    void Q_update();
    void R_update(cv::Mat &z);
    void set_state(cv::Mat &state);
    void motion_model(cv::Mat &state);
    cv::Mat observation_model(cv::Mat &state);
    cv::Mat jacob_f();
    cv::Mat jacob_h();
    void EKF_predict();
    void EKF_update(cv::Mat &Z);

    double dt_;

    double q_xyz_;
    double r_xyz_;

    cv::Mat state_;
    cv::Mat P_;
    cv::Mat Q_;
    cv::Mat R_;
};