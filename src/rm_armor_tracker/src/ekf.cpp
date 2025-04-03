#include "ekf.hpp"

EKF::EKF()
{
    std::cout << "EKF constructed!" << std::endl;
    state_ = cv::Mat::zeros(9, 1, CV_64F);
    P_ = cv::Mat::eye(9, 9, CV_64F);
    Q_ = cv::Mat::zeros(9, 9, CV_64F);
    R_ = cv::Mat::zeros(4, 4, CV_64F);
    dt_ = 0.0;
    q_xyz_ = 0.0;
    q_yaw_ = 0.0;
    q_r_ = 0.0;
    r_xyz_ = 0.0;
    r_yaw_ = 0.0;
}

void EKF::Q_update()
{
    double t = dt_, x = q_xyz_, y = q_yaw_, r = q_r_;
    double q_x_x = pow(t, 4) / 4 * x, q_x_vx = pow(t, 3) / 2 * x, q_vx_vx = pow(t, 2) * x;
    double q_y_y = pow(t, 4) / 4 * y, q_y_vy = pow(t, 3) / 2 * x, q_vy_vy = pow(t, 2) * y;
    double q_r = pow(t, 4) / 4 * r;

    Q_ = (cv::Mat_<double>(9, 9) << q_x_x,  q_x_vx, 0,      0,      0,      0,      0,      0,      0,
                                    q_x_vx, q_vx_vx,0,      0,      0,      0,      0,      0,      0,
                                    0,      0,      q_x_x,  q_x_vx, 0,      0,      0,      0,      0,
                                    0,      0,      q_x_vx, q_vx_vx,0,      0,      0,      0,      0,
                                    0,      0,      0,      0,      q_x_x,  q_x_vx, 0,      0,      0,
                                    0,      0,      0,      0,      q_x_vx, q_vx_vx,0,      0,      0,
                                    0,      0,      0,      0,      0,      0,      q_y_y,  q_y_vy, 0,
                                    0,      0,      0,      0,      0,      0,      q_y_vy, q_vy_vy,0,
                                    0,      0,      0,      0,      0,      0,      0,      0,      q_r);
}

void EKF::R_update(const cv::Mat &z)
{
    R_ = (cv::Mat_<double>(4, 4) << abs(r_xyz_ * z.at<double>(0,0)), 0,      0,      0,
                                    0,      abs(r_xyz_ * z.at<double>(1,0)), 0,      0,
                                    0,      0,      abs(r_xyz_ * z.at<double>(2,0)), 0,
                                    0,      0,      0,      r_yaw_);
}

void EKF::set_state(cv::Mat &state)
{
    state_ = state;
}

void EKF::motion_model(cv::Mat &state)
{
    //                                    xc xc_v yc yc_v za za_v yaw yaw_v r
    cv::Mat F = (cv::Mat_<double>(9,9) << 1, dt_, 0, 0, 0, 0, 0, 0, 0,
                                           0, 1, 0, 0, 0, 0, 0, 0, 0,
                                           0, 0, 1, dt_, 0, 0, 0, 0, 0,
                                           0, 0, 0, 1, 0, 0, 0, 0, 0,
                                           0, 0, 0, 0, 1, dt_, 0, 0, 0,
                                           0, 0, 0, 0, 0, 1, 0, 0, 0,
                                           0, 0, 0, 0, 0, 0, 1, dt_, 0,
                                           0, 0, 0, 0, 0, 0, 0, 1, 0,
                                           0, 0, 0, 0, 0, 0, 0, 0, 1);

    state = F * state;                                        
}

cv::Mat EKF::observation_model(cv::Mat &state)
{
    cv::Mat Z = cv::Mat::zeros(4, 1, CV_64F);
    double yaw = state.at<double>(6, 0);
    double r = state.at<double>(8, 0);
    Z.at<double>(0, 0) = state.at<double>(0, 0) - r * sin(yaw);
    Z.at<double>(1, 0) = state.at<double>(2, 0) + r * cos(yaw);
    Z.at<double>(2, 0) = state.at<double>(4, 0);
    Z.at<double>(3, 0) = state.at<double>(6, 0);
    return Z;
}

cv::Mat EKF::jacob_f()
{
    cv::Mat jF = (cv::Mat_<double>(9, 9) << 1, dt_, 0, 0, 0, 0, 0, 0, 0,
                                            0, 1, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 1, dt_, 0, 0, 0, 0, 0,
                                            0, 0, 0, 1, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, 1, dt_, 0, 0, 0,
                                            0, 0, 0, 0, 0, 1, 0, 0, 0,
                                            0, 0, 0, 0, 0, 0, 1, dt_, 0,
                                            0, 0, 0, 0, 0, 0, 0, 1, 0,
                                            0, 0, 0, 0, 0, 0, 0, 0, 1);
    return jF;
}

cv::Mat EKF::jacob_h(const cv::Mat &state)
{
    double yaw = state.at<double>(6,0);
    double r = state.at<double>(8,0);
    cv::Mat jH;
    jH = (cv::Mat_<double>(4, 9) << 1,   0,   0,   0,   0,   0,   -r * cos(yaw),          0,   -sin(yaw),
                                    0,   0,   1,   0,   0,   0,   -r * sin(yaw),          0,   cos(yaw),
                                    0,   0,   0,   0,   1,   0,   0,          0,   0,
                                    0,   0,   0,   0,   0,   0,   1,          0,   0);
    return jH;  
}

void EKF::EKF_predict()
{
    // 1. Predict
    Q_update();
    motion_model(state_);
    cv::Mat jF = jacob_f();
    P_ = jF * P_ * jF.t() + Q_;
}

void EKF::EKF_update(const cv::Mat &z)
{
    // 2. Update
    R_update(z);
    cv::Mat jH = jacob_h(state_);
    cv::Mat z_pred = observation_model(state_);
    cv::Mat y = z - z_pred;
    cv::Mat S = jH * P_ * jH.t() + R_;
    cv::Mat K = P_ * jH.t() * S.inv(cv::DECOMP_SVD);
    state_ = state_ + K * y;
    P_ = (cv::Mat::eye(state_.rows, state_.rows, CV_64F) - K * jH) * P_;
}