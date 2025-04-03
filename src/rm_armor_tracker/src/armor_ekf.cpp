#include "armor_ekf.hpp"

ArmorEKF::ArmorEKF()
{
    std::cout << "ArmorEKF constructed!" << std::endl;
    state_ = cv::Mat::zeros(4, 1, CV_64F);
    P_ = cv::Mat::eye(4, 4, CV_64F);
    Q_ = cv::Mat::zeros(4, 4, CV_64F);
    R_ = cv::Mat::zeros(2, 2, CV_64F);
    dt_ = 0.0;
    q_xyz_ = 0.0;
    r_xyz_ = 0.0;
}

void ArmorEKF::Q_update()
{
    double t = dt_, x = q_xyz_;
    double q_x_x = pow(t, 4) / 4 * x, q_x_vx = pow(t, 3) / 2 * x, q_vx_vx = pow(t, 2) * x;

    Q_ = (cv::Mat_<double>(4, 4) << q_x_x,  q_x_vx, 0,      0,
                                    q_x_vx, q_vx_vx,0,      0,
                                    0,      0,      q_x_x,  q_x_vx,
                                    0,      0,      q_x_vx, q_vx_vx);
}

void ArmorEKF::R_update(cv::Mat &z)
{
    R_ = (cv::Mat_<double>(2, 2) << abs(r_xyz_ * z.at<double>(0,0)), 0,
                                    0, abs(r_xyz_ * z.at<double>(1,0)));
}

void ArmorEKF::set_state(cv::Mat &state)
{
    state_ = state;
}

void ArmorEKF::motion_model(cv::Mat &state)
{
    //                                    xa xa_v ya ya_v
    cv::Mat F = (cv::Mat_<double>(4,4) << 1, dt_, 0, 0,
                                           0, 1, 0, 0,
                                           0, 0, 1, dt_,
                                           0, 0, 0, 1);
    state = F * state;                                        
}

cv::Mat ArmorEKF::observation_model(cv::Mat &state)
{
    cv::Mat Z = cv::Mat::zeros(2, 1, CV_64F);
    Z.at<double>(0, 0) = state.at<double>(0, 0);
    Z.at<double>(1, 0) = state.at<double>(2, 0);
    return Z;
}

cv::Mat ArmorEKF::jacob_f()
{
    cv::Mat jF = (cv::Mat_<double>(4, 4) << 1, dt_, 0, 0,
                                            0, 1, 0, 0,
                                            0, 0, 1, dt_,
                                            0, 0, 0, 1);
    return jF;
}

cv::Mat ArmorEKF::jacob_h()
{
    cv::Mat jH = (cv::Mat_<double>(2, 4) << 1, 0, 0, 0,
                                            0, 0, 1, 0);
    return jH;
}

void ArmorEKF::EKF_predict()
{
    // 1. Predict
    Q_update();
    motion_model(state_);
    cv::Mat jF = jacob_f();
    P_ = jF * P_ * jF.t() + Q_;
}

void ArmorEKF::EKF_update(cv::Mat &z)
{
    // 2. Update
    R_update(z);
    cv::Mat jH = jacob_h();
    cv::Mat z_pred = observation_model(state_);
    cv::Mat y = z - z_pred;
    cv::Mat S = jH * P_ * jH.t() + R_;
    cv::Mat K = P_ * jH.t() * S.inv(cv::DECOMP_SVD);
    state_ = state_ + K * y;
    P_ = (cv::Mat::eye(state_.rows, state_.rows, CV_64F) - K * jH) * P_;
}

    