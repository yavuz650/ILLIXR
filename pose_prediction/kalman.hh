#ifndef KALMAN_HH__
#define KALMAN_HH__

#include <chrono>
#include "common/data_format.hh"
#include <Eigen/Dense>

using namespace ILLIXR;

class kalman_filter {
    public:
        kalman_filter();
        Eigen::Vector3f predict_values(imu_type, float);
        void add_estimate(imu_type);
        void update_estimates(Eigen::Quaternionf);

    private:
        // We dont do anything with these here but they should be calculated per device which is
        // done by taking the average value of N measurements of the device at rest
        float _phi_offset = 0;
        float _theta_offset = 0;

        float _phi_estimate = 0;
        float _theta_estimate = 0;
        float _rho_est = 0;

        Eigen::MatrixXf C{2, 4};
        Eigen::MatrixXf P = Eigen::MatrixXf::Identity(4, 4);
        Eigen::MatrixXf Q = Eigen::MatrixXf::Identity(4, 4);
        Eigen::MatrixXf R = Eigen::MatrixXf::Identity(2, 2);
        Eigen::MatrixXf _state_estimate{4, 1};
};

#endif
