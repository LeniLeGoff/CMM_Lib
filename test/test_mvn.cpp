#include <iagmm/component.hpp>
#include <iostream>
#include <chrono>

int main(int argc, char** argv){
    if(argc != 3){
        std::cerr << "usage: dimension nbr_repl" << std::endl;
        return 1;
    }

    int dimension = atoi(argv[1]);
    iagmm::Component comp(dimension,2);

    comp.set_mu(Eigen::VectorXd::Constant(dimension,1));
    comp.set_covariance(Eigen::MatrixXd::Identity(dimension,dimension)*0.001);
    std::chrono::system_clock::time_point timer;
    timer  = std::chrono::system_clock::now();


    for(int i = 0; i < atoi(argv[2]); i++){
        comp.compute_multivariate_normal_dist(Eigen::VectorXd::Constant(dimension,0.1));
    }

    std::cout << "time spent : "
              << std::chrono::duration_cast<std::chrono::milliseconds>(
                     std::chrono::system_clock::now() - timer).count() << std::endl;
}
