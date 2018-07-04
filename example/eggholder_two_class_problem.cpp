#include <iostream>
#include <fstream>
#include <random>
#include <memory>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Eigenvalues>
#include <math.h>
#ifndef NO_PARALLEL
    #include <tbb/tbb.h>
#endif
#include <ctime>

#include <iagmm/gmm.hpp>

#define MAX_X 100
#define MAX_Y 100
#define PI 3.14159265359

using namespace iagmm;

double eggholder(double A,double x, double y){
    return -(y+A)*sin(sqrt(abs(x/2+(y+A))))-x*sin(sqrt(abs(x-(y+A))));
}

int main(int argc, char** argv){
    srand(std::time(NULL));

#ifndef NO_PARALLEL
    tbb::task_scheduler_init init;
#endif

    if(argc < 4){
        std::cout << "usage : alpha and outlier threshold" << std::endl;
        return 1;
    }

    double A;

    if(argc == 4)
        A = std::stod(argv[3]);
    else
        A = rand()%100;
    std::cout << "A = " << A << std::endl;
    int real_space[MAX_X][MAX_Y];
    double estimated_space[MAX_X][MAX_Y];
    std::vector<int> label;

    Component::_alpha = std::stod(argv[1]);
    Component::_outlier_thres = std::stod(argv[2]);

    GMM gmm(2,2);
    gmm.set_distance_function(
        [](const Eigen::VectorXd& s1,const Eigen::VectorXd& s2) -> double {
        return (s1 - s2).squaredNorm();
    });
    gmm.set_loglikelihood_driver(false);
    gmm.set_update_mode(iagmm::GMM::STOCHASTIC);
    Eigen::VectorXd choice_dist_map = Eigen::VectorXd::Zero(MAX_Y*MAX_X);

    double error;

    std::multimap<double,Eigen::Vector2i> choice_distribution;
    double cumul_est;


    Eigen::Vector2i coord(0,0);
    std::vector<std::pair<Eigen::VectorXd,std::vector<double>>> all_sample(MAX_Y*MAX_X);

    double val;
    for(int i = 0; i < MAX_X*MAX_Y; i++){
        coord[0] = i%MAX_X;
        if(coord[0] == MAX_X - 1)
            coord[1]++;
        if(coord[1] >= MAX_Y)
            coord[1] = 0;

        val = eggholder(A,coord[0] - MAX_X/2,coord[1] - MAX_Y/2);
        if(val > 0 )
            real_space[coord[0]][coord[1]] = 1;
        else real_space[coord[0]][coord[1]] = 0;

        estimated_space[coord[0]][coord[1]] = 0.;
    }

    int iteration = 0;
    int total_it = 1000;
    int bootstrap = 2;
    error = 1.;

    Eigen::VectorXd next_s;
    coord[0] = rand()%MAX_X;
    coord[1] = rand()%MAX_Y;
    while(iteration < total_it){
        std::vector<double> errors(2,0);
        std::vector<int> nbr_samples(2,0);
        label.push_back(real_space[coord[0]][coord[1]]);

        gmm.add(Eigen::Vector2d((double)coord[0]/(double)MAX_X,(double)coord[1]/(double)MAX_Y),
              real_space[coord[0]][coord[1]]);
        gmm.update();

        error = 0;


        if(gmm.get_samples().size() > bootstrap){
            cumul_est = 0;
            choice_distribution.clear();
#ifdef NO_PARALLEL
            for(int i = 0; i < MAX_X*MAX_Y; i++){
#else
            tbb::parallel_for(tbb::blocked_range<size_t>(0,MAX_X*MAX_Y),
                              [&](const tbb::blocked_range<size_t>& r){
                for(int i = r.begin(); i != r.end(); ++i){
#endif
                    std::vector<double> est_vect(2);

                    if(i%MAX_X == MAX_X - 1)
                        coord[1]++;
                    if(coord[1] >= MAX_Y)
                        coord[1] = 0;

                    est_vect = gmm.compute_estimation(
                                    Eigen::Vector2d((double)(i%MAX_X)/(double)MAX_X,(double)(i/MAX_X)/(double)MAX_Y));

                    all_sample[i] =
                                std::make_pair(
                                    Eigen::Vector2d((double)(i%MAX_X)/(double)MAX_X,(double)(i/MAX_X)/(double)MAX_Y),est_vect);
                }
#ifndef NO_PARALLEL
            });

#endif
            for(int i = 0; i < MAX_X*MAX_Y; i++){
                error += 1. - all_sample[i].second[real_space[i%MAX_X][i/MAX_X]];
                errors[real_space[i%MAX_X][i/MAX_X]] += 1. - all_sample[i].second[real_space[i%MAX_X][i/MAX_X]];
                nbr_samples[real_space[i%MAX_X][i/MAX_X]]++;
            }
            next_s = all_sample[gmm.next_sample(all_sample,choice_dist_map)].first;
            coord[0] = next_s(0)*MAX_X;
            coord[1] = next_s(1)*MAX_Y;
        }else{
            coord[0] = rand()%MAX_X;
            coord[1] = rand()%MAX_Y;
        }

        error = error/(double)(MAX_X*MAX_Y);
        for(int i : {0,1})
            errors[i] = errors[i]/(double)nbr_samples[i];

        std::cout << "ITERATION -- " << iteration << std::endl;
        for(int i = 0; i < 2; i++)
            std::cout << "class " << i << " : " << gmm.model()[i].size()
                      << " : " << gmm.get_samples().get_data(i).size()
                      << " : " << errors[i] << std::endl;
        std::cout << "ERROR = " << error << std::endl;
        iteration++;
    }

//    std::ofstream of("archive_gmm");
//    boost::archive::binary_oarchive oarch(of);
//    oarch << gmm;

    return 0;
}
