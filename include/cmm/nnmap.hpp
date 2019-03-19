#ifndef NNMAP_HPP
#define NNMAP_HPP

#include "functional"
#include "data.hpp"
#include "classifier.hpp"

namespace cmm {

/**
 * @brief The NNMap class
 * Simple classifier based on Nearest Neighbor Map
 */
class NNMap : public Classifier{

public:
    NNMap(){}
    NNMap(int dimension, double dist_thre, double incr) :
        Classifier(dimension, 2),
        distance_threshold(dist_thre),
        increment_factor(incr){
        _distance = [](const Eigen::VectorXd& s1,const Eigen::VectorXd& s2) -> double {
            return (s1 - s2).squaredNorm();
        };
    }
    NNMap(const NNMap& nnmap) :
        Classifier(nnmap),
        distance_threshold(nnmap.distance_threshold),
        increment_factor(nnmap.increment_factor){}


    std::vector<double> compute_estimation(const Eigen::VectorXd& sample) const;
    void update(){}
    double confidence(const Eigen::VectorXd& sample) const {return 1.;}
    int next_sample(const std::vector<std::pair<Eigen::VectorXd, std::vector<double>> >& samples, Eigen::VectorXd &choice_dist_map) {}

    //parameters
    double distance_threshold;
    double increment_factor;
    double default_estimation = .5;

};

}

#endif //NNMAP_HPP
