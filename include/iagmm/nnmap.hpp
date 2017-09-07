#ifndef NNMAP_HPP
#define NNMAP_HPP

#include "functional"
#include "data.hpp"
#include "classifier.hpp"

namespace iagmm {

/**
 * @brief The NNMap class
 * Simple classifier based on Nearest Neighbor Map
 */
class NNMap : public Classifier{

public:
    NNMap(){}
    NNMap(int dimension, int nbr_class, double dist_thre, double incr) :
        Classifier(dimension, nbr_class),
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


    double compute_estimation(const Eigen::VectorXd& sample, int label = 1);
    void update(){}
    double confidence(const Eigen::VectorXd& sample) const {return 1.;}
    int next_sample(const std::vector<std::pair<Eigen::VectorXd, double> >& samples, Eigen::VectorXd &choice_dist_map) {}

    //parameters
    double distance_threshold;
    double increment_factor;
    double default_estimation = .5;

};

}

#endif //NNMAP_HPP
