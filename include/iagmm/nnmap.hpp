#ifndef NNMAP_HPP
#define NNMAP_HPP

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
        increment_factor(incr){}
    NNMap(const NNMap& nnmap) :
        Classifier(nnmap),
        distance_threshold(nnmap.distance_threshold),
        increment_factor(nnmap.increment_factor){}


    double compute_estimation(const Eigen::VectorXd& sample, int label = 1);

    //parameters
    double distance_threshold;
    double increment_factor;
    double default_estimation = 0.5;

};

}

#endif //NNMAP_HPP
