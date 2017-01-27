#ifndef CLASSIFIER_HPP
#define CLASSIFIER_HPP

#include <eigen3/Eigen/Core>
#include "data.hpp"

namespace iagmm {

class Classifier{
public:
    Classifier(){}
    Classifier(int dimension, int nbr_class = 2) :
        _dimension(dimension), _nbr_class(nbr_class){}
    Classifier(const Classifier &c) :
        _nbr_class(c._nbr_class),
        _dimension(c._dimension),
        _samples(c._samples){}

    ~Classifier(){}

    virtual double compute_estimation (const Eigen::VectorXd& sample, int lbl) = 0;
    void add(const Eigen::VectorXd& sample, int lbl){
        _samples.add(lbl,sample);
    }

    size_t dataset_size() const {return _samples.size();}
    const TrainingData& get_samples() const {return _samples;}

protected:
    int _nbr_class;
    int _dimension;
    TrainingData _samples;

};

}
#endif //CLASSIFIER_HPP
