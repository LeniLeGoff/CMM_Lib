#ifndef CLASSIFIER_HPP
#define CLASSIFIER_HPP

#include <eigen3/Eigen/Core>
#include "data.hpp"

namespace iagmm {

/**
 * @brief The Classifier class
 * Mother class of all classifier. this is an abstract class.
 */
class Classifier{
public:
    /**
     * @brief default Constructor
     */
    Classifier(){}

    /**
     * @brief basic constructor. Specify the dimension of the features space and the number of class
     * @param dimension of the feature space
     * @param nbr_class number of class
     */
    Classifier(int dimension, int nbr_class = 2) :
        _dimension(dimension), _nbr_class(nbr_class){}

    /**
     * @brief copy constructor
     * @param c a classifier.
     */
    Classifier(const Classifier &c) :
        _nbr_class(c._nbr_class),
        _dimension(c._dimension),
        _samples(c._samples){}

    ~Classifier(){}

    /**
     * @brief compute_estimation : Compute a probability for a "sample" to be of the class "lbl"
     * @param sample
     * @param label of the class
     * @return the probability of the sample to be part of the class lbl
     */
    virtual double compute_estimation (const Eigen::VectorXd& sample, int lbl) = 0;

    /**
     * @brief add a sample to the training set of the classifier
     * @param sample
     * @param the label of the classifier
     */
    void add(const Eigen::VectorXd& sample, int lbl){
        _samples.add(lbl,sample);
    }

    /**
     * @brief dataset_size
     * @return size of the training dataset
     */
    size_t dataset_size() const {return _samples.size();}

    /**
     * @brief get_samples
     * @return all the samples of the training dataset
     */
    const TrainingData& get_samples() const {return _samples;}

protected:
    int _nbr_class;
    int _dimension;
    TrainingData _samples;

};

}
#endif //CLASSIFIER_HPP
