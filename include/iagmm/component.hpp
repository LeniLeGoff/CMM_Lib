#ifndef COMPONENT_HPP
#define COMPONENT_HPP

#include <memory>
#include <vector>

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>

#include <iagmm/serialization.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/shared_ptr.hpp>


#define PI 3.14159265359


namespace iagmm {

typedef std::vector<Eigen::VectorXd> samples_t;

/**
 * @brief The Component class
 * This class is relative to the GMM classifier. A component represent a gaussian. It is constitute of a set of sample of same class, a dimension of the feature space.
 *
 * This class compute Gaussian distribution with a set of sample.
 *
 * TODO put the formula
 *
 */
class Component{
public:

    typedef boost::shared_ptr<Component> Ptr;
    typedef boost::shared_ptr<const Component> ConstPtr;

    /**
     * @brief default constructor
     */
    Component(){}

    /**
     * @brief Basic constructor
     * @param dimension of the feature space
     * @param label of the class
     */
    Component(int dimension, int lbl)
        : _dimension(dimension), _label(lbl), _factor(0),_max(1){}

    /**
     * @brief Copy constructor
     * @param  a component
     */
    Component(const Component& c) :
        _covariance(c._covariance), _mu(c._mu), _label(c._label),
        _samples(c._samples), _dimension(c._dimension), _factor(c._factor),_max(c._max){}

    /**
     * @brief update_parameters
     * Erase the previous parameters and compute them with the samples stored in the component.
     * CAUTION: Be sure to have all the training samples before using this function
     */
    void update_parameters();

//    /**
//     * @brief update_parameters
//     * update the parameters by using recursive formulas of the sample average and the sample covariance
//     * This method consider only the latest sample
//     */
//    void update_parameters();
    double compute_multivariate_normal_dist(Eigen::VectorXd X) const;
    Component::Ptr merge(const Component::Ptr c);
    Component::Ptr split();


    //Modifiers
    void add(Eigen::VectorXd sample){
        _samples.push_back(sample);
//        _incr_parameters(sample);
    }
    void clear(){_samples.clear();}

    //Statistics
    double component_score() const;
    double get_standard_deviation() const;
    std::vector<double> get_intern_estimation() const;
    void compute_eigenvalues(Eigen::VectorXd& eigenvalues, Eigen::MatrixXd& eigenvectors) const;
    double entropy();

    /**
     * @brief compute the mahalanobis distance
     * @param X a sample
     * @return
     */
    double distance(const Eigen::VectorXd& X) const;

    //Getters & Setters
    void set_factor(double f){_factor = f;}
    double get_factor() const {return _factor;}
    int get_label() const {return _label;}
    const Eigen::VectorXd& get_mu() const {return _mu;}
    const Eigen::VectorXd& get_sample(int i) const {return _samples[i];}
    const std::vector<Eigen::VectorXd>& get_samples() const {return _samples;}
    int size() const {return _samples.size();}
    int get_dimension() const {return _dimension;}
    const Eigen::MatrixXd& get_covariance() const {return _covariance;}

    std::string print_parameters() const;


    double diameter(){
        double dist , diameter = (_samples[0] - _mu).squaredNorm();
        for(auto& s : _samples){
            dist = (s - _mu).squaredNorm();
            if(dist > diameter)
                diameter = dist;
        }
        return diameter;
    }

    Eigen::MatrixXd covariance_pseudoinverse() const;

    template <typename archive>
    void serialize(archive& arch, const unsigned int v){
        boost::serialization::serialize(arch,_covariance,v);
        boost::serialization::serialize(arch,_mu,v);
        arch & _samples;
        arch & _dimension;
        arch & _factor;
        arch & _label;
    }

private:
    void _incr_parameters(const Eigen::VectorXd& X);


    Eigen::MatrixXd _covariance;
    Eigen::VectorXd _mu;
    std::vector<Eigen::VectorXd> _samples;
    int _dimension;
    double _factor;
    int _label;
    double _max;
};

}

#endif //COMPONENT_HPP
