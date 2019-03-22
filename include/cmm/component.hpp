#ifndef COMPONENT_HPP
#define COMPONENT_HPP

#include <memory>
#include <vector>

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/SVD>

#include <cmm/serialization.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/shared_ptr.hpp>


#define PI 3.14159265359


namespace cmm {

typedef std::vector<Eigen::VectorXd> samples_t;

/**
 * @brief The Component class
 * This class is relative to the GMM classifier. A component represent a gaussian. 
 * It is constituted of a set of sample of same class, a dimension of the feature space.
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
        : _dimension(dimension), _label(lbl), _factor(0){}

    /**
     * @brief Copy constructor
     * @param  a component
     */
    Component(const Component& c) :
        _covariance(c._covariance), _mu(c._mu), _label(c._label),
        _samples(c._samples), _dimension(c._dimension), _factor(c._factor),
        _size(c._size)
    {}

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
    void merge(const Component::Ptr c);
    Component::Ptr split();


    //Modifiers
    void add(Eigen::VectorXd sample){
        _samples.push_back(sample);
        _size++;
    }
    void clear(){_samples.clear();}

    //Statistics
    double get_standard_deviation() const;
    void compute_eigenvalues(Eigen::VectorXd& eigenvalues, Eigen::MatrixXd& eigenvectors) const;
    double entropy();

    /**
     * @brief compute the mahalanobis distance
     * @param X a sample
     * @return
     */
    double distance(const Eigen::VectorXd& X) const;

    /**
     * @brief compute intersectation condition of this with comp
     * @param comp
     * @return true if the components are intersecting false otherwise
     */
    bool intersect(const Component::Ptr comp) const;

    //** Getters & Setters
    void set_factor(double f){_factor = f;}
    double get_factor() const {return _factor;}
    int get_label() const {return _label;}
    const Eigen::VectorXd& get_mu() const {return _mu;}
    void set_mu(const Eigen::VectorXd& mu){_mu = mu;}
    const Eigen::VectorXd& get_sample(int i) const {return _samples[i];}
    const std::vector<Eigen::VectorXd>& get_samples() const {return _samples;}
    int size() const {return _size;}
    void set_size(int n){_size = n;}
    int get_dimension() const {return _dimension;}
    const Eigen::MatrixXd& get_covariance() const {return _covariance;}
    void set_covariance(const Eigen::MatrixXd& covariance){_covariance = covariance;}
    //*/

    /**
     * @brief remove a sample from the component by index
     * @param index of the samples
     */
    void remove_sample(int i){
        _samples.erase(_samples.begin() + i);
        _samples.shrink_to_fit();
    }

    /**
     * @brief printable format of the parameters values
     * @return a string containing the value of the variable
     */
    std::string print_parameters() const;

    /**
     * @brief Compute the true inverse of the covariance. It is assumed that the covariance matrix is invertible
     * @param inverse matrix
     * @param determinant of the covariance matrix
     */
    void covariance_inverse(Eigen::MatrixXd& inverse, double& determinant) const;

    /**
     * @brief Compute the pseudo inverse of the covariance matrix by using its singular value decomposition
     * @param pseudo inverse matrix
     * @param pseudo determinant of the covariance matrix
     */
    void covariance_pseudoinverse(Eigen::MatrixXd& inverse, double& determinant) const;


    template <typename archive>
    void serialize(archive& arch, const unsigned int v){
        boost::serialization::serialize(arch,_covariance,v);
        boost::serialization::serialize(arch,_mu,v);
        arch & _samples;
        arch & _dimension;
        arch & _factor;
        arch & _label;
        arch & _size;
    }

    /**
     * @brief add X to the dataset and then do an incremental update of the parameters
     * @param the new samples added in the dataset
     */
    void _incr_parameters(const Eigen::VectorXd& X);

    static double _alpha; /**<hyperparameters controlling the sensibility of intersection criterion*/


private:
    /**
     * @brief check and remove artefakt samples
     */
    void _check_samples();

    Eigen::MatrixXd _covariance; /**<covariance matrix of the normal distribution encoding the component*/
    Eigen::VectorXd _mu; /**<the mean of the normal distribution encoding the component*/
    std::vector<Eigen::VectorXd> _samples; /**<the dataset on which the parameters of the normal distribution are computed*/
    int _size = 0; /**<Number of samples in the dataset*/
    int _dimension; /**<the dimension of the multivariate normal distribution*/
    double _factor; /**<the multiplicator factor of the component used when combined in the mixture*/
    int _label; /**<the label of the component corresponding to the class all the samples belong*/
};

}

#endif //COMPONENT_HPP
