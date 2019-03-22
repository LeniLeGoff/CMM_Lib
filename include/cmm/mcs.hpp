#ifndef MCS_HPP
#define MCS_HPP

#include <cmm/mcs_fct.hpp>
#include <eigen3/Eigen/IterativeLinearSolvers>
#include "boost/random.hpp"

#ifndef NO_PARALLEL
#include <tbb/tbb.h>
#endif


namespace cmm
{

/**
 * @brief An implementation of a Multi-Classifier System. This class combined several classifier with a combination fonction. Each classifier is specific to a feature space.
 * The current implementation of MCS consider the parameters being the confidences of classification of each classifier. But it is easy to extend to other type of weighting.
 */
class MCS{

public:

    /**
     * @brief default constructor
     */
    MCS(){
        srand(time(NULL));
        _gen.seed(rand());
    }

    /**
     * @brief basic constructor. Initiate the class from a list of classifiers associate with a name, a combination function and a parameter function.
     * The combination function combine all the prediction of the different classifiers weighted by parameters possibly filterd by a function called parameter function.
     * @param the list of classifiers
     * @param Combination function
     * @param parameter function
     */
    MCS(std::map<std::string,Classifier::Ptr>& class_list,const comb_fct_t& cmb,const param_fct_t param_fct):
        _classifiers(class_list), _comb_fct(cmb), _param_fct(param_fct){
        _dimension = _classifiers.begin()->second->get_dimension();
        _nbr_class = _classifiers.begin()->second->get_nbr_class();
        _parameters = Eigen::VectorXd::Constant(class_list.size(),1.);
        srand(time(NULL));
        _gen.seed(rand());
    }

    /**
     * @brief copy constructor
     * @param mcs
     */
    MCS(const MCS& mcs) :
        _dimension(mcs._dimension), _nbr_class(mcs._nbr_class), _parameters(mcs._parameters),
        _classifiers(mcs._classifiers), _comb_fct(mcs._comb_fct), _param_fct(mcs._param_fct){
        srand(time(NULL));
        _gen.seed(rand());
    }

    /**
     * @brief compute the estimation for a sample represented with a list of features
     * @param list of features.
     * @return a list of probability of membership to each class
     */
    std::vector<double> compute_estimation(const std::map<std::string,Eigen::VectorXd> &sample);

    /**
     * @brief add a new sample in the dataset
     * @param sample
     * @param label
     */
    void add(const std::map<std::string,Eigen::VectorXd> &sample, int lbl);

    /**
     * @brief update each classifier according to the current dataset
     */
    void update();

    /**
     * @brief choose the next sample to add in the dataset from a list of unknown samples
     * @param samples
     * @param choice_dist_map
     * @return index of the new samples.
     */
    int next_sample(const std::map<std::string, std::vector<std::pair<Eigen::VectorXd, std::vector<double>> > > &samples, Eigen::VectorXd& choice_dist_map);

    /**
     * @brief accessor to the classifiers
     * @return
     */
    std::map<std::string,Classifier::Ptr>& access_classifiers(){return _classifiers;}

    //** GETTERS & SETTERS
    void set_samples(std::string mod, Data& data);
    void set_parameters(const Eigen::VectorXd& param){_parameters = param;}
    int get_nb_samples(){return _classifiers.begin()->second->get_samples().size();}
    const Data& get_samples(){return _classifiers.begin()->second->get_samples();}
    int get_nbr_class(){return _nbr_class;}
    //*/

private:
    int _dimension; /**<dimension of the feature space*/
    int _nbr_class; /**<number of class*/
    Eigen::VectorXd _parameters; /**<the parameters use as weight in the combination function*/
    std::map<std::string,Classifier::Ptr> _classifiers; /**<list of the classifier*/
    comb_fct_t _comb_fct; /**<The combination function*/
    param_fct_t _param_fct;/**<The function to filter the parameters*/
    boost::random::mt19937 _gen;

};
}

#endif // MCS_HPP
