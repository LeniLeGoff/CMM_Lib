#ifndef CollabMM_HPP
#define CollabMM_HPP

#include <iostream>
#include <memory>
#include <math.h>
#include <vector>
#include <map>

#include "boost/random.hpp"

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Eigenvalues>

#include <boost/serialization/vector.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/shared_ptr.hpp>

#ifndef NO_PARALLEL
    #include <tbb/tbb.h>
#endif

#include "component.hpp"
#include "classifier.hpp"


namespace cmm{


class CollabMM : public Classifier {
public:


    typedef enum update{BATCH,STOCHASTIC} update_mode_t;
    typedef std::map<int, std::vector<Component::Ptr>> model_t;

    /**
     * @brief default constructor
     */
    CollabMM(){
        srand(time(NULL));
        _gen.seed(rand());
        _distance = [](const Eigen::VectorXd& s1,const Eigen::VectorXd& s2) -> double {
            return (s1 - s2).squaredNorm();
        };
    }

    /**
     * @brief basic constructor initiate with dimension of the feature space, number of class and optionally a max number of components per class
     * @param dimension
     * @param nbr_class
     * @param max_nb_comp
     */
    CollabMM(int dimension, int nbr_class, int max_nb_comp = 0) :
        Classifier(dimension,nbr_class){
        for(int i = 0; i < nbr_class; i++)
            _model.emplace(i,std::vector<Component::Ptr>());

        srand(time(NULL));
        _gen.seed(rand());
        _distance = [](const Eigen::VectorXd& s1,const Eigen::VectorXd& s2) -> double {
            return (s1 - s2).squaredNorm();
        };
        _max_nb_components = max_nb_comp;
    }


    /**
     * @brief Constructor which initiate by copying an existant model
     * @param model
     */
    CollabMM(const model_t& model){

        _dimension = model.at(0)[0]->get_dimension();
        _nbr_class = model.size();
        for(const auto& comps : model){
            _model.emplace(comps.first,std::vector<Component::Ptr>());
            for(const auto& comp : comps.second)
                _model[comps.first].push_back(Component::Ptr(new Component(*(comp))));
        }
        srand(time(NULL));
        _gen.seed(rand());
        _distance = [](const Eigen::VectorXd& s1,const Eigen::VectorXd& s2) -> double {
            return (s1 - s2).squaredNorm();
        };
    }


//    CollabMM(const CollabMM& colmm) :
//        Classifier(colmm),
//        CollabMM(colmm._model),_gen(colmm._gen),
//        _last_index(colmm._last_index), _last_label(colmm._last_label),
//        _membership(colmm._membership), _update_mode(colmm._update_mode),
//    _llhood_drive(colmm._llhood_drive), _max_nb_components(colmm._max_nb_components){}

    ~CollabMM(){
        for(auto& comps: _model)
            for(auto& c : comps.second)
                c.reset();
    }

    /**
     * @brief compute the estimation for a sample
     * @param sample
     * @return a vector of probability membership to each class
     */
    std::vector<double> compute_estimation(const Eigen::VectorXd& sample) const;

    /**
     * @brief accessor to the model
     * @return reference to the model
     */
    model_t& model(){return _model;}

    /**
     * @brief constant accessor to the model
     * @return constant reference to the model
     */
    const model_t& model() const {return _model;}

    /**
     * @brief add a sample to the model with its label
     * @param sample
     * @param label
     */
    void add(const Eigen::VectorXd &sample, int lbl);

    /**
     * @brief append a list of samples associated with their labels
     * @param vector of samples
     * @param vector of label with same indexing as the samples
     */
    void append(const std::vector<Eigen::VectorXd> &samples,const std::vector<int>& lbl);

    /**
     * @brief append a sample with its label and update the parameters of the model
     * @param sample
     * @param label
     * @return return the index of component in which the new sample was added.
     */
    int append(const Eigen::VectorXd &samples,const int& lbl);

    /**
     * @brief update the model according to the current dataset
     */
    void update();

    /**
     * @brief update model in batch mode: by reevaluating all the component
     */
    void update_model();

    /**
     * @brief update model stochastic mode : only reeavulate the last modified component and by reevaluating one random component per class.
     * @param index of the last component modified
     * @param label of the class of the last comment modified
     */
    void update_model(int ind, int lbl);

    /**
     * @brief update the factors of all components
     */
    void update_factors();

    /**
     * @brief compute the confidence of classification for the sample X
     * @param a sample
     * @return a real value representing the confidence of classification
     */
    double confidence(const Eigen::VectorXd& X) const;

    /**
     * @brief choice of the next sample among a set of samples based on the uncertainty and the confidence of the classifier
     * @param set of new unlabelled samples
     * @param choice distribution map which represents the probability of choice the proposed samples
     * @return index of the chosen sample
     */
    int next_sample(const std::vector<std::pair<Eigen::VectorXd,std::vector<double>>> &samples,
                    Eigen::VectorXd& choice_dist_map);

    /**
     * @brief choice of the next sample among a set of samples based on the uncertainty and the confidence of the classifier.
     * And by multiplying the choice distribution by a prior distribution
     * @param set of new unlabelled samples
     * @param choice distribution map which represents the probability of choice the proposed samples
     * @param prior distribution
     * @return index of the chosen sample
     */
    int next_sample(const std::vector<std::pair<Eigen::VectorXd,std::vector<double>>> &samples,
                    Eigen::VectorXd& choice_dist_map, Eigen::VectorXd& filter);


    /**
     * @brief estimate probability of membership in the class lbl of a set of unknown samples
     * @param samples
     * @param output predictions
     * @param label of the asked class. default value 1
     */
    void estimate_features(const std::vector<Eigen::VectorXd> &samples, Eigen::VectorXd& predictions, int lbl = 1);

    /**
     * @brief create a new component with default covariance matrix and sample as center
     * @param sample
     * @param label
     */
    void new_component(const Eigen::VectorXd &samples, int label);


    /**
     * @brief k nearst neighbor
     * @param samples
     * @param output
     * @param k
     */
    void knn(const Eigen::VectorXd& center,Data& output, int k);

    /**
     * @brief compute the number of samples contain in each component
     * @return
     */
    int number_of_samples(){
        int nbr_s = 0;
        for(const auto& components : _model)
            for(const auto& c : components.second)
                nbr_s += c->size();

        return nbr_s;
    }

    template <typename archive>
    void serialize(archive& arch, const unsigned int v){
        arch & _nbr_class;
        arch & _dimension;
        arch & _model;
    }

    /**
     * @brief readable information about the model
     * @return a string containing printable information
     */
    std::string print_info();


    /**
     * @brief compute the loglikelihood of the whole model
     * @return
     */
    double loglikelihood();

    /**
     * @brief compute the loglikelihood for one specific gaussian mixture model
     * @param label of the GMM
     * @return
     */
    double loglikelihood(int label);

    //** GETTERS & SETTERS
    void set_dataset_size_max(int dsm){_dataset_size_max = dsm;}
    int get_dataset_size_max(){return _dataset_size_max;}
    void set_update_mode(update_mode_t um){_update_mode = um;}
    void set_max_nb_components(int max_nb){_max_nb_components = max_nb;}
    void set_loglikelihood_driver(bool ll){_llhood_drive = ll;}
    bool get_loglikelihood_driver(){return _llhood_drive;}
    void use_confidence(bool c){_use_confidence = c;}
    void use_uncertainty(bool u){_use_uncertainty = u;}
    bool get_use_confidence(){return _use_confidence;}
    bool get_use_uncertainty(){return _use_uncertainty;}
    //*/

    bool skip_bootstrap = false; /**< boolean attribute indicating if a bootstrap phase should be applied before choosing the next sample. During the bootstrap phase the choice is random. The phase is off 10 samples in the dataset**/

private:

    /**
     * @brief merge operation to merge the component comp with another component of the same class
     * @param candidate component pointer
     * @return true of the component was merged
     */
    bool _merge(const Component::Ptr& comp);

    /**
     * @brief compute a score for a specific component based on the k nearest neighbors samples from its center. The score is based on the prediction of these samples.
     * @param index of the component
     * @param label of the component
     * @return
     */
    double _component_score(int i, int lbl);

    /**
     * @brief split operation to split a component comp
     * @param candidate component pointer
     * @return true if the component was split
     */
    bool _split(const Component::Ptr& comp);

    /**
     * @brief update factors of the GMM corresponding to the class of label lbl
     * @param label
     */
    void _update_factors(int lbl);

    /**
     * @brief compute the coefficients for the intersection condition between the component of index ind1 and label lbl1 and the component of index ind2 and label lbl2
     * @param ind1
     * @param lbl1
     * @param ind2
     * @param lbl2
     * @return return
     */
    std::pair<double,double> _coeff_intersection(int ind1, int lbl1, int ind2, int lbl2);

    model_t _model; /**<the model : a gaussian mixture model per class*/

    update_mode_t _update_mode = STOCHASTIC; /**<mode of update of CMM : Stochastic or Batch. Stochastic for online learning and Batch for offline*/

    boost::random::mt19937 _gen;

    bool _llhood_drive = false;
    bool _use_confidence = true;
    bool _use_uncertainty = true;

    int _last_index;
    int _last_label;

    int _max_nb_components = 0;

    int _dataset_size_max = 1000;

    /**
     * @brief The _score_calculator class is a helper class to compute the loglikelihood in parallel with parallel reduce algo of intel tbb.
     */
    class _score_calculator{
    public:
        _score_calculator(CollabMM* model, Data samples, bool all_samples = true, int lbl = 0) :
            _model(model), _samples(samples), _sum(0), _label(lbl), _all_samples(all_samples){}

#ifndef NO_PARALLEL
        _score_calculator(const _score_calculator &sc, tbb::split) :
            _model(sc._model), _samples(sc._samples), _sum(0), _label(sc._label), _all_samples(sc._all_samples){}

        void operator ()(const tbb::blocked_range<size_t>& r);
        void join(const _score_calculator& sc){
            _sum += sc._sum;
        }
#endif
        double compute();

    private:
        CollabMM* _model;
        double _sum;
        int _label;
        bool _all_samples;
        Data _samples;
    };
};
}

#endif //CollabMM_HPP
