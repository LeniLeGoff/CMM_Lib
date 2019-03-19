#ifndef GMM_HPP
#define GMM_HPP

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


class GMM : public Classifier {
public:

    typedef enum update{BATCH,STOCHASTIC} update_mode_t;

    int max_component;
//    typedef boost::shared_ptr<GMM> Ptr;
//    typedef boost::shared_ptr<const GMM> ConstPtr;

    typedef std::map<int, std::vector<Component::Ptr>> model_t;

    GMM(){
        srand(time(NULL));
        _gen.seed(rand());
        _distance = [](const Eigen::VectorXd& s1,const Eigen::VectorXd& s2) -> double {
            return (s1 - s2).squaredNorm();
        };
    }



    GMM(int dimension, int nbr_class, int max_nb_comp = 0) :
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



    GMM(const model_t& model){

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

    GMM(const GMM& gmm) :
        Classifier(gmm),
        _model(gmm._model),_gen(gmm._gen),
        _last_index(gmm._last_index), _last_label(gmm._last_label),
        _membership(gmm._membership), _update_mode(gmm._update_mode),
    _llhood_drive(gmm._llhood_drive), _max_nb_components(gmm._max_nb_components){}

    ~GMM(){
        for(auto& comps: _model)
            for(auto& c : comps.second)
                c.reset();
    }

//    void operator()(const tbb::blocked_range<size_t>& r);

    std::vector<double> compute_estimation(const Eigen::VectorXd& sample) const;
    void compute_normalisation();
    model_t& model(){return _model;}
    const model_t& model() const {return _model;}
//    double get_result(int lbl){
//        double sum_of_sums;
//        for(const auto& sum : _sum_map)
//            sum_of_sums +=  sum.second;
//        return _sum_map[lbl]/(sum_of_sums);
//    }

    void add(const Eigen::VectorXd &sample, int lbl);
    void append(const std::vector<Eigen::VectorXd> &samples,const std::vector<int>& lbl);
    int append(const Eigen::VectorXd &samples,const int& lbl);
    void append_EM(const Eigen::VectorXd &samples,const int& lbl);

    void update();
    void update_model();
    void update_model(int ind, int lbl);

    std::vector<int> find_closest_components(double& min_dist, int lbl);

    int find_closest(int i, double& min_dist, int lbl);

    void update_factors();
    double unit_factor();

    std::vector<double> model_scores();

    double entropy(int i, int sign);

    Eigen::VectorXd mean_shift(const Eigen::VectorXd& X, int lbl);

    double confidence(const Eigen::VectorXd& X) const;

    double novelty(const Eigen::VectorXd& feature);
    int next_sample(const std::vector<std::pair<Eigen::VectorXd,std::vector<double>>> &samples,
                    Eigen::VectorXd& choice_dist_map);
    int next_sample(const std::vector<std::pair<Eigen::VectorXd,std::vector<double>>> &samples,
                    Eigen::VectorXd& choice_dist_map, Eigen::VectorXd& filter);
    void estimate_features(const std::vector<Eigen::VectorXd> &samples, Eigen::VectorXd& predictions, int lbl = 1);

    void EM_init();
    void EM_step();
    void new_component(const Eigen::VectorXd &samples, int label);


    /**
     * @brief k nearst neighbor
     * @param samples
     * @param output
     * @param k
     */
    void knn(const Eigen::VectorXd& center,TrainingData& output, int k);


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

    std::string print_info();

    double get_normalisation(){return _normalisation;}

    double compute_quality(const Eigen::VectorXd&,int lbl);
//    void update_dataset();
//    void update_dataset_thres(double);

    void set_dataset_size_max(int dsm){_dataset_size_max = dsm;}
    int get_dataset_size_max(){return _dataset_size_max;}

    void set_update_mode(update_mode_t um){_update_mode = um;}

    void set_max_nb_components(int max_nb){_max_nb_components = max_nb;}

    double loglikelihood();
    double loglikelihood(int label);
    void set_loglikelihood_driver(bool ll){_llhood_drive = ll;}
    bool get_loglikelihood_driver(){return _llhood_drive;}
    void use_confidence(bool c){_use_confidence = c;}
    void use_uncertainty(bool u){_use_uncertainty = u;}
    void use_novelty(bool n){_use_novelty = n;}
    bool get_use_confidence(){return _use_confidence;}
    bool get_use_uncertainty(){return _use_uncertainty;}
    bool get_use_novelty(){return _use_novelty;}

    bool skip_bootstrap = false;

private:

    bool _merge(const Component::Ptr& comp);
    double _component_score(int i, int lbl);
    bool _split(const Component::Ptr& comp);
    void _update_factors(int lbl);

    void _redescription();

    void _expectation(int lbl);
    void _maximisation(int lbl);

    std::pair<double,double> _coeff_intersection(int ind1, int lbl1, int ind2, int lbl2);

    model_t _model;
    std::map<int, Eigen::MatrixXd> _membership;

    std::vector<GMM> _base_classifiers;

    update_mode_t _update_mode = STOCHASTIC;

    boost::random::mt19937 _gen;
    double _normalisation;

    bool _llhood_drive = false;
    bool _use_confidence = true;
    bool _use_uncertainty = true;
    bool _use_novelty = false;

    int _last_index;
    int _last_label;

    int _max_nb_components = 0;

    int _dataset_size_max = 1000;

    class _score_calculator{
    public:
        _score_calculator(GMM* model, TrainingData samples, bool all_samples = true, int lbl = 0) :
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
        GMM* _model;
        double _sum;
        int _label;
        bool _all_samples;
        TrainingData _samples;
    };
};
}

#endif //GMM_HPP
