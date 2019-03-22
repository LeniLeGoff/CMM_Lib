#ifndef TRAINER_HPP
#define TRAINER_HPP

#include <iostream>
#include <string>
#include <eigen3/Eigen/Eigen>
#include <cmm/data.hpp>
#include <boost/random.hpp>
#include <chrono>

#ifndef NO_PARALLEL
    #include <tbb/tbb.h>
#endif

namespace cmm{

template <class Classifier>
/**
 * @brief Class to train a classifier in batch from a test dataset and a train dataset
 */
class Trainer {
public:
    /**
     * @brief Default constructor
     */
    Trainer(){
        srand(time(NULL));
        _gen.seed(rand());
        _batch_size = 10;
    }

    /**
     * @brief construct the class from one dataset file which will be the train dataset
     * @param dataset file in yaml format
     * @param size of the batch. Default value is 10
     */
    Trainer(const std::string &data_file, int batch_size = 10) :
        _batch_size(batch_size){
        srand(time(NULL));
        _gen.seed(rand());

        int dimension, nbr_class;
        _train_data.load_yml(data_file,dimension,nbr_class);
        _classifier = Classifier(dimension, nbr_class);
    }

    /**
     * @brief Construct the class from two dataset files: a train dataset and a test dataset.
     * @param train dataset file in yaml format
     * @param test dataset file in yaml format
     * @param size of the batch. Default value is 10
     */
    Trainer(const std::string &train_data_file, const std::string &test_data_file, int batch_size = 10) :
        _batch_size(batch_size){
        srand(time(NULL));
        _gen.seed(rand());

        int dimension, nbr_class;
        _train_data.load_yml(train_data_file,dimension,nbr_class);
        _test_data.load_yml(test_data_file,dimension,nbr_class);
        _classifier = Classifier(dimension, nbr_class);
    }

    /**
     * @brief Construct from a train dataset
     * @param dataset
     * @param dimension of the data
     * @param number of classes
     * @param size of the batch. Default value is 10
     */
    Trainer(const Data& data, int dimension, int nbr_class, int batch_size = 10) :
        _train_data(data), _batch_size(batch_size){
        srand(time(NULL));
        _gen.seed(rand());

        _classifier = Classifier(dimension, nbr_class);
    }

    /**
     * @brief Construct from a train dataset and a test dataset
     * @param train dataset
     * @param test dataset
     * @param dimension of the data
     * @param number of classes
     * @param size of the batch. Default value is 10
     */
    Trainer(const Data& train_data, const Data& test_data,  int dimension, int nbr_class, int batch_size = 10) :
        _train_data(train_data), _test_data(test_data), _batch_size(batch_size){
        srand(time(NULL));
        _gen.seed(rand());

        _classifier = Classifier(dimension, nbr_class);
    }

    /**
     * @brief copy constructor
     * @param t
     */
    Trainer(const Trainer& t) :
        _train_data(t._train_data), _test_data(t._test_data),
        _batch_size(t._batch_size), _classifier(t._classifier){}

    /**
     * @brief an epoch evaluate one batch
     */
    void epoch(){

        int n;
        int upper_bound = _g_count + 10*_batch_size;
        if(_g_count + 10*_batch_size > _train_data.size())
            upper_bound -= upper_bound - _train_data.size();
        boost::random::uniform_int_distribution<> dist(_g_count,upper_bound);
        std::chrono::system_clock::time_point timer;
        timer  = std::chrono::system_clock::now();

        for(int i = 0; i < _batch_size; i++){
            n = dist(_gen);
            _classifier.add(_train_data[n].second,_train_data[n].first);
        }
        std::cout << "add step, time spent : "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(
                         std::chrono::system_clock::now() - timer).count() << std::endl;
        timer  = std::chrono::system_clock::now();
        _classifier.update();
        std::cout << "update step, time spent : "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(
                         std::chrono::system_clock::now() - timer).count() << std::endl;
        _g_count += _batch_size;
        if(_g_count > _train_data.size())
            _g_count = 0;
    }

    /**
     * @brief evaluate the test dataset
     * @param a vector of the errors relative to each class
     * @return the global error
     */
    double test(std::vector<double> &errors){
        double error = 0;
        errors.resize(_classifier.get_nbr_class(),0);
#ifdef NO_PARALLEL
        double est;
        for(int i = 0; i < _test_data.size(); i++){
            est =  _classifier.compute_estimation(_test_data[i].second)[_test_data[i].first];
            error += 1 - est;
            errors[_test_data[i].first]+=1-est;
        }
#else
        _error_computer ec(_classifier,_test_data);
        tbb::parallel_reduce(tbb::blocked_range<size_t>(0,_test_data.size()),ec);
        error = ec.get_error();
        errors = ec.get_errors();
#endif
        error = error/(double) _test_data.size();
        for(int i = 0; i < errors.size(); i++)
            errors[i] = errors[i]/(double)_test_data.get_data(i).size();
        return error;
    }

    /**
     * @brief accessor to the classifier
     * @return
     */
    Classifier& access_classifier(){return _classifier;}

    //** GETTERS & SETTERS
    void set_train_data(const Data& data){_train_data = data;}
    void set_test_data(const Data& data){_test_data = data;}
    //*/

private:
    Data _test_data;/**<the test dataset*/
    Data _train_data;/**<the train dataset*/
    Classifier _classifier;/**<the classifier*/

    int _batch_size;
    int _g_count = 0;

    boost::random::mt19937 _gen;

    /**
     * @brief Helper class to compute the error in parallel using parallel reduce algo of intel tbb.
     */
    class _error_computer{
    public:
        _error_computer(Classifier& model, Data samples) :
            _model(model), _samples(samples), _sum(0){
            _sums.resize(_model.get_nbr_class(),0);
        }

#ifndef NO_PARALLEL
        _error_computer(const _error_computer &sc, tbb::split) :
            _model(sc._model), _samples(sc._samples), _sum(0){
            _sums.resize(_model.get_nbr_class(),0);
        }

        void operator ()(const tbb::blocked_range<size_t>& r){
            double sum = _sum;
            double est;
            std::vector<double> sums = _sums;
            for(int i = r.begin(); i != r.end(); i++){
                est =  _model.compute_estimation(_samples[i].second)[_samples[i].first];
                sum += 1 - est;
                sums[_samples[i].first] = 1 - est;
            }
            _sum = sum;
            _sums = sums;
        }

        void join(const _error_computer& sc){
            _sum += sc._sum;
            for(int i = 0; i < _sums.size(); i++)
                _sums[i] += sc._sums[i];
        }
#endif

        double get_error(){return _sum;}
        std::vector<double> get_errors(){return _sums;}

    private:
        Classifier _model;
        double _sum;
        std::vector<double> _sums;
        Data _samples;
    };

};
} // cmm

#endif //TRAINER_HPP
