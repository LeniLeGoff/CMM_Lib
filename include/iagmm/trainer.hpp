#ifndef TRAINER_HPP
#define TRAINER_HPP

#include <iostream>
#include <string>
#include <eigen3/Eigen/Eigen>
#include <iagmm/data.hpp>
#include <boost/random.hpp>

#ifndef NO_PARALLEL
    #include <tbb/tbb.h>
#endif

namespace iagmm{

template <class Classifier>
class Trainer {
public:
    Trainer(){
        srand(time(NULL));
        _gen.seed(rand());
        _batch_size = 10;
    }
    Trainer(const std::string &data_file, int batch_size = 10) :
        _batch_size(batch_size){
        srand(time(NULL));
        _gen.seed(rand());

        int dimension, nbr_class;
        _train_data.load_yml(data_file,dimension,nbr_class);
        _classifier = Classifier(dimension, nbr_class);
    }

    Trainer(const std::string &train_data_file, const std::string &test_data_file, int batch_size = 10) :
        _batch_size(batch_size){
        srand(time(NULL));
        _gen.seed(rand());

        int dimension, nbr_class;
        _train_data.load_yml(train_data_file,dimension,nbr_class);
        _test_data.load_yml(test_data_file,dimension,nbr_class);
        _classifier = Classifier(dimension, nbr_class);
    }

    Trainer(const TrainingData& data, int dimension, int nbr_class, int batch_size = 10) :
        _train_data(data), _batch_size(batch_size){
        srand(time(NULL));
        _gen.seed(rand());

        _classifier = Classifier(dimension, nbr_class);
    }

    Trainer(const TrainingData& train_data, const TrainingData& test_data,  int dimension, int nbr_class, int batch_size = 10) :
        _train_data(train_data), _test_data(test_data), _batch_size(batch_size){
        srand(time(NULL));
        _gen.seed(rand());

        _classifier = Classifier(dimension, nbr_class);
    }



    Trainer(const Trainer& t) :
        _train_data(t._train_data), _test_data(t._test_data),
        _batch_size(t._batch_size), _classifier(t._classifier){}

//    void initialize(){
//        //*build the training and test data set
//        int index;
//        std::vector<Eigen::VectorXd> train_data, test_data;
//        int size;
//        for(int i = 0 ; i < _classifier.get_nbr_class(); i++){
//            train_data = _data.get_data(i);
//            test_data.clear();
//            size = train_data.size();
//            while((float)test_data.size()/(float)size < _test_set){
//                boost::random::uniform_int_distribution<> dist(0,train_data.size());
//                index = dist(_gen);
//                test_data.push_back(train_data[index]);
//                train_data.erase(train_data.begin() + index);
//            }
//            for(int j = 0; j < train_data.size(); j++)
//                _train_data.add(i,train_data[j]);
//            for(int j = 0; j <test_data.size(); j++)
//                _test_data.add(i,test_data[j]);
//        }
////        std::vector<int> subst(_train_data.size());
////        for(int i = 0; i < _train_data.size()/2; i++){
////            boost::random::uniform_int_distribution<> dist(_train_data.size()/2+1,_train_data.size());
////            int index = dist(_gen);
////            subst[i] = index;
////            subst[index] = i;
////        }
////        for(int i = 0; i < _train_data.size(); i++){

////        }

//        //*/
//    }

    void epoch(){

        int n;
        int upper_bound = _g_count + 10*_batch_size;
        if(_g_count + 10*_batch_size > _train_data.size())
            upper_bound -= upper_bound - _train_data.size();
        boost::random::uniform_int_distribution<> dist(_g_count,upper_bound);
        for(int i = 0; i < _batch_size; i++){
            n = dist(_gen);
            _classifier.add(_train_data[n].second,_train_data[n].first);
        }
        _classifier.update();
        _g_count += _batch_size;
        if(_g_count > _train_data.size())
            _g_count = 0;
    }

    double test(){
        double error = 0;
#ifdef NO_PARALLEL
        for(int i = 0; i < _test_data.size(); i++){
            error += 1 - _classifier.compute_estimation(_test_data[i].second)[_test_data[i].first];
        }
#else
        _error_computer ec(_classifier,_test_data);
        tbb::parallel_reduce(tbb::blocked_range<size_t>(0,_test_data.size()),ec);
        error = ec.get_error();
#endif
        error = error/(double) _test_data.size();
        return error;
    }

    Classifier& access_classifier(){return _classifier;}
    void set_train_data(const TrainingData& data){_train_data = data;}
    void set_test_data(const TrainingData& data){_test_data = data;}


private:
    TrainingData _test_data;
    TrainingData _train_data;
    Classifier _classifier;

    int _batch_size;
    int _g_count = 0;

    boost::random::mt19937 _gen;

    class _error_computer{
    public:
        _error_computer(Classifier& model, TrainingData samples) :
            _model(model), _samples(samples), _sum(0){}

#ifndef NO_PARALLEL
        _error_computer(const _error_computer &sc, tbb::split) :
            _model(sc._model), _samples(sc._samples), _sum(0){}

        void operator ()(const tbb::blocked_range<size_t>& r){
            double sum = _sum;
            for(int i = r.begin(); i != r.end(); i++)
                sum += 1 - _model.compute_estimation(_samples[i].second)[_samples[i].first];
            _sum = sum;
        }

        void join(const _error_computer& sc){
            _sum += sc._sum;
        }
#endif

        double get_error(){return _sum;}

    private:
        Classifier _model;
        double _sum;
        TrainingData _samples;
    };

};
} // iagmm

#endif //TRAINER_HPP
