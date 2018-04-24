#ifndef TRAINER_HPP
#define TRAINER_HPP

#include <iostream>
#include <string>
#include <eigen3/Eigen/Eigen>
#include <iagmm/data.hpp>
#include <boost/random.hpp>

namespace iagmm{

template <class Classifier>
class Trainer {
public:
    Trainer(){
        srand(time(NULL));
        _gen.seed(rand());
        _batch_size = 10;
    }
    Trainer(const std::string data_file, int batch_size = 10, float test_set = 0.2) :
        _batch_size(batch_size), _test_set(test_set){
        srand(time(NULL));
        _gen.seed(rand());

        int dimension, nbr_class;
        _data.load_yml(data_file,dimension,nbr_class);
        _classifier = Classifier(dimension, nbr_class);
    }

    Trainer(const TrainingData& data, int dimension, int nbr_class, int batch_size = 10, float test_set = 0.2) :
        _data(data), _batch_size(batch_size), _test_set(test_set){
        srand(time(NULL));
        _gen.seed(rand());

        _classifier = Classifier(dimension, nbr_class);
    }

    Trainer(const Trainer& t) :
        _data(t._data), _train_data(t._train_data), _test_data(t._test_data),
        _test_set(t._test_set), _batch_size(t._batch_size), _classifier(t._classifier){}

    void initialize(){
        //*build the training and test data set
        int index;
        std::vector<Eigen::VectorXd> train_data, test_data;
        int size;
        for(int i = 0 ; i < _classifier.get_nbr_class(); i++){
            train_data = _data.get_data(i);
            test_data.clear();
            size = train_data.size();
            while((float)test_data.size()/(float)size < _test_set){
                boost::random::uniform_int_distribution<> dist(0,train_data.size());
                index = dist(_gen);
                test_data.push_back(train_data[index]);
                train_data.erase(train_data.begin() + index);
            }
            for(int j = 0; j < train_data.size(); j++)
                _train_data.add(i,train_data[j]);
            for(int j = 0; j <test_data.size(); j++)
                _test_data.add(i,test_data[j]);
        }
//        std::vector<int> subst(_train_data.size());
//        for(int i = 0; i < _train_data.size()/2; i++){
//            boost::random::uniform_int_distribution<> dist(_train_data.size()/2+1,_train_data.size());
//            int index = dist(_gen);
//            subst[i] = index;
//            subst[index] = i;
//        }
//        for(int i = 0; i < _train_data.size(); i++){

//        }

        //*/
    }

    void epoch(){
        int n;
//        boost::random::uniform_int_distribution<> dist(0,_train_data.size()-1);
        for(int i = _g_count; i < _batch_size+_g_count && i < _train_data.size(); i++){
//            n = dist(_gen);
            _classifier.add(_train_data[i].second,_train_data[i].first);
        }
        _classifier.update();
        _g_count += _batch_size;
        if(_g_count > _train_data.size() - _batch_size)
            _g_count = 0;
    }

    double test(){
        double error = 0;
        for(int i = 0; i < _test_data.size(); i++){
            error += 1 - _classifier.compute_estimation(_test_data[i].second)[_test_data[i].first];
        }
        error = error/(double) _test_data.size();
        return error;
    }

    Classifier& access_classifier(){return _classifier;}

private:
    TrainingData _data;
    TrainingData _test_data;
    TrainingData _train_data;
    Classifier _classifier;

    int _batch_size;
    float _test_set;
    int _g_count = 0;

    boost::random::mt19937 _gen;

};
} // iagmm

#endif //TRAINER_HPP
