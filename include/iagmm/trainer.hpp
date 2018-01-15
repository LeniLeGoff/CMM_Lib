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
    Trainer(const std::string data_file, int dimension, int nbr_class, int batch_size = 10, float test_set = 0.2) :
        _batch_size(batch_size), _test_set(test_set){
        srand(time(NULL));
        _gen.seed(rand());

        _data.load_yml(data_file);
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
        int i;
        _train_data = TrainingData(_data);
        do{
            boost::random::uniform_int_distribution<> dist(0,_train_data.size());
            i = dist(_gen);
            _test_data.add(_train_data[i]);
            _train_data.erase(i);
        }while((float)_test_data.size()/(float)_train_data.size() < _test_set);
        //*/
    }

    void epoch(){
        int n;
        boost::random::uniform_int_distribution<> dist(0,_train_data.size()-1);
        for(int i = 0; i < _batch_size; i++){
            n = dist(_gen);
            _classifier.add(_train_data[n].second,_train_data[n].first);
        }
        _classifier.update();
    }

    Classifier& access_classifier(){return _classifier;}

private:
    TrainingData _data;
    TrainingData _test_data;
    TrainingData _train_data;
    Classifier _classifier;

    int _batch_size;
    float _test_set;

    boost::random::mt19937 _gen;

    void _split_data(const TrainingData& data){

    }
};
} // iagmm

#endif //TRAINER_HPP
