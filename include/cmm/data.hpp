#ifndef TRAINING_DATA_HPP
#define TRAINING_DATA_HPP

#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <eigen3/Eigen/Core>
#include <yaml-cpp/yaml.h>

namespace cmm{

class Data{

public:

    typedef std::shared_ptr<Data> Ptr;
    typedef const std::shared_ptr<Data> ConstPtr;

    /**
     * @brief element_t : (label; data)
     */
    typedef std::pair<int,Eigen::VectorXd> element_t;
    typedef std::vector<element_t> data_t;
    typedef std::vector<std::vector<double>> estimation_t;

    /**
     * @brief default constructor
     */
    Data(){}

    /**
     * @brief Basic constructor. Initialize the dataset with the dimension of the samples and the number of class.
     * @param dim
     * @param nbr_class
     */
    Data(int dim, int nbr_class){
        _dimension = dim;
        _nbr_class = nbr_class;
    }

    /**
     * @brief copy constructor
     * @param d
     */
    Data(const Data& d) :
        _data(d._data),
        estimations(d.estimations),
        _dimension(d._dimension),
        _nbr_class(d._nbr_class)
    {}

    /**
     * @brief add an element in the data
     * @param a label
     * @param a sample
     */
    void add(int label,const Eigen::VectorXd& d){
        _data.push_back(std::make_pair(label,d));
    }


    /**
     * @brief add an element in the data
     * @param elt : a pair of a label and a sample
     */
    void add(const element_t& elt){
        _data.push_back(elt);
    }


    /**
     * @brief erase a sample from the dataset by index
     * @param index i
     */
    void erase(int i){
        _data.erase(_data.begin() + i);
        _data.shrink_to_fit();
    }

    /**
     * @brief erase all data
     */
    void clear(){
        _data.clear();
    }

    /**
     * @brief access to a data per index
     * @param i
     * @return constant reference to the element
     */
    const element_t& operator [](size_t i) const {return _data[i];}

    /**
     * @brief access to a data per index
     * @param i
     * @return refereence to the element
     */
    element_t& operator [](size_t i){return _data[i];}

    /**
     * @brief number of elements
     * @return number of elements
     */
    size_t size() const {return _data.size();}

    /**
     * @brief get all data
     * @return vector of paired label and data
     */
    const data_t& get() const {return _data;}

    /**
     * @brief get data of a specific class
     * @return vector of Eigen::VectorXd samples of a specific class
     */
    std::vector<Eigen::VectorXd> get_data(int class_lbl) const{
        std::vector<Eigen::VectorXd> res;
        for(const auto& data : _data)
            if(data.first == class_lbl)
                res.push_back(data.second);

        return res;
    }

    /**
     * @brief access to the last element added.
     * @return constant reference to this element
     */
    const element_t& last(){return _data.back();}

    estimation_t estimations;/**<estimations of the samples in the dataset. They must be computed with an external classifier*/

    /**
     * @brief share randomly the dataset between a training and test dataset according to a ratio
     * @param train dataset
     * @param test dataset
     * @param train test size ratio
     */
    void generate_train_test_dataset(Data& train, Data& test, float train_test_ratio);

    /**
     * @brief load a dataset from yaml file
     * @param filename
     * @param dimension
     * @param nbr_class
     * @return true if the dataset is successfully loaded
     */
    bool load_yml(const std::string& filename, int& dimension, int& nbr_class);

    /**
     * @brief export the dataset into a yaml
     * @param filename
     * @return true if the dataset is successfully exported
     */
    bool save_yml(const std::string& filename) const;

    /**
     * @brief export the dataset into a libsvm format
     * @param filename
     * @return true if the dataset is successfully exported
     */
    bool save_libsvm(const std::string& filename) const;

    /**
     * @brief export the dataset into a .data/.labels format
     * @param filename
     * @return true if the dataset is successfully exported
     */
    bool save_data_label(const std::string& filename) const;



protected:
    data_t _data; /**<the dataset*/
    int _dimension;/**<the dimension of the samples*/
    int _nbr_class;/**<the number of class*/
};

template <typename D>
inline std::ostream& operator<< (std::ostream& os,const Data& dataset ){

    typename Data::data_t data_set = dataset.get();
    for(auto data : data_set){
        os << "1 : ";
        os << "l : ";
        if(data.first)
            os << "1";
        else os << "0";

        os << " - ";
        os << data.second;
    }

    return os;
}
}//cmm

#endif //TRAINING_DATA_HPP
