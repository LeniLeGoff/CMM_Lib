#ifndef TRAINING_DATA_HPP
#define TRAINING_DATA_HPP

#include <iostream>
#include <vector>
#include <memory>
#include <eigen3/Eigen/Core>

namespace iagmm{

class TrainingData{

public:

    typedef std::shared_ptr<TrainingData> Ptr;
    typedef const std::shared_ptr<TrainingData> ConstPtr;

    typedef std::pair<int,Eigen::VectorXd> element_t;
    typedef std::vector<element_t> data_t;

    /**
     * @brief default constructor
     */
    TrainingData(){}

    /**
     * @brief copy constructor
     * @param d
     */
    TrainingData(const TrainingData& d) : _data(d._data){}

    /**
     * @brief add an element in the data
     * @param positive data or negative data
     * @param data
     */
    void add(int label,const Eigen::VectorXd& d){
        _data.push_back(std::make_pair(label,d));
    }

    void add(const element_t& elt){
        _data.push_back(elt);
    }

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
     * @brief operator []
     * @param i
     * @return
     */
    const element_t& operator [](size_t i) const {return _data[i];}

    /**
     * @brief operator []
     * @param i
     * @return
     */
    element_t& operator [](size_t i){return _data[i];}

    /**
     * @brief number of elements
     * @return
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

protected:
    data_t _data;

};

template <typename Data>
inline std::ostream& operator<< (std::ostream& os,const TrainingData& dataset ){

    typename TrainingData::data_t data_set = dataset.get();
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
}//iagmm

#endif //TRAINING_DATA_HPP
