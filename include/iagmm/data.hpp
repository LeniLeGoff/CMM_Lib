#ifndef TRAINING_DATA_HPP
#define TRAINING_DATA_HPP

#include <iostream>
#include <vector>
#include <memory>
#include <eigen3/Eigen/Core>
#include <yaml-cpp/yaml.h>

namespace iagmm{

class TrainingData{

public:

    typedef std::shared_ptr<TrainingData> Ptr;
    typedef const std::shared_ptr<TrainingData> ConstPtr;

    /**
     * @brief element_t : (label; data)
     */
    typedef std::pair<int,Eigen::VectorXd> element_t;
    typedef std::vector<element_t> data_t;
    typedef std::vector<double> estimation_t;

    /**
     * @brief default constructor
     */
    TrainingData(){}

    /**
     * @brief copy constructor
     * @param d
     */
    TrainingData(const TrainingData& d) :
        _data(d._data),
        estimations(d.estimations){}

    /**
     * @brief add an element in the data
     * @param positive data or negative data
     * @param data
     */
    void add(int label,const Eigen::VectorXd& d){
        _data.push_back(std::make_pair(label,d));
    }

    void add(int label,const Eigen::VectorXd& d,double quality){
        _data.push_back(std::make_pair(label,d));
        _qualities.push_back(quality);
    }

    void add(const element_t& elt){
        _data.push_back(elt);
    }

    void add(const element_t &elt, double quality){
        _data.push_back(elt);
        _qualities.push_back(quality);
    }

    void erase(int i){

        if(_qualities.size() == _data.size()){
            _qualities.erase(_qualities.begin() + i);
            _qualities.shrink_to_fit();
        }
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

    const element_t& last(){return _data.back();}

    estimation_t estimations;

    bool load_yml(const std::string& filename, int& dimension, int& nbr_class){
        std::cout << "load dataset : " << filename << std::endl;


        YAML::Node fileNode = YAML::LoadFile(filename);
        if (fileNode.IsNull()) {
            std::cerr << "File not found." << std::endl;
            return false;
        }

        YAML::Node features = fileNode["frame_0"]["features"];

        dimension = features["feature_0"]["value"].size();
        nbr_class = 0;

        for (unsigned int i = 0; i < features.size(); ++i) {
            std::stringstream stream;
            stream << "feature_" << i;
            YAML::Node tmp_node = features[stream.str()];

            Eigen::VectorXd feature(tmp_node["value"].size());
            for(size_t i = 0; i < tmp_node["value"].size(); ++i)
                feature(i) = tmp_node["value"][i].as<double>();

            add(tmp_node["label"].as<int>(),feature);
            if(tmp_node["label"].as<int>() > nbr_class)
                nbr_class = tmp_node["label"].as<int>();
        }
        nbr_class++;

        return true;
    }


    const estimation_t& get_qualities(){return _qualities;}

protected:
    data_t _data;
    estimation_t _qualities;
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
