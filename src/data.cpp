#include <iagmm/data.hpp>

using namespace iagmm;

void TrainingData::generate_train_test_dataset(TrainingData& train, TrainingData& test,
                                                float train_test_ratio){



}

bool TrainingData::load_yml(const std::string& filename, int& dimension, int& nbr_class){
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

    _dimension = dimension;
    _nbr_class = nbr_class;

    return true;
}

bool TrainingData::save_yml(const std::string& filename) const{

    std::ofstream ofs(filename,std::ofstream::out);
    if(!ofs.is_open())
        return false;

    std::string frame_id, feat_id;
    YAML::Emitter emitter;

    frame_id = "frame_0";

    emitter << YAML::BeginMap //BEGIN MAP_0
                << YAML::Key << frame_id << YAML::Value
                << YAML::BeginMap //BEGIN MAP_1
                    << YAML::Key << "features" << YAML::Value
                    << YAML::BeginMap; //BEGIN MAP_3

    for (unsigned int i = 0; i < _data.size(); ++i) {
        feat_id = "feature_" + std::to_string(i);

        emitter << YAML::Key << feat_id << YAML::Value
                << YAML::BeginMap //BEGIN MAP_4
                    << YAML::Key << "label" << YAML::Value << _data[i].first
                    << YAML::Key << "value" << YAML::Value
                    << YAML::BeginSeq;
        for (unsigned int j = 0; j < _data[i].second.size(); ++j) {
            emitter  << _data[i].second[j];
        }
        emitter << YAML::EndSeq
                << YAML::EndMap; //END MAP_4
    }
    emitter << YAML::EndMap //END MAP_3
            << YAML::EndMap //END MAP_1
            << YAML::EndMap //END MAP_0
            << YAML::Newline;

    ofs << emitter.c_str();
    ofs.close();
    return true;
}

bool TrainingData::save_libsvm(const std::string& filename) const{
    std::ofstream ofs(filename,std::ofstream::out);
    if(!ofs.is_open())
        return false;

    ofs << size() << " " << _dimension << " " << _nbr_class << std::endl;
    for(const element_t &elt : _data){
        ofs << elt.first << " ";
        for(int i = 0; i < elt.second.rows(); i++){
            ofs << i << ":" << elt.second(i) << " ";
        }
        ofs << std::endl;
    }
    ofs.close();

}

bool TrainingData::save_data_label(const std::string& filename) const{
    std::ofstream ofs(filename + ".data",std::ofstream::out);
    if(!ofs.is_open())
        return false;

    std::ofstream ofs2(filename + ".labels",std::ofstream::out);
    if(!ofs2.is_open())
        return false;

    ofs << size() << " " << _dimension << std::endl;
    ofs2 << size() << " " << _nbr_class << std::endl;
    for(const element_t &elt : _data){

        for(int i = 0; i < elt.second.rows(); i++){
           ofs << elt.second(i) << " ";

        }
        ofs2 << elt.first << std::endl;
        ofs << std::endl;
    }
    ofs.close();
    ofs2.close();
}
