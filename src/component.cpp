#include <iagmm/component.hpp>
#include <map>
#include <iostream>

using namespace iagmm;

void Component::update_parameters(){
    if(_samples.size() <= 3){
        _covariance = Eigen::MatrixXd::Identity(_dimension,_dimension);
//        _factor = _sign;
        _mu = _samples[0];
        return;
    }

    Eigen::MatrixXd m_sum = Eigen::MatrixXd::Zero(_dimension,_dimension);
    Eigen::VectorXd v_sum = Eigen::VectorXd::Zero(_dimension);

    for(const auto& sample : _samples){
        v_sum += sample;
    }

    _mu = 1./_samples.size()*v_sum;

    for(const auto& sample : _samples)
        m_sum += (sample - _mu)*(sample - _mu).transpose();

    _covariance = 1./(_samples.size()-1)*m_sum;

//    _factor = _sign*_samples.size()/(2*nbr_samples*nbr_Components)
            //_sign*get_standard_deviation()*_samples.size();
//            _sign*_samples.size()/nbr_samples;


}

double Component::compute_multivariate_normal_dist(Eigen::VectorXd X) const {
    double cm_determinant = (2*PI*_covariance).determinant();
    double exp_arg = -1./2.*((X - _mu).transpose()*_covariance.inverse()).dot(X - _mu);

    return 1/cm_determinant*exp(exp_arg);
}


Component::Ptr Component::merge(const Component::Ptr c){
    Component::Ptr new_c(new Component(*this));

    for(int i = 0; i < c->size(); i++)
        new_c->add(c->get_sample(i));

    new_c->update_parameters();

    return new_c;
}


Component::Ptr Component::split(){
//    std::cout << "split" << std::endl;

    //compute distance matrix.
    Eigen::MatrixXd m_dist(_samples.size(),_samples.size());
    for(int i = 0; i < _samples.size(); i++){
        for(int j = 0; j < _samples.size(); j++){
            if(i==j){
                m_dist(i,j) = 1000.;
                continue;
            }
            m_dist(i,j) = (_samples[i] - _samples[j]).squaredNorm();
        }
    }

    Eigen::VectorXd minIndexes(_samples.size());

    std::multimap<int,int> graph;
    int r,c;
    for(int i = 0; i < _samples.size(); i++){
       m_dist.row(i).minCoeff(&r,&c);
       graph.emplace(i,c);
       graph.emplace(c,i);
       minIndexes(i) = c;
    }


    std::vector<int> indexes;

    std::function<void(int)> rec = [&](int current_i){
        for(auto i : indexes)
            if(i == current_i)
                return;
        indexes.push_back(current_i);
        auto range = graph.equal_range(current_i);
        for(auto it = range.first; it != range.second; it++)
            rec(it->second);
    };
    rec(0);

//    std::cout << indexes.size() << std::endl;
//    for(auto i : indexes)
//        std::cout << i << std::endl;

    if(indexes.size() == _samples.size())
        return NULL;

    Component::Ptr new_c(new Component(_dimension,_label));
//    std::cout << "samples size : " << _samples.size() << std::endl;
    for(int i : indexes){
//        std::cout << i << " : " << _samples[i] << std::endl;
        new_c->add(_samples[i]);
    }

    std::vector<Eigen::VectorXd> cpy_samples = _samples;
    _samples.clear();
    for(int i = 0; i < cpy_samples.size(); i++){
        if([=](int i) -> bool {for(int ind : indexes) {if(i == ind ) return false;} return true;}(i))
            _samples.push_back(cpy_samples[i]);
    }

    update_parameters();
    new_c->update_parameters();

    return new_c;
}

double Component::get_standard_deviation() const{
    if(_samples.size() <= 1) return 0.;
    return sqrt(_covariance.diagonal().dot(_covariance.diagonal()));
}

std::vector<double> Component::get_intern_estimation() const {
    std::vector<double> res;
    for(auto s : _samples)
        res.push_back(compute_multivariate_normal_dist(s));
    return res;
}

double Component::component_score() const {
    double sum = 0;
    for(auto s : _samples)
        sum += (compute_multivariate_normal_dist(s)/compute_multivariate_normal_dist(_mu)-1.)*
                (compute_multivariate_normal_dist(s)/compute_multivariate_normal_dist(_mu)-1.);
    return sum/(double)_samples.size();
}

double Component::entropy(){
    return _factor*(-std::log(_factor) + 1./2.*std::log((2.*PI*std::exp(1.)*_covariance).determinant()));
}

void Component::compute_eigenvalues(Eigen::VectorXd& eigenvalues, Eigen::MatrixXd& eigenvectors) const {
    Eigen::EigenSolver<Eigen::MatrixXd> solver(_covariance);

    eigenvalues = Eigen::VectorXd(_dimension);
    for(int i = 0; i < _dimension; i++)
        eigenvalues(i) = solver.eigenvalues()(i).real();
    eigenvectors = Eigen::MatrixXd(_dimension,_dimension);
    for(int i = 0; i < _dimension; i++){
        for(int j = 0; j < _dimension; j++)
            eigenvectors(i,j) = solver.eigenvectors()(i,j).real();
    }
}


void Component::print_parameters() const {
    std::cout << "----------------------" << std::endl;
    std::cout << "lbl : " << _label << std::endl;
    std::cout << "covariance : " << _covariance << std::endl;
    std::cout << "mu : " << _mu << std::endl;
    std::cout << "factor : " << _factor << std::endl;
    std::cout << "size : " << _samples.size() << std::endl;
    std::cout << "score : " << component_score() << std::endl;
    std::cout << "standard deviation : " << get_standard_deviation() << std::endl;
    std::cout << "----------------------" << std::endl;
}
// ----------------------
