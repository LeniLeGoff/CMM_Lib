#include <iagmm/component.hpp>
#include <map>
#include <iostream>

using namespace iagmm;


void Component::update_parameters(){
    if(_samples.size() <= 2){
        _covariance = Eigen::MatrixXd::Identity(_dimension,_dimension)*1e-10;
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

//void Component::update_parameters(){
//    if(_size <= 1){
//        _mu = _samples.back();
//        _covariance = Eigen::MatrixXd::Identity(_dimension,_dimension);
//        return;
//    }
//    double f_size = _size;
//    _mu = (f_size-1)/f_size*_mu + 1/f_size*_samples.back();
//    _covariance = (f_size-2)/(f_size-1)*_covariance
//            + (f_size)/((f_size-1)*(f_size-1))*
//            (_samples.back()-_mu)*(_samples.back()-_mu).transpose();
//}

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
       if([&]()-> bool{auto range = graph.equal_range(i);
               for(auto& it = range.first; it != range.second; it++)
               if(c == it->second) return false; return true;}()){
           graph.emplace(i,c);
           graph.emplace(c,i);
       }
       minIndexes(i) = c;
    }


    std::vector<std::vector<int>> indexes;
    while(!graph.empty())
    {
        std::vector<int> tmp_ind;
        std::function<void(int)> rec = [&](int current_i){
            for(auto i : tmp_ind)
                if(i == current_i)
                    return;
            tmp_ind.push_back(current_i);
            auto range = graph.equal_range(current_i);
            for(auto it = range.first; it != range.second; it++)
                rec(it->second);
        };
        rec(graph.begin()->first);

        for(const auto& i : tmp_ind)
            graph.erase(i);

        indexes.push_back(tmp_ind);
    }

//    std::cout << indexes.size() << std::endl;
//    for(auto i : indexes)
//        std::cout << i << std::endl;

    if(indexes.size() == 1)
        return NULL;

    Eigen::MatrixXd means = Eigen::MatrixXd::Zero(_dimension,indexes.size());
    Eigen::MatrixXd dist(indexes.size(),indexes.size());

    while(indexes.size() > 2)
    {
        means = Eigen::MatrixXd::Zero(_dimension,indexes.size());
        for(size_t k = 0; k < indexes.size(); k++){
            for(const auto& i : indexes[k]){
                means.col(k) += _samples[i];
            }
            means.col(k) = means.col(k)/(double)indexes[k].size();
        }
        dist.resize(indexes.size(),indexes.size());
        for(size_t k = 0; k < indexes.size(); k++){
            for(size_t j = 0; j < indexes.size(); j++){
                if(k == j){ dist(k,j) = 100000000000; continue;}
                dist(k,j) = (means.col(k) - means.col(j)).squaredNorm();
            }
        }

        dist.minCoeff(&r,&c);
        for(const auto& i : indexes[r]){
            indexes[c].push_back(i);
        }
        indexes.erase(indexes.begin() + r);
        indexes.shrink_to_fit();
    }

    Component::Ptr new_c(new Component(_dimension,_label));
    //    std::cout << "samples size : " << _samples.size() << std::endl;
    for(int i : indexes[0]){
        //        std::cout << i << " : " << _samples[i] << std::endl;
        new_c->add(_samples[i]);
    }

    std::vector<Eigen::VectorXd> cpy_samples = _samples;
    _samples.clear();
    for(int i = 0; i < cpy_samples.size(); i++){
        if([=](int i) -> bool {for(int ind : indexes[1]) {if(i == ind ) return true;} return false;}(i))
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
        sum += fabs(compute_multivariate_normal_dist(s)/compute_multivariate_normal_dist(_mu)-1.);
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
