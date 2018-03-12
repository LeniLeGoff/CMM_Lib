#include <iagmm/component.hpp>
#include <map>
#include <iostream>
#include <boost/math/distributions/fisher_f.hpp>
#define COEF 1.

using namespace iagmm;


void Component::update_parameters(){
    if(_samples.size() <= 4){
        _covariance = Eigen::MatrixXd::Identity(_dimension,_dimension)*COEF;
//        _factor = _sign;
        _mu = _samples[0];
        return;
    }
    _check_samples();


    Eigen::MatrixXd m_sum = Eigen::MatrixXd::Zero(_dimension,_dimension);
    Eigen::VectorXd v_sum = Eigen::VectorXd::Zero(_dimension);

    for(int i = 0 ; i < _samples.size(); i++){
        v_sum += _samples[i];
    }



    _mu = 1./_samples.size()*v_sum;

    for(int i = 0; i < _mu.rows(); i++)
        if(_mu(i) != _mu(i))
            _mu(i) = 0;



    for(const auto& sample : _samples){
        m_sum += (sample - _mu)*(sample - _mu).transpose();
    }

    if((m_sum-Eigen::MatrixXd::Zero(_dimension,_dimension)).squaredNorm() < 1e-4)
        _covariance = Eigen::MatrixXd::Identity(_dimension,_dimension)*COEF;
    else
        _covariance = 1./(_samples.size()-1)*m_sum*COEF;

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(2*PI*_covariance, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::VectorXd singularVal = svd.singularValues();
    double cm_determinant = 1.;
    for(int i = 0; i < singularVal.rows(); ++i){
        if(singularVal(i) > 1e-4)
            cm_determinant = cm_determinant*singularVal(i);
    }
    double exp_arg = -1./2.*((_mu - _mu).transpose()*covariance_pseudoinverse()).dot(_mu - _mu);
    _max = 1/cm_determinant*exp(exp_arg);
//    _factor = _sign*_samples.size()/(2*nbr_samples*nbr_Components)
//            _sign*get_standard_deviation()*_samples.size();
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

void Component::_incr_parameters(const Eigen::VectorXd& X){
    if(_samples.size() <= 1){
        _mu = X;
        _covariance = Eigen::MatrixXd::Identity(_dimension,_dimension)*COEF;
        return;
    }
    double f_size = _samples.size();
    _mu = (f_size-1)/f_size*_mu + 1/f_size*X;
    _covariance = (f_size-2)/(f_size-1)*_covariance
            + f_size/((f_size-1)*(f_size-1))*(X - _mu)*(X-_mu).transpose();

}

double Component::compute_multivariate_normal_dist(Eigen::VectorXd X) const {
    if((_covariance - Eigen::MatrixXd::Identity(_dimension,_dimension)*COEF).squaredNorm() == 0)
        return (X-_mu).squaredNorm() < 1e-4 ? _max : 0;

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(2*PI*_covariance, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::VectorXd singularVal = svd.singularValues();

    double cm_determinant = 1.;
    for(int i = 0; i < singularVal.rows(); ++i){
        if(singularVal(i) != singularVal(i))
            singularVal(i) = 0;
        if(singularVal(i) > 1e-4)
            cm_determinant = cm_determinant*singularVal(i);
    }

    double exp_arg = -1./2.*((X - _mu).transpose()*covariance_pseudoinverse()).dot(X - _mu);
    if(exp_arg > 0){
        std::cerr << "The covariance matrix is not positive definite" << std::endl;
        return 0;
    }
    double res = 1/cm_determinant*exp(exp_arg);
    if(res == res)
        return res;
    else return 0;
}


void Component::merge(const Component::Ptr c){
//    Component::Ptr new_c(new Component(*this));

    for(int i = 0; i < c->size(); i++)
        add(c->get_sample(i));

    update_parameters();

    return ;
}


Component::Ptr Component::split(){
//    std::cout << "split" << std::endl;
    _check_samples();

    //* compute distance matrix between the samples of the components
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
    //*/

    //* create a graph of minimal distances of the samples
    Eigen::VectorXd minIndexes(_samples.size());
    std::multimap<int,int> graph;
    int r,c;
    for(int i = 0; i < _samples.size(); i++){
       m_dist.row(i).minCoeff(&r,&c); //search the closest sample from the iest sample
       if([&]()-> bool{auto range = graph.equal_range(i);
               for(auto& it = range.first; it != range.second; it++)
               if(c == it->second) return false; return true;}()){ //search if this edge already exist  in the graph
           graph.emplace(i,c);
           graph.emplace(c,i);
       }
       minIndexes(i) = c;
    }
    //*/

    //* Go through each connected nodes
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
        rec(graph.begin()->first);    Eigen::VectorXd diff_mu, ellipse_vect1,ellipse_vect2;


        for(const auto& i : tmp_ind)
            graph.erase(i);

        indexes.push_back(tmp_ind);
    }
    //*/

    if(indexes.size() == 1)//if there only one list of indexes split is aborted
        return NULL;

    //*reduce the indexes into list
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
    //*/

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

bool Component::intersect(const Component::Ptr comp) const {
    Eigen::VectorXd diff_mu, ellipse_vect1, ellipse_vect2;
    double n1 = _samples.size(), n2 = comp->get_samples().size(), p = _dimension;
    if(n1 <= p /*|| n2 <= p*/)
        return false;
    boost::math::fisher_f_distribution<double> F1(p,n1 - p);
//    boost::math::fisher_f_distribution<double> F2(p,n2-p);
    double q1 = boost::math::quantile(F1,0.75);
//    double q2 = boost::math::quantile(F2,0.95);
    double factor1 = (n1-1)*p*(n1+1)/((n1 - p)*n1)*q1;
//    double factor2 = (n2-1)*p*(n2+1)/((n2-p)*n2)*q2;
//    std::cout << factor1 << " " << factor2 << std::endl;
//    double dist_mu;
    //* compute the vectors for intersection criterion
    diff_mu = (comp->get_mu()-_mu);
//    dist_mu = diff_mu.norm();
//    ellipse_vect1 = 1/factor1*(comp->covariance_pseudoinverse()*diff_mu/diff_mu.norm());
//    ellipse_vect1 = factor1*(_covariance*diff_mu/diff_mu.norm());
    //*/
    double val = distance(comp->get_mu());
    if(val != val){
        val = 0;
    }

    std::cout << val << " <=? " << factor1 << std::endl;

    return val <= factor1;
}

double Component::distance(const Eigen::VectorXd& X) const {
    return ((X - _mu).transpose()*covariance_pseudoinverse()).dot(X - _mu);
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

Eigen::MatrixXd Component::covariance_pseudoinverse() const{
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(_covariance, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::VectorXd singularVal = svd.singularValues();
    Eigen::MatrixXd singularValInv = Eigen::MatrixXd::Zero(_dimension,_dimension);
    for(int i = 0; i < _dimension; i++){
        if(singularVal(i) > 1e-4)
            singularValInv(i,i) = 1./singularVal(i);
        if(singularValInv(i,i) != singularValInv(i,i))
            singularValInv(i,i) = 0;
    }
    Eigen::MatrixXd V = svd.matrixV();
    Eigen::MatrixXd U = svd.matrixU();
    for(int i = 0; i < V.rows(); i++){
        for(int j = 0; j < V.cols(); j++){
            if(V(i,j) != V(i,j))
                V(i,j) = 0;
            if(U(j,i) != U(j,i))
                U(j,i) = 0;
        }
    }


    return V*singularValInv*U.transpose();
}

void Component::_check_samples(){
    for(int i = 0; i < _samples.size(); i++){
        if(_samples[i].rows() == 0){
            _samples.erase(_samples.begin() + i);
            i--;
            continue;
        }
        for(int j = 0; j < _samples[i].rows(); j++){
            if(_samples[i](j) != _samples[i](j))
                _samples[i](j) = 0;
        }
    }
    _samples.shrink_to_fit();
}

std::string Component::print_parameters() const {
    std::stringstream stream;
    stream << "----------------------" << std::endl;
    stream << "lbl : " << _label << std::endl;
    stream << "pseudoinverse covariance : \n" << covariance_pseudoinverse() << std::endl;
    stream << "mu : \n" << _mu << std::endl;
    stream << "factor : " << _factor << std::endl;
    stream << "size : " << _samples.size() << std::endl;
//    stream << "score : " << component_score() << std::endl;
    stream << "standard deviation : " << get_standard_deviation() << std::endl;
//    stream << "maximum value : " << _max << std::endl;
    stream << "----------------------" << std::endl;
    return stream.str();
}
// ----------------------
