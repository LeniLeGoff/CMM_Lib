
#include "iagmm/gmm.hpp"
#include <map>
#include <boost/chrono.hpp>

using namespace iagmm;

//ESTIMATOR
void GMM::_estimator::operator ()(const tbb::blocked_range<size_t>& r){
    double val;
    double sum = _sum_map[_current_lbl];

    Eigen::VectorXd X = _X;

    for(size_t i=r.begin(); i != r.end(); ++i){
        val = _model->model()[_current_lbl][i]->get_factor()*
                _model->model()[_current_lbl][i]->compute_multivariate_normal_dist(X);
        sum += val;

    }
    _sum_map[_current_lbl] = sum;
}

double GMM::_estimator::estimation(int lbl){

    for(_current_lbl = 0; _current_lbl < _model->get_nbr_class(); _current_lbl++)
        tbb::parallel_reduce(tbb::blocked_range<size_t>(0,_model->model()[_current_lbl].size()),*this);


    double sum_of_sums = 0;
    for(const auto& sum : _sum_map)
        sum_of_sums += sum.second;
    return (1 + _sum_map[lbl])/(2 + sum_of_sums);
}
//--ESTIMATOR

double GMM::compute_estimation(const Eigen::VectorXd& X, int lbl){

    if([&]() -> bool { for(int i = 0; i < _nbr_class; i++){if(!_model[i].empty()) return false;} return true;}())
        return 0.5;

    _estimator estimator(this, X);

    return estimator.estimation(lbl);
}

void GMM::compute_normalisation(){
    double sum_of_sums = 0;
    double val;
    for(const auto& model : _model){
        for(const auto& component : model.second){
            val = component->get_factor()*
                    component->compute_multivariate_normal_dist(component->get_mu());
            sum_of_sums += val;
        }
    }
    _normalisation = sum_of_sums;
}

//SCORE_CALCULATOR
void GMM::_score_calculator::operator()(const tbb::blocked_range<size_t>& r){
    double sum = _sum;

    for(size_t i = r.begin(); i != r.end(); ++i){
        sum += fabs(_model->compute_estimation(_samples[i].second,_samples[i].first) - 1.);
    }
    _sum = sum;
}

double GMM::_score_calculator::compute(){
//    _model->compute_normalisation();
    tbb::parallel_reduce(tbb::blocked_range<size_t>(0,_samples.size()),*this);
    return _sum/((double)_samples.size());
}
//--SCORE_CALCULATOR


//DISTRIBUTION_CONSTRUCTOR
//void GMM::_distribution_constructor::operator()(const tbb::blocked_range<size_t>& r){
//    for()
//}

//--DISTRIBUTION_CONSTRUCTOR

void GMM::update_factors(){

    double sum_size = 0;
    for(auto& components : _model)
        sum_size += components.second.size();

    for(auto& components : _model){
        for(auto& c: components.second)
            c->set_factor((double)c->size()/(/*sum_size*/(double)_samples.size()));
    }
}



double GMM::unit_factor(){

    double sum_size = 0;
    for(auto& components : _model)
        sum_size += components.second.size();

    return 1./((double)_samples.size());
}


void GMM::_new_component(const Eigen::VectorXd& sample, int label){
    Component::Ptr component(new Component(_dimension,label));
    component->add(sample);
    component->update_parameters();
    _model[label].push_back(component);
    update_factors();
}

std::vector<double> GMM::model_scores(){
    double score = 0;
    std::vector<double> scores;

    for(const auto& components: _model){
        for(const auto& comp: components.second){
            score = 0;
            for(const auto& s: comp->get_samples()){
                score += fabs(compute_estimation(s,components.first) - 1);
            }
            scores.push_back(score/(double)comp->size());
        }
    }
    return scores;
}

void GMM::knn(const Eigen::VectorXd& center, TrainingData& output, int k){
    double min_dist, dist;
    int min_index;
    TrainingData cpy_samples(_samples);
    for(int j = 0; j < k; j++){
        min_dist = sqrt((cpy_samples[0].second - center).transpose()*(cpy_samples[0].second - center));
        min_index = 0;
        for(int i = 1; i < cpy_samples.size(); i++){
            dist =  sqrt((cpy_samples[i].second - center).transpose()*(cpy_samples[i].second - center));
            if(dist < min_dist){
                min_index = i;
                min_dist = dist;
            }
        }
        output.add(cpy_samples[min_index]);
        cpy_samples.erase(min_index);
    }
}

Eigen::VectorXd  GMM::mean_shift(const Eigen::VectorXd& X, int lbl){
    double estimation = 0;
    Eigen::VectorXd numerator = Eigen::VectorXd::Zero(_dimension);
//    for(const auto& comps : _model){
        for(int i = 0; i < _model[lbl].size(); i++){
            numerator += _model[lbl][i]->get_factor()*_model[lbl][i]->compute_multivariate_normal_dist(X)*_model[lbl][i]->get_mu();
            estimation += _model[lbl][i]->get_factor()*_model[lbl][i]->compute_multivariate_normal_dist(X);
        }
//    }

    return numerator/estimation - X;
}

void GMM::_merge(int ind, int lbl){
    std::cout << "merge function" << std::endl;
    boost::chrono::system_clock::time_point timer;
    timer  = boost::chrono::system_clock::now();

    GMM candidate;
    double score, score2, candidate_score;

    _score_calculator sc(this,_samples);
    score = sc.compute();

    Eigen::VectorXd eigenval, eigenval2, diff_mu, ellipse_vect1, ellipse_vect2;
    Eigen::MatrixXd eigenvect, eigenvect2;
    _model[lbl][ind]->compute_eigenvalues(eigenval,eigenvect);
    for(int i = 0; i < _model[lbl].size(); i++){
        if(i == ind)
            continue;
        _model[lbl][i]->compute_eigenvalues(eigenval2,eigenvect2);
        diff_mu = _model[lbl][i]->get_mu() - _model[lbl][ind]->get_mu();
        ellipse_vect1 = (_model[lbl][i]->covariance_pseudoinverse().transpose()*diff_mu/diff_mu.squaredNorm());
        ellipse_vect2 = (_model[lbl][ind]->covariance_pseudoinverse().transpose()*diff_mu/diff_mu.squaredNorm());
//        for(int i = 0; i < eigenval.rows(); i++){
//            ellipse_vect1(i) = ellipse_vect1(i)*eigenval(i);
//            ellipse_vect2(i) = ellipse_vect2(i)*eigenval2(i);
//        }

        if(diff_mu.squaredNorm() < (ellipse_vect1.squaredNorm() + ellipse_vect2.squaredNorm())){
//            score = _component_score(ind,lbl);
//            score2 = _component_score(i,lbl);

            candidate = GMM(_model);
            candidate.set_samples(_samples);
            candidate.model()[lbl][ind] =
                    candidate.model()[lbl][ind]->merge(candidate.model()[lbl][i]);
//            TrainingData knn_output;
//            candidate.knn(candidate.model()[lbl][ind]->get_mu(),knn_output,candidate.model()[lbl][ind]->size());

            candidate.model()[lbl].erase(candidate.model()[lbl].begin() + i);
            candidate.update_factors();

            _score_calculator candidate_sc(&candidate,candidate.get_samples());
            candidate_score = candidate_sc.compute();


            if(candidate_score <= score ){
                std::cout << "-_- MERGE _-_" << std::endl;
                _model[lbl][ind] = _model[lbl][ind]->merge(_model[lbl][i]);
                _model[lbl].erase(_model[lbl].begin() + i);
                update_factors();
                std::cout << "Merge finish, time spent : "
                          << boost::chrono::duration_cast<boost::chrono::milliseconds>(
                                 boost::chrono::system_clock::now() - timer) << std::endl;
                return;
            }
        }
    }
    std::cout << "Merge finish, time spent : "
              << boost::chrono::duration_cast<boost::chrono::milliseconds>(
                     boost::chrono::system_clock::now() - timer) << std::endl;

}

std::pair<double,double> GMM::_coeff_intersection(int ind1, int lbl1, int ind2, int lbl2){
    std::pair<double,double> coeffs;
    Eigen::VectorXd eigenval, eigenval2, diff_mu;
    Eigen::MatrixXd eigenvect, eigenvect2;
    _model[lbl1][ind1]->compute_eigenvalues(eigenval,eigenvect);
    _model[lbl2][ind2]->compute_eigenvalues(eigenval2,eigenvect2);
    diff_mu = _model[lbl1][ind1]->get_mu() - _model[lbl1][ind1]->get_mu();

    coeffs.first = diff_mu.dot(eigenvect.col(0)) - diff_mu.squaredNorm()*diff_mu.squaredNorm();
    coeffs.second = diff_mu.dot(eigenvect2.col(0)) - diff_mu.squaredNorm()*diff_mu.squaredNorm();

    return coeffs;
}

double GMM::_component_score(int i, int lbl){
    TrainingData knn_output;
    knn(_model[lbl][i]->get_mu(), knn_output,_model[lbl][i]->size());
    _score_calculator sc(this,knn_output);
    return sc.compute();
}


void GMM::_split(int ind, int lbl){
    std::cout << "split function" << std::endl;
    boost::chrono::system_clock::time_point timer;
    timer  = boost::chrono::system_clock::now();

    if(_model[lbl][ind]->size() < 4)
       return;
    GMM candidate;

    Eigen::VectorXd eigenval, eigenval2, diff_mu, ellipse_vect1,ellipse_vect2;
    Eigen::MatrixXd eigenvect, eigenvect2;
    _model[lbl][ind]->compute_eigenvalues(eigenval,eigenvect);
    double cand_score1, cand_score2, score;
    for(int l = 0; l < _nbr_class; l++){
        if(l == lbl)
            continue;
        for(const auto& comp :  _model[l]){
            comp->compute_eigenvalues(eigenval2,eigenvect2);

            diff_mu = (comp->get_mu()-_model[lbl][ind]->get_mu());
            ellipse_vect1 = (comp->covariance_pseudoinverse().transpose()*diff_mu/diff_mu.squaredNorm());
            ellipse_vect2 = (_model[lbl][ind]->covariance_pseudoinverse().transpose()*diff_mu/diff_mu.squaredNorm());
//            for(int i = 0; i < eigenval.rows(); i++){
//                ellipse_vect1(i) = ellipse_vect1(i)*eigenval(i);
//                ellipse_vect2(i) = ellipse_vect2(i)*eigenval2(i);
//            }

            if(diff_mu.squaredNorm() < ellipse_vect1.squaredNorm() + ellipse_vect2.squaredNorm()){
                candidate = GMM(_model);
                candidate.set_samples(_samples);
                Component::Ptr new_component = candidate.model()[lbl][ind]->split();

                if(new_component){
                    candidate.model()[lbl].push_back(new_component);
                    candidate.update_factors();
                    cand_score1 = candidate._component_score(ind,lbl);
                    cand_score2 = candidate._component_score(candidate.model()[lbl].size()-1,lbl);
                    score = _component_score(ind,lbl);
                    if((cand_score1+cand_score2)/2. < score){
                        std::cout << "-_- SPLIT _-_" << std::endl;
                        _model = candidate.model();
                        update_factors();

                        return;
                    }
                }
            }
        }
    }
    std::cout << "Split finish, time spent : "
              << boost::chrono::duration_cast<boost::chrono::milliseconds>(
                     boost::chrono::system_clock::now() - timer) << std::endl;
}

int GMM::next_sample(const std::vector<std::pair<Eigen::VectorXd,double>> &samples, Eigen::VectorXd &choice_dist_map){
    choice_dist_map = Eigen::VectorXd::Zero(samples.size());

    if(_samples.size() == 0)
        return rand()%samples.size();

//    TrainingData::element_t last_sample = _samples.last();
    double total = 0,cumul = 0;
    double max_dist = 0;

    std::map<double,int> choice_distibution;
    boost::random::uniform_real_distribution<> distrib(0,1);

//    tbb::parallel_for(tbb::blocked_range<size_t>(0,samples.size()),
//                      [&](const tbb::blocked_range<size_t>& r){
//        double dist;
//        for(int i = r.begin(); i != r.end(); ++i){

//           dist = 0;
//           for(const auto& s : _samples.get()){
//               dist += _distance(s.second,samples[i].first);
//           }
//           if(dist > max_dist)
//               max_dist = dist;
//           choice_dist_map(i) = dist;
//        }
//    });
//    if(max_dist > 0)
//        choice_dist_map = choice_dist_map/max_dist;
    for(int i = 0; i < choice_dist_map.rows(); ++i){
        double est = samples[i].second;
        if(est > .5)
            est = (1. - est) * 2.;
        else
            est = est*2.;
//        choice_dist_map(i) = 1/(1. + exp(-40.*((choice_dist_map(i) + samples.size()*est)/(1+samples.size()) - 0.5)));
//        choice_dist_map(i) = (choice_dist_map(i) + samples.size()*est)/(1+samples.size());
        choice_dist_map(i) = est;
        total += choice_dist_map(i);
    }
    for(int i = 0; i < choice_dist_map.rows(); ++i){
        cumul += choice_dist_map(i);
        choice_distibution.emplace(cumul/total,i);
    }
    return choice_distibution.lower_bound(distrib(_gen))->second;
}

//int GMM::next_sample(const samples_t& samples, Eigen::VectorXd& choice_dist_map){
//    std::cout << "next_sample function" << std::endl;
//    boost::chrono::system_clock::time_point timer;
//    timer  = boost::chrono::system_clock::now();

//    choice_dist_map = Eigen::VectorXd::Zero(samples.size());
//    std::map<double,int> choice_distibution;
//    double total = 0, cumul = 0, max_val = 0;
//    boost::random::uniform_real_distribution<> distrib(0,1);

//    if([&]() -> bool {for(auto& comp : _model) if(comp.second.empty()) return true; return false;}())
//        return rand()%(samples.size());


//    tbb::parallel_for(tbb::blocked_range<size_t>(0,samples.size()),
//            [&](const tbb::blocked_range<size_t>& r){
//        double dist, min_dist, comp_radius;
//        int min_ind, min_lbl;
//        Eigen::VectorXd eigenval, diff;
//        Eigen::MatrixXd eigenvect;
//        for(size_t j = r.begin(); j != r.end(); ++j){

//            min_dist = _model[0][0]->distance(samples[j]);
//            min_ind = 0;
//            min_lbl = 0;

//            for(const auto& comps : _model){
//                for(int i = 0; i != comps.second.size(); ++i){
//                    //                std::cout << comps.second[i]->get_covariance() << std::endl;
//                    dist = comps.second[i]->distance(samples[j]);
//                    if(dist < min_dist){
//                        min_dist = dist;
//                        min_ind = i;
//                        min_lbl = comps.first;
//                    }
//                }
//            }


//            //            _model[min_lbl][min_ind]->compute_eigenvalues(eigenval,eigenvect);
//            diff = _model[min_lbl][min_ind]->get_mu() - samples[j];
//            if(diff.squaredNorm() == 0 || min_dist != min_dist)
//                choice_dist_map(j) = 0;
//            else{
//                comp_radius = ((_model[min_lbl][min_ind]->covariance_pseudoinverse().transpose()*(diff/diff.squaredNorm()))).squaredNorm();
//                if(min_dist <= comp_radius){
//                    choice_dist_map(j) = _model[min_lbl][min_ind]->get_factor()*(1. - min_dist/comp_radius);
//                    if(choice_dist_map(j) > max_val)
//                        max_val = choice_dist_map(j);
//                }
//                else choice_dist_map(j) = 0;
//            }
//        }
//    });
//    int r,c;
//    if(max_val > 0)
//        choice_dist_map = choice_dist_map/max_val;

//    choice_dist_map = Eigen::VectorXd::Constant(samples.size(),1.) - choice_dist_map;

//    for(int i = 0; i < choice_dist_map.rows(); ++i){
//        total += choice_dist_map(i);
//    }
//    for(int i = 0; i < choice_dist_map.rows(); ++i){
//        cumul += choice_dist_map(i);
//        choice_distibution.emplace(cumul/total,i);
//    }
//    std::cout << "next_sample finish, time spent : "
//              << boost::chrono::duration_cast<boost::chrono::milliseconds>(
//                     boost::chrono::system_clock::now() - timer) << std::endl;
//    return choice_distibution.lower_bound(distrib(_gen))->second;
//}

void GMM::append(const std::vector<Eigen::VectorXd> &samples, const std::vector<int>& lbl){
    int r,c; //row and column indexes
    for(int i = 0 ; i < samples.size(); i++){
        if(!append(samples[i],lbl[i]))
            continue;
    }
}


int GMM::append(const Eigen::VectorXd &sample,const int& lbl){
    int r,c; //row and column indexes
    add(sample,lbl);

    if(_model[lbl].empty()){
        _new_component(sample,lbl);
        return 0;
    }

    Eigen::VectorXd distances(_model[lbl].size());

    for (int j = 0; j < _model[lbl].size(); j++) {
        distances(j) = (sample-_model[lbl][j]->get_mu()).squaredNorm();
    }
    distances.minCoeff(&r,&c);
    _model[lbl][r]->add(sample);
    _model[lbl][r]->update_parameters();

    return r;
}

void GMM::update_model(int ind, int lbl){

    int n,rand_ind;
    n = _model[lbl].size();
    _split(ind,lbl);
    if(n > 1)
        _merge(ind,lbl);

    for(int i = 0; i < _nbr_class; i++){
        n = _model[i].size();
        if(n < 2) break;
        do
            rand_ind = rand()%n;
        while(rand_ind == ind);
        _split(rand_ind,i);

        do
            rand_ind = rand()%n;
        while(rand_ind == ind);
        _merge(rand_ind,i);
    }

    for(auto& components : _model)
        for(auto& comp : components.second)
            comp->update_parameters();

}

void GMM::fit(const Eigen::VectorXd& sample, const int& lbl){
    int ind = append(sample, lbl);
    update_model(ind, lbl);
}

std::vector<int> GMM::find_closest_components(double& min_dist, int lbl){

    std::vector<int> indexes(2);
    indexes[0] = 0;
    indexes[1] = 1;


    min_dist = (_model[lbl][0]->get_mu()-_model[lbl][1]->get_mu()).squaredNorm();

    double dist;
    for(int i = 1; i < _model[lbl].size(); i++){
        for(int j = i+1; j < _model[lbl].size(); j++){
            dist = (_model[lbl][i]->get_mu()-_model[lbl][j]->get_mu()).squaredNorm();
            if(dist < min_dist){
                indexes[0] = i;
                indexes[1] = j;
            }
        }
    }

    return indexes;
}

int GMM::find_closest(int i, double &min_dist, int lbl){

    Eigen::VectorXd distances(_model[lbl].size()-1);
    int k = 0;
    for(int j = 0; j < _model[lbl].size(); j++){
        if(j == i)
            continue;

        distances(k) =  (_model[lbl][i]->get_mu() - _model[lbl][j]->get_mu()).squaredNorm();
        k++;
    }
    int r, c;
    min_dist = distances.minCoeff(&r,&c);


    if(r >= i) return r+1;
    else return r;
}

std::string GMM::print_info(){
    std::string infos = "";
    for(const auto& comps : _model){
        infos += "class " + std::to_string(comps.first) + " have " + std::to_string(comps.second.size()) + " components\n";
        for(const auto& comp : comps.second){
            infos += comp->print_parameters();
        }
    }
    return infos;
}

std::string GMM::to_string(){

}
