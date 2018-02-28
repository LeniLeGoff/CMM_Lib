#include "iagmm/gmm.hpp"
#include <map>
#include <chrono>
#include <cmath>

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

    if([&]() -> bool { for(int i = 0; i < _nbr_class; i++){if(!_model.at(i).empty()) return false;} return true;}())
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
        if(_samples[i].first == _label || _all_samples)
            sum += std::log(_samples.estimations[i]);
    }
    _sum = sum;
}

double GMM::_score_calculator::compute(){
//    _model->compute_normalisation();
    tbb::parallel_reduce(tbb::blocked_range<size_t>(0,_samples.size()),*this);
    return _sum/((double)_samples.get_data(_label).size());
}
//--SCORE_CALCULATOR


//DISTRIBUTION_CONSTRUCTOR
//void GMM::_distribution_constructor::operator()(const tbb::blocked_range<size_t>& r){
//    for()
//}

//--DISTRIBUTION_CONSTRUCTOR

void GMM::update_factors(){
    for(int i = 0; i < _nbr_class; i++)
        _update_factors(i);
}


void GMM::_update_factors(int lbl){
    double sum_size = 0;
    for(auto& components : _model[lbl])
        sum_size += components->size();

    for(auto& c : _model[lbl]){
            c->set_factor((double)c->size()/((double)sum_size));
    }
}

double GMM::unit_factor(){

    double sum_size = 0;
    for(auto& components : _model)
        sum_size += components.second.size();

    return 1./((double)sum_size);
}


void GMM::new_component(const Eigen::VectorXd& sample, int label){
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

double GMM::confidence(const Eigen::VectorXd& X) const{
    int size = _model.at(0).size();
    for(int i = 1; i < _model.size() ; i++)
        size += _model.at(i).size();

    Eigen::VectorXd distances(size);
    int k = 0;
    for(int i = 0 ; i < _model.size(); i++){
        for (int j = 0; j < _model.at(i).size(); j++) {
            distances(k) = _model.at(i).at(j)->distance(X);
            k++;
        }
    }
    int r,c;
    distances.minCoeff(&r,&c);

    int lbl = 0, s = _model.at(lbl).size();
    while(r - s >= 0){
        r = r - s;
        lbl++;
        s = _model.at(lbl).size();
    }

    return _model.at(lbl).at(r)->compute_multivariate_normal_dist(X)/
            _model.at(lbl).at(r)->compute_multivariate_normal_dist(_model.at(lbl).at(r)->get_mu());
}


double GMM::loglikelihood(){
    _score_calculator sc(this,_samples);
    return sc.compute();
}

double GMM::loglikelihood(int label){
    _score_calculator sc(this,_samples,false,label);
    return sc.compute();
}

bool GMM::_merge(const Component::Ptr& comp){

    if(comp->size() < 5 || _model[ comp->get_label()].size() == 1)
        return false;

    int lbl = comp->get_label();
    int ind;
    for(ind = 0; ind < _model[lbl].size(); ind++)
        if(_model[lbl][ind].get() == comp.get())
            break;
    if(ind == _model[lbl].size())
        return false;

    std::cout << "merge function" << std::endl;
    std::chrono::system_clock::time_point timer;
    timer  = std::chrono::system_clock::now();

    GMM candidate;
    double score, score2, candidate_score;


    score = loglikelihood();/*
    _score_calculator sc(this,_samples);
    score = sc.compute();*/


    Eigen::VectorXd distances(_model[lbl].size());

    int r, c;
    for (int j = 0; j < _model[lbl].size(); j++) {
        if(_model[lbl][j] == comp){
            distances(j) = 1000000;
            continue;
        }
        distances(j) = comp->distance(_model[lbl][j]->get_mu());
    }
    distances.minCoeff(&r,&c);
    if(_model[lbl][r]->get_samples().size() < 5)
        return false;



    if(comp->intersect(_model[lbl][r])){
        //            score = _component_score(ind,lbl);
        //            score2 = _component_score(i,lbl);


        candidate = GMM(_model);
        candidate.set_samples(_samples);
        candidate.model()[lbl][ind]->merge(candidate.model()[lbl][r]);
        //            TrainingData knn_output;
        //            candidate.knn(candidate.model()[lbl][ind]->get_mu(),knn_output,candidate.model()[lbl][ind]->size());

        candidate.model()[lbl].erase(candidate.model()[lbl].begin() + r);
        candidate.update_factors();

        candidate._estimate_training_dataset();
//        _score_calculator candidate_sc(&candidate,/*knn_output*/candidate.get_samples());
//        candidate_score = candidate_sc.compute();
        candidate_score = candidate.loglikelihood();

        std::cout << candidate_score << " >? " << score << std::endl;
        if(candidate_score > score){
            std::cout << "-_- MERGE _-_" << std::endl;
            _model[lbl][ind]->merge(_model[lbl][r]);
            _model[lbl].erase(_model[lbl].begin() + r);
            update_factors();
            std::cout << "Merge finish, time spent : "
                      << std::chrono::duration_cast<std::chrono::milliseconds>(
                             std::chrono::system_clock::now() - timer).count() << std::endl;
            return true;
        }

    }
    std::cout << "Merge finish, time spent : "
              << std::chrono::duration_cast<std::chrono::milliseconds>(
                     std::chrono::system_clock::now() - timer).count() << std::endl;


    return false;
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
    _score_calculator sc(this,knn_output,lbl);
    return sc.compute();
}


bool GMM::_split(const Component::Ptr& comp){

    //*If the component have less than 4 element abort
    if(comp->size() < 5)
       return false;
    //*/

    int lbl = comp->get_label();
    int ind;
    for(ind = 0; ind < _model[lbl].size(); ind++)
        if(_model[lbl][ind].get() == comp.get())
            break;
    if(ind == _model[lbl].size())
        return false;


    //*/verify the model of other classes are empty. If all the model of other classes are empty abort
    bool keep_going = false;
    for(int l = 0; l < _nbr_class; l++){
        if(l == lbl)
            continue;
        if(_model[l].empty())
            continue;
        keep_going = true;
        break;
    }
    if(!keep_going)
        return false;
    //*/


    //* Capture time. TO DO put an option at compilation
    std::cout << "split function" << std::endl;
    std::chrono::system_clock::time_point timer;
    timer  = std::chrono::system_clock::now();
    //*/


    //* Local variables needed for the algorithm
    GMM candidate;
    double cand_score, score;
    //*/

    //* compute of the score of the current model
    score = loglikelihood();
//    _score_calculator sc(this,_samples);
//    score = sc.compute();
    //*/

    for(int l = 0; l < _nbr_class; l++){
        if(l == lbl) // only consider models of other classes
            continue;

//        if(_model[l].empty()) // if the model of classes l is empty
//            continue;

        //* compute the distances the component candidate for splitting and the components of the model of class l
        Eigen::VectorXd distances(_model[l].size());
        int closest_comp_ind, c;
        for (int j = 0; j < _model[l].size(); j++) {
            distances(j) = comp->distance(_model[l][j]->get_mu());
        }
        distances.minCoeff(&closest_comp_ind,&c); // take the indice of the closest component
        //*/

        if(comp->intersect(_model[l][closest_comp_ind])){ //if the components intersect

            candidate = GMM(_model); //Create a model candidate
            candidate.set_samples(_samples);
            Component::Ptr new_component = candidate.model()[lbl][ind]->split(); //split the component

            if(new_component){ //if the component is splitted
                candidate.model()[lbl].push_back(new_component);
                candidate.update_factors();

                candidate._estimate_training_dataset();//estimation for dataset

                //* compute the score of the model candidate
//                _score_calculator candidate_sc(&candidate,candidate.get_samples());
//                cand_score = candidate_sc.compute();
                cand_score = candidate.loglikelihood();
                //*/

                std::cout << cand_score << " >? " << score << std::endl;
                if(cand_score > score){ //if the candidate model score is greater than the score of the current model
                    std::cout << "-_- SPLIT _-_" << std::endl;
                    new_component = _model[lbl][ind]->split();
                    _model[lbl].push_back(new_component);
                    update_factors();

                    //* Display time spent for the algorithm. TO DO put an option at compilation
                    std::cout << "Split finish, time spent : "
                              << std::chrono::duration_cast<std::chrono::milliseconds>(
                                     std::chrono::system_clock::now() - timer).count() << std::endl;
                    //*/

                    return true;
                }
            }

        }
    }

    //* Display time spent for the algorithm. TO DO put an option at compilation
    std::cout << "Split finish, time spent : "
              << std::chrono::duration_cast<std::chrono::milliseconds>(
                     std::chrono::system_clock::now() - timer).count() << std::endl;
    //*/

    return false;
}

int GMM::next_sample(const std::vector<std::pair<Eigen::VectorXd,double>> &samples, Eigen::VectorXd &choice_dist_map){
    choice_dist_map = Eigen::VectorXd::Constant(samples.size(),0.5);
    boost::random::uniform_int_distribution<> dist_uni(0,samples.size());

    if(_samples.size() <= 50)
        return dist_uni(_gen);

    double total = 0,cumul = 0, avg = 0;

    std::map<double,int> choice_distibution;
    std::vector<double> w(samples.size());
    boost::random::uniform_real_distribution<> distrib(0,1);

    tbb::parallel_for(tbb::blocked_range<size_t>(0,choice_dist_map.rows()),
                      [&](const tbb::blocked_range<size_t>& r){
        double est;
        bool pos = _samples.get_data(1).size() <= _samples.get_data(0).size();
        for(size_t i = r.begin(); i != r.end(); i++){
            est = samples[i].second;
            if(pos){
                est = -(2*est*log(2*est)-2*est);
//                if(est < .5)
//                    est = 0;
            }
            else {
                est = 1-est;
                est = -(2*est*log(2*est)-2*est);
//                if(est <= .5)
//                    est = (1-est);
//                else
//                    est = 0.;
            }

//            if(est > .5)
//                est = 2.* (1 - est);
//            else est = 2.*est;
//            double c = confidence(samples[i].first);
//            if(c > 1) c = 1;
//            else if (c < 10e-4) c = 0;
////            std::cout << "c : " << c << " -- u : " << est << std::endl;
////            w[i] = (fabs(1-c) + est)/2.;

            w[i] = est;//*(1-c);
//            w[i] = 1./(1. + exp(-60.*((/*fabs(1-c)*/ + est)/*/2.*/ - .5)));
//            if(w[i] != w[i])
//                w[i] = 0;
//            else if(w[i] > 1)
//                w[i] = 1;
        }
    });

//    for(int i = 0; i < choice_dist_map.rows(); ++i){
//        avg += w[i];
//    }
//    avg = avg/(double)w.size();
//    for(int i = 0; i < w.size(); i++){
//        w[i] = 1./(1. + exp(-50.*(w[i] - avg)));


//    }
    bool all_zero = true;
    for(int i = 0; i < choice_dist_map.rows(); ++i){
//        std::cout << "w : " << w[i] << std::endl;

        choice_dist_map(i) =  w[i] /*>= avg ? w[i] : 0*/;
        all_zero = all_zero && choice_dist_map(i) == 0;
        total += choice_dist_map(i);
    }
    if(all_zero)
        return dist_uni(_gen);
    for(int i = 0; i < choice_dist_map.rows(); ++i){
        cumul += choice_dist_map(i);
        choice_distibution.emplace(cumul/total,i);
    }
    return choice_distibution.lower_bound(distrib(_gen))->second;
}


void GMM::append(const std::vector<Eigen::VectorXd> &samples, const std::vector<int>& lbl){
    int r,c; //row and column indexes
    for(int i = 0 ; i < samples.size(); i++){
        if(!append(samples[i],lbl[i]))
            continue;
    }
}

void GMM::add(const Eigen::VectorXd &sample, int lbl){
    _last_index = append(sample,lbl);
    _last_label = lbl;
}

int GMM::append(const Eigen::VectorXd &sample,const int& lbl){
    int r,c; //row and column indexes
    double q = compute_quality(sample,lbl);

    _samples.add(lbl,sample,q);

    if(_model[lbl].empty()){
        new_component(sample,lbl);
        return 0;
    }
    Eigen::VectorXd distances(_model[lbl].size());

    for (int j = 0; j < _model[lbl].size(); j++) {
        distances(j) = _model[lbl][j]->distance(sample);
    }
    distances.minCoeff(&r,&c);
    _model[lbl][r]->add(sample);
    _model[lbl][r]->update_parameters();

    return r;
}


void GMM::append_EM(const Eigen::VectorXd &sample,const int& lbl){
    int r,c; //row and column indexes
    _samples.add(lbl,sample);
}

void GMM::update(){
    if(_update_mode == STOCHASTIC)
        update_model(_last_index,_last_label);
    else update_model();
}

void GMM::update_model(){
    std::vector<Component::Ptr> comp;
    for(int i = 0; i < _nbr_class; i++){
        for(int j = 0; j < _model[i].size(); j++){
            comp.push_back(_model[i][j]);
        }
    }
    for(int i = 0; i < comp.size(); i++){
        _estimate_training_dataset();

        if(!_split(comp[i]))
            _merge(comp[i]);
    }
    for(auto& components : _model)
        for(auto& comp : components.second)
            comp->update_parameters();
}

void GMM::update_model(int ind, int lbl){

    int n,rand_ind/*,max_size,max_ind,min_ind,min_size*/;
    _estimate_training_dataset();


    n = _model[lbl].size();
//    _split(ind,lbl);
    if(!_split(_model[lbl][ind]) && n > 1)
        _merge(_model[lbl][ind]);



    for(int i = 0; i < _nbr_class; i++){
        n = _model[i].size();

        if(n < 2) break;
        _estimate_training_dataset();

//        max_size = _model[i][0]->size();
//        max_ind = 0;
//        min_size = _model[i][0]->size();
//        min_ind = 0;
//        for(int j = 1; j < _model[i].size(); j++){
//            if(max_size < _model[i][j]->size()){
//                max_size = _model[i][j]->size();
//                max_ind = j;
//            }
//            if(min_ind > _model[i][j]->size()){
//                min_ind = _model[i][j]->size();
//                min_ind = j;
//            }
//        }

//        if(min_ind == max_ind){
//            if(!_split(max_ind,i))
//                _merge(max_ind,i);
//        }else{
//            _split(max_ind,i);
//            _merge(min_ind,i);
//        }
        do
            rand_ind = rand()%n;
        while(rand_ind == ind);
        if(!_split(_model[i][rand_ind]))
            _merge(_model[i][rand_ind]);
//        do
//            rand_ind = rand()%n;
//        while(rand_ind_2 == ind);

    }

    for(auto& components : _model)
        for(auto& comp : components.second)
            comp->update_parameters();
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

double GMM::compute_quality(const Eigen::VectorXd& sample,int lbl)
{
    double score;
    score = 0;
    for(int k = 0; k < _nbr_class ; k++){
        for(int j = 0; j < _model[k].size(); j++){
            if(k == lbl)
                score += 1/_distance(sample,_model[k][j]->get_mu());
            else score -= 1/_distance(sample,_model[k][j]->get_mu());
        }
    }
    return score;
}

void GMM::update_dataset_thres(double threshold){
    std::vector<int> indexes_sample, indexes_model, indexes_model_s, lbls;
    for(int i = 0; i < _samples.size(); i++){
        for(auto& model : _model){
            for(int k = 0; k < model.second.size(); k++){
                for(int j = 0; j < model.second[k]->get_samples().size(); j++){
                    if(_samples[i].second == model.second[k]->get_samples()[j]){
                        if(_samples.get_qualities()[i] < threshold){
                            indexes_sample.push_back(i);
                            indexes_model.push_back(k);
                            indexes_model_s.push_back(j);
                            lbls.push_back(model.first);
                        }
                    }
                }
            }
        }
    }

    for(int& i : indexes_sample)
        _samples.erase(i);
    for(int i = 0; i < indexes_model.size(); i++){
        _model[lbls[i]][indexes_model[i]]->remove_sample(indexes_model_s[i]);
        if(_model[lbls[i]][indexes_model[i]]->size() == 0){
            _model[lbls[i]].erase(_model[lbls[i]].begin() + indexes_model[i]);
            _model[lbls[i]].shrink_to_fit();
        }
    }
}

void GMM::update_dataset(){
    if(_samples.size() < _dataset_size_max)
        return;
    double min_q = _samples.get_qualities()[0];
    int min_lbl = 0, min_ind_comp = 0, min_ind_comp_s = 0, min_ind_s = 0;

    for(auto& model : _model){
        for(int k = 0; k < model.second.size(); k++){
            for(int i = 0; i < model.second[k]->get_samples().size(); i++){
                if(_samples[0].second == model.second[k]->get_samples()[i]){
                    min_q = _samples.get_qualities()[0];
                    min_lbl = model.first;
                    min_ind_comp = k;
                    min_ind_comp_s = i;
                    min_ind_s = 0;
                }
            }
        }
    }
    while(_samples.size() > _dataset_size_max){

        for(auto& model : _model){
            for(int k = 0; k < model.second.size(); k++){
                for(int i = 0; i < model.second[k]->get_samples().size(); i++){
                    for(int j = 1; j < _samples.size();j++){
                        if(_samples[j].second == model.second[k]->get_samples()[i]
                                && _samples.get_qualities()[j] < min_q){
                            min_q = _samples.get_qualities()[j];
                            min_lbl = model.first;
                            min_ind_comp = k;
                            min_ind_comp_s = i;
                            min_ind_s = j;
                        }
                    }
                }
            }
        }
        std::cout << "----- minimum quality :  " << min_q << std::endl;
        _model[min_lbl][min_ind_comp]->remove_sample(min_ind_comp_s);
        if(_model[min_lbl][min_ind_comp]->size() == 0){
            _model[min_lbl].erase(_model[min_lbl].begin()+min_ind_comp);
            _model[min_lbl].shrink_to_fit();
        }
        _samples.erase(min_ind_s);
    }
}

void GMM::_expectation(int lbl){
    std::vector<Eigen::VectorXd> samples = _samples.get_data(lbl);

    _membership.emplace(lbl,Eigen::MatrixXd(_model[lbl].size(),samples.size()));

    Eigen::VectorXd estimations(_model[lbl].size());
    double sum;
    for(int k = 0; k < samples.size(); k++){
        sum = 0;
        for(int i = 0; i < _model[lbl].size(); i++){
            estimations(i) = _model[lbl][i]->compute_multivariate_normal_dist(samples[k])/
                    _model[lbl][i]->compute_multivariate_normal_dist( _model[lbl][i]->get_mu());
            sum += _model[lbl][i]->get_factor()* estimations(i);
            _membership[lbl](i,k) = _model[lbl][i]->get_factor()* estimations(i);
        }
        _membership[lbl].col(k) = _membership[lbl].col(k)/sum;
    }
}

void GMM::_maximisation(int lbl){
    Eigen::VectorXd normalisation = Eigen::VectorXd::Zero(_model[lbl].size());
    Eigen::VectorXd new_mu;
    Eigen::MatrixXd new_covariance;
    std::vector<Eigen::VectorXd> samples = _samples.get_data(lbl);
    for(int k = 0; k < _membership[lbl].cols(); k++){
        normalisation += _membership[lbl].col(k).transpose();
    }
    for(int i = 0; i < _model[lbl].size(); i++){
        //new factor
        _model[lbl][i]->set_factor(normalisation(i)/samples.size());

        //new mean
        new_mu = Eigen::VectorXd::Zero(_model[lbl][i]->get_dimension());
        for(int k = 0; k < _membership[lbl].cols(); k++)
            new_mu += _membership[lbl](i,k)*samples[k];
        new_mu = new_mu / normalisation(i);
        _model[lbl][i]->set_mu(new_mu);

        //new covariance
        new_covariance = Eigen::MatrixXd::Zero(_model[lbl][i]->get_dimension(),
                                               _model[lbl][i]->get_dimension());
        for(int k = 0; k < _membership[lbl].cols(); k++){
            new_covariance += _membership[lbl](i,k)*(samples[k] - new_mu)*(samples[k] - new_mu).transpose();
        }
        new_covariance = new_covariance / normalisation(i);
        _model[lbl][i]->set_covariance(new_covariance);
    }
}

void GMM::EM_init(){
    boost::random::uniform_real_distribution<> dist(0,1);
    for(int j = 0; j < _model.size(); j++){
        _membership.emplace(j,Eigen::MatrixXd(max_component,max_component));
        for(int i = 0; i < max_component; i++){
            Eigen::VectorXd mu(_dimension);
            Eigen::MatrixXd covariance(_dimension,_dimension);
            for(int k = 0; k < _dimension; k++)
                mu(k) = dist(_gen);

            new_component(mu,j);
            for(int l = 0; l < max_component; l++)
                _membership[j](i,l) = dist(_gen);
        }
    }


    for(int j = 0; j < _model.size(); j++)
        _maximisation(j);

}

void GMM::EM_step(){
    _membership.clear();
    for(int lbl = 0; lbl < _model.size(); lbl++){
        _expectation(lbl);
        _maximisation(lbl);
    }
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


