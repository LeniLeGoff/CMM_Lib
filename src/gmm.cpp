#include "iagmm/gmm.hpp"
#include <map>

using namespace iagmm;

void GMM::_estimator::operator ()(const tbb::blocked_range<size_t>& r){
    double val;
    double sum = _sum_map[_current_lbl];

    Eigen::VectorXd X = _X;

    for(size_t i=r.begin(); i != r.end(); ++i){
        val = _model->model()[_current_lbl][i]->get_factor()*
                _model->model()[_current_lbl][i]->compute_multivariate_normal_dist(X)
               /* /_model[_current_lbl][i]->compute_multivariate_normal_dist(_model[_current_lbl][i]->get_mu())*/;
        sum += val;

    }
    _sum_map[_current_lbl] = sum;
}

double GMM::_estimator::estimation(int lbl){

    for(_current_lbl = 0; _current_lbl < _model->get_nbr_class(); _current_lbl++)
        tbb::parallel_reduce(tbb::blocked_range<size_t>(0,_model->model()[_current_lbl].size()),*this);

    double sum_of_sums = 0;
    for(const auto& sum : _sum_map)
        sum_of_sums +=  sum.second;

    return _sum_map[lbl]/sum_of_sums;
}

double GMM::compute_estimation(const Eigen::VectorXd& X, int lbl){

    if([&]() -> bool { for(int i = 0; i < _nbr_class; i++){if(!_model[i].empty()) return false;} return true;}())
        return 0.5;

    _estimator estimator(this, X);

    return estimator.estimation(lbl);
}

void GMM::_score_calculator::operator()(const tbb::blocked_range<size_t>& r){
    double sum = _sum;

    for(size_t i = r.begin(); i != r.end(); ++i){
        sum += fabs(_model->compute_estimation(_samples[i].second,_samples[i].first) - 1.);
    }
    _sum = sum;
}

double GMM::_score_calculator::compute(){
    tbb::parallel_reduce(tbb::blocked_range<size_t>(0,_samples.size()),*this);
    return _sum/((double)_samples.size());
}

void GMM::update_factors(){

    double sum_size = 0;
    for(auto& components : _model)
        sum_size += components.second.size();

    for(auto& components : _model){
        for(auto& c: components.second)
            c->set_factor((double)c->size()/(sum_size*(double)_samples.size()));
    }
}

double GMM::unit_factor(){

    double sum_size = 0;
    for(auto& components : _model)
        sum_size += components.second.size();

    return 1./(sum_size*(double)_samples.size());
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
    model_t candidate_comp;
    double dist, score = 0, score2, candidate_score;
    int index;
    GMM candidate;
    boost::random::uniform_real_distribution<> distri(0,1);

    std::vector<Eigen::VectorXd> local_samples;
//    std::vector<int> rand_ind(_model[lbl].size());
//    for(int i = 0; i < rand_ind.size(); i++)
//        rand_ind[i] = i;
//    for(int i = 0; i < rand_ind.size(); i++){
//        int n = rand()%rand_ind.size();
//        int tmp = rand_ind[n];
//        rand_ind[n] = rand_ind[i];
//        rand_ind[i] = tmp;
//    }
//    for(auto& i : rand_ind){
        index = find_closest(ind,dist,lbl);

//        if(dist <= _model[lbl][ind]->diameter()+_model[lbl][index]->diameter()){
            score2 = _component_score(index,lbl);
            score = _component_score(ind,lbl);

            candidate = GMM(_model);
            candidate.set_samples(_samples);

            candidate.model()[lbl][ind] =
                    candidate.model()[lbl][ind]->merge(candidate.model()[lbl][index]);
//            local_samples = candidate.model()[lbl][ind]->get_samples();
            TrainingData knn_output;
            candidate.knn(candidate.model()[lbl][ind]->get_mu(),knn_output,candidate.model()[lbl][ind]->size());
            candidate.model()[lbl].erase(candidate.model()[lbl].begin() + index);



            candidate.update_factors();

            candidate_score = 0;
            for(const auto& s: knn_output.get()){
                candidate_score += fabs(candidate.compute_estimation(s.second,s.first)-1.);
            }
            candidate_score = candidate_score/((double)knn_output.size());

            std::cout << lbl << " merge : candidate " << candidate_score << " vs  others " << (score + score2)/2. << std::endl;
            if(candidate_score < (score + score2)/2. ){
                std::cout << "-_- MERGE _-_" << std::endl;

                _model[lbl] = candidate.model()[lbl];

                update_factors();
//                break;
            }
//        }
//    }
}

void GMM::_merge_eigen(int ind, int lbl){
    GMM candidate;
    double score, score2, candidate_score;

    _score_calculator sc(this,_samples);
    score = sc.compute();

//    score = 0;
//    for(const auto& s: _samples.get()){
//        score += fabs(compute_estimation(s.second,s.first)-1.);
//    }
//    score = score/_samples.size();

    Eigen::VectorXd eigenval, eigenval2, diff_mu, ellipse_vect1, ellipse_vect2;
    Eigen::MatrixXd eigenvect, eigenvect2;
    _model[lbl][ind]->compute_eigenvalues(eigenval,eigenvect);
    for(int i = 0; i < _model[lbl].size(); i++){
        if(i == ind)
            continue;
        _model[lbl][i]->compute_eigenvalues(eigenval2,eigenvect2);
        diff_mu = _model[lbl][i]->get_mu() - _model[lbl][ind]->get_mu();
        ellipse_vect1 = _model[lbl][ind]->get_mu()
                + (eigenvect.transpose()*diff_mu/diff_mu.squaredNorm());
        ellipse_vect2 = _model[lbl][i]->get_mu()
                + (eigenvect2.transpose()*diff_mu/diff_mu.squaredNorm());

        if(diff_mu.squaredNorm() < ellipse_vect1.squaredNorm() + ellipse_vect2.squaredNorm()){


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
            candidate_score  = candidate_sc.compute();

            //            candidate_score = 0;
//            for(const auto& s: candidate.get_samples().get()){
//                candidate_score += fabs(candidate.compute_estimation(s.second,s.first)-1.);
//            }
//            candidate_score = candidate_score/((double)candidate.get_samples().size());
            std::cout << candidate_score << " <> " << score << std::endl;
            if(candidate_score <= score ){
                std::cout << "-_- MERGE _-_" << std::endl;
                _model[lbl][ind] = _model[lbl][ind]->merge(_model[lbl][i]);
                _model[lbl].erase(_model[lbl].begin() + i);
                update_factors();
                return;
            }
        }
    }
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
    double score = 0;
    for(const auto& s: _model[lbl][i]->get_samples()){
        score += fabs(compute_estimation(s,lbl)-1);
    }
    return score/(double)_model[lbl][i]->get_samples().size();
}

void GMM::_split(int ind, int lbl){
    double score = 0, intern_score = 0;
    TrainingData knn_output;
    std::vector<Component::Ptr> new_comps;
    boost::random::uniform_real_distribution<> dist(0,1);
//    std::vector<int> rand_ind(_model[lbl].size());
//    for(int i = 0; i < rand_ind.size(); i++)
//        rand_ind[i] = i;
//    for(int i = 0; i < rand_ind.size(); i++){
//        int n = rand()%rand_ind.size();
//        int tmp = rand_ind[n];
//        rand_ind[n] = rand_ind[i];
//        rand_ind[i] = tmp;
//    }
//    for(int& ind : rand_ind){
         if(_model[lbl][ind]->size() < 4)
            return;
        knn_output.clear();
        score = 0;
        knn(_model[lbl][ind]->get_mu(),knn_output,_model[lbl][ind]->size());


        double p;
        for(int i = 0; i < knn_output.size(); i++){
            if(lbl == knn_output[i].first)
                p = _model[lbl][ind]->compute_multivariate_normal_dist(knn_output[i].second)
                        /_model[lbl][ind]->compute_multivariate_normal_dist(_model[lbl][ind]->get_mu());
            else p = 1 - _model[lbl][ind]->compute_multivariate_normal_dist(knn_output[i].second)
                    /_model[lbl][ind]->compute_multivariate_normal_dist(_model[lbl][ind]->get_mu());
            score += fabs(p - 1);
        }
        score = score/(double)knn_output.size();
        intern_score = _model[lbl][ind]->component_score();
        std::cout << lbl << " split : " << "score : " << score << " vs intern score : " << intern_score << std::endl;

        if(score > intern_score){
            Component::Ptr new_component = _model[lbl][ind]->split();
            if(new_component){
                std::cout << "-_- SPLIT _-_" << std::endl;
                new_comps.push_back(new_component);
//                break;
            }
        }
//    }
    for(auto& comp : new_comps)
        _model[lbl].push_back(comp);
    update_factors();
}

void GMM::_split_eigen(int ind, int lbl){
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
            ellipse_vect1 = _model[lbl][ind]->get_mu()
                    + (eigenvect.transpose()*diff_mu/diff_mu.squaredNorm());
            ellipse_vect2 = comp->get_mu()
                    + (eigenvect2.transpose()*diff_mu/diff_mu.squaredNorm());

            if(diff_mu.squaredNorm() < fabs(ellipse_vect1.squaredNorm() - ellipse_vect2.squaredNorm())){
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


//                Component::Ptr new_component = _model[lbl][ind]->split();
//                if(new_component){
//                    std::cout << "-_- SPLIT _-_" << std::endl;
//                    _model[lbl].push_back(new_component);
//                    update_factors();
//                    return;
//                }
            }

        }
    }
}


int GMM::next_sample(const samples_t& samples, Eigen::VectorXd& choice_dist_map){
    choice_dist_map = Eigen::VectorXd::Zero(samples.size());



    if([&]() -> bool {for(auto& comp : _model) if(comp.second.empty()) return true; return false;}())
        return rand()%(samples.size());

    std::vector<double> scores = model_scores();
    std::multimap<double,Eigen::VectorXd> choice_distribution;

    int k =0, i = 0 ,min_k;
    Eigen::VectorXd k_map = Eigen::VectorXd::Zero(samples.size());
    double min, dist = 0, cumul = 0.;
    if([&]() -> bool {for(auto& comp : _model) if(comp.second.empty()) return false; return true;}()){
        for(const auto& s : samples){
            k=0,min_k=0;

            min = (s - _model[0][0]->get_mu()).squaredNorm()/
                    (_model[0][0]->get_factor());

            for(const auto& comps : _model){
                for(const auto& c : comps.second){
                    dist = (s - c->get_mu()).squaredNorm()/(c->get_factor());
                    if(min > dist){
                        min = dist;
                        min_k = k;
                    }
                    k++;
                }
            }

            choice_dist_map(i) = min;
            k_map(i) = min_k;
            i++;
        }
        double maxcoeff = choice_dist_map.maxCoeff();
        choice_dist_map = choice_dist_map/choice_dist_map.maxCoeff();
        i = 0;
        for(const auto& s : samples){
            choice_dist_map(i) = fabs((scores[k_map(i)]) - choice_dist_map(i));
            cumul += choice_dist_map(i) ;
            choice_distribution.emplace(cumul,s);
            i++;
        }

        if(cumul != cumul)
            return rand()%(samples.size());


        boost::random::uniform_real_distribution<> distrib(0.,cumul);
        double rand_nb = distrib(_gen);
        auto it = choice_distribution.lower_bound(rand_nb);
        double val = it->first;
        std::vector<Eigen::VectorXd> possible_choice;
        while(it->first == val){
            possible_choice.push_back(it->second);
            it++;
        }

        int rnb = rand()%(possible_choice.size());

        return rnb;
    }
}

void GMM::append(const std::vector<Eigen::VectorXd> &samples, const std::vector<int>& lbl){
    int r,c; //row and column indexes
    for(int i = 0 ; i < samples.size(); i++){
        add(samples[i],lbl[i]);

        if(_model[lbl[i]].empty()){
            _new_component(samples[i],lbl[i]);
            continue;
        }

        Eigen::VectorXd distances(_model[lbl[i]].size());

        for (int j = 0; j < _model[lbl[i]].size(); j++) {
                distances(j) = (samples[i]-_model[lbl[i]][j]->get_mu()).squaredNorm();
        }
        distances.minCoeff(&r,&c);
        _model[lbl[i]][r]->add(samples[i]);
        _model[lbl[i]][r]->update_parameters();
    }

    update_factors();
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


    update_factors();

    return r;
}

void GMM::update_model(int ind, int lbl){

    int n,rand_ind;
    n = _model[lbl].size();
    _split_eigen(ind,lbl);
    if(n > 1)
        _merge_eigen(ind,lbl);

//    for(int i = 0; i < _nbr_class; i++){
//        n = _model[i].size();
//        if(n < 2) break;
//        do
//            rand_ind = rand()%n;
//        while(rand_ind == ind);
//        _split_eigen(rand_ind,i);



//        do
//            rand_ind = rand()%n;
//        while(rand_ind == ind);
//        _merge_eigen(rand_ind,i);


//    }
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

std::string GMM::print_info(){
    std::string infos = "";
    for(const auto& comps : _model)
        infos += "class " + std::to_string(comps.first) + " have " + std::to_string(comps.second.size()) + " components\n";
    return infos;
}

std::string GMM::to_string(){

}
