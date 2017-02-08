#include "iagmm/gmm.hpp"
#include <map>

using namespace iagmm;

void GMM::operator ()(const tbb::blocked_range<size_t>& r){
    double val;
    double sum = _sum_map[_current_lbl];

    Eigen::VectorXd X = _X;

    for(size_t i=r.begin(); i != r.end(); ++i){
        val = _model[_current_lbl][i]->get_factor()*_model[_current_lbl][i]->compute_multivariate_normal_dist(X)
                /_model[_current_lbl][i]->compute_multivariate_normal_dist(_model[_current_lbl][i]->get_mu());
        sum += val;

    }
    _sum_map[_current_lbl] = sum;
}

double GMM::compute_estimation(const Eigen::VectorXd& X, int lbl){
    _X = X;

    for(auto& sum : _sum_map)
        sum.second = 0;

    for(_current_lbl = 0; _current_lbl < _nbr_class; _current_lbl++)
        tbb::parallel_reduce(tbb::blocked_range<size_t>(0,_model[_current_lbl].size()),*this);

    double sum_of_sums = 0;
    for(const auto& sum : _sum_map)
        sum_of_sums +=  sum.second;


    return _sum_map[lbl]/sum_of_sums;
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
                score += compute_estimation(s,components.first);
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

void GMM::_merge(int lbl){
    model_t candidate_comp;
    double dist, score = 0, score2, candidate_score;
    int index;
    GMM candidate;
    std::vector<Eigen::VectorXd> local_samples;
    for(int i = 0; i < _model[lbl].size(); i++){
        index = find_closest(i,dist,lbl);

        if(dist <= _model[lbl][i]->diameter()+_model[lbl][index]->diameter()){
            score2 = _component_score(index,lbl);
            score = _component_score(i,lbl);

            candidate = GMM(_model);

            candidate.model()[lbl][i] =
                    candidate.model()[lbl][i]->merge(candidate.model()[lbl][index]);
            local_samples = candidate.model()[lbl][i]->get_samples();
            candidate.model()[lbl].erase(candidate.model()[lbl].begin() + index);

            candidate.update_factors();

            candidate_score = 0;
            for(const auto& s: local_samples){
                candidate_score += candidate.compute_estimation(s,lbl);
            }
            candidate_score = candidate_score/(double)local_samples.size();

            if(candidate_score >= (score + score2)/2.){
                std::cout << "-_- MERGE _-_" << std::endl;

                _model[lbl] = candidate.model()[lbl];

                update_factors();
                break;
            }
        }
    }
}

double GMM::_component_score(int i, int lbl){
    double score = 0;
    for(const auto& s: _model[lbl][i]->get_samples()){
        score += compute_estimation(s,lbl);
    }
    return score/(double)_model[lbl][i]->get_samples().size();
}

void GMM::_split(int lbl){
    double score = 0, intern_score = 0;
    TrainingData knn_output;
    std::vector<Component::Ptr> new_comps;
    for(auto& comp : _model[lbl]){
        if(comp->size() < 4)
            continue;
        knn_output.clear();
        score = 0;
        knn(comp->get_mu(),knn_output,comp->size());

        for(int i = 0; i < knn_output.size(); i++){
            score += (comp->compute_multivariate_normal_dist(knn_output[i].second)
                      /comp->compute_multivariate_normal_dist(comp->get_mu())- knn_output[i].first)*
                    (comp->compute_multivariate_normal_dist(knn_output[i].second)
                     /comp->compute_multivariate_normal_dist(comp->get_mu()) - knn_output[i].first);
        }
        score = score/(double)knn_output.size();
        intern_score = comp->component_score();
//        std::cout << "score : " << score << " vs intern score : " << intern_score << std::endl;

        if(score > intern_score){
            Component::Ptr new_component = comp->split();
            if(new_component){
                std::cout << "-_- SPLIT _-_" << std::endl;
                new_comps.push_back(new_component);
            }
        }
    }
    for(auto& comp : new_comps)
        _model[lbl].push_back(comp);/*window.isOpen()*/

    update_factors();
}

Eigen::VectorXd GMM::next_sample(const samples_t& samples, Eigen::VectorXd& choice_dist_map){
    choice_dist_map = Eigen::VectorXd::Zero(samples.size());



    if([&]() -> bool {for(auto& comp : _model) if(comp.second.empty()) return true; return false;}())
        return samples[rand()%(samples.size())];

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

        choice_dist_map = choice_dist_map/choice_dist_map.maxCoeff();
        i = 0;
        for(const auto& s : samples){
            choice_dist_map(i) = fabs((1 - scores[k_map(i)]) - choice_dist_map(i));
            cumul += choice_dist_map(i) ;
            choice_distribution.emplace(cumul,s);
            i++;
        }

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

        return possible_choice[rnb];
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


void GMM::append(const Eigen::VectorXd &sample,const int& lbl){
    int r,c; //row and column indexes
    add(sample,lbl);

    if(_model[lbl].empty()){
        _new_component(sample,lbl);
        return;
    }

    Eigen::VectorXd distances(_model[lbl].size());

    for (int j = 0; j < _model[lbl].size(); j++) {
        distances(j) = (sample-_model[lbl][j]->get_mu()).squaredNorm();
    }
    distances.minCoeff(&r,&c);
    _model[lbl][r]->add(sample);
    _model[lbl][r]->update_parameters();


    update_factors();
}

void GMM::update_model(){

    int n;
    for(int i = 0; i < _nbr_class; i++){
        n = _model[i].size();
        _split(i);
        if(n > 1)
            _merge(i);
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

std::string GMM::print_info(){
    std::string infos = "";
    for(const auto& comps : _model)
        infos += "class " + std::to_string(comps.first) + " have " + std::to_string(comps.second.size()) + " components\n";
    return infos;
}
