#include "iagmm/gmm.hpp"

using namespace iagmm;

void GMM::operator ()(const tbb::blocked_range<size_t>& r){
    double sum = _sign > 0 ? _pos_sum : _neg_sum;
    double val;

    model_t model = _sign > 0 ? _pos_components : _neg_components;

    Eigen::VectorXd X = _X;

    for(size_t i=r.begin(); i != r.end(); ++i){
        val = model[i]->get_factor()*model[i]->compute_multivariate_normal_dist(X)
                /model[i]->compute_multivariate_normal_dist(model[i]->get_mu());
        sum += val;

    }
    if(_sign > 0)
        _pos_sum = sum;
    else _neg_sum = sum;
}

double GMM::compute_GMM(Eigen::VectorXd X){
    _X = X;
    _pos_sum = 0;
    _neg_sum = 0;
    _sign = 1;
    tbb::parallel_reduce(tbb::blocked_range<size_t>(0,_pos_components.size()),*this);
    _sign = -1;
    tbb::parallel_reduce(tbb::blocked_range<size_t>(0,_neg_components.size()),*this);

    return _pos_sum/(_pos_sum+_neg_sum);
}



void GMM::update_factors(){
    for(auto& c: _pos_components)
        c->set_factor((double)c->size()/(((double)_pos_components.size() +
                                            (double)_neg_components.size())*(double)number_of_samples()));

    for(auto& c: _neg_components)
        c->set_factor((double)c->size()/(((double)_pos_components.size() +
                                            (double)_neg_components.size())*(double)number_of_samples()));

}

double GMM::unit_factor(){
    return 1./(((double)_pos_components.size() +
             (double)_neg_components.size())*(double)number_of_samples());
}


void GMM::_new_component(const std::vector<Eigen::VectorXd>& samples, double label){
    Component::Ptr component(new Component(2,label));

    for(const auto& sample : samples)
        component->add(sample);

    component->update_parameters();

    if(label > 0)
        _pos_components.push_back(component);
    else _neg_components.push_back(component);
    update_factors();
}

double GMM::model_score(const std::vector<Eigen::VectorXd>& samples, const std::vector<double> &label){

    double score = 0;
    std::vector<double> predictions;

    for(const auto& s : samples)
        predictions.push_back(compute_GMM(s));

    for(int i = 0; i < predictions.size(); i++)
        score += (predictions[i]-label[i])*(predictions[i]-label[i]);

   return 1.f/score/*/predictions.size()*/;
}

std::vector<double> GMM::model_scores(){
    double score = 0;
    std::vector<double> scores;
    for(const auto& comp: _pos_components){
        for(const auto& s: comp->get_samples()){
            score += compute_GMM(s);
        }
        scores.push_back(score/(double)comp->get_samples().size());
    }
    for(const auto& comp: _neg_components){
        for(const auto& s: comp->get_samples()){
            score += (1-compute_GMM(s));
        }
        scores.push_back(score/(double)comp->get_samples().size());
    }
    return scores;
}

void GMM::knn(const Eigen::VectorXd& center, const std::vector<Eigen::VectorXd> &samples, const std::vector<double> &label,
              std::vector<Eigen::VectorXd> &output, std::vector<double> &label_output, int k){
    double min_dist, dist;
    int min_index;
    std::vector<Eigen::VectorXd> cpy_samples(samples);
    std::vector<double> cpy_label(label);
    for(int j = 0; j < k; j++){
        min_dist = sqrt((cpy_samples[0] - center).transpose()*(cpy_samples[0] - center));
        min_index = 0;
        for(int i = 1; i < cpy_samples.size(); i++){
            dist =  sqrt((cpy_samples[i] - center).transpose()*(cpy_samples[i] - center));
            if(dist < min_dist){
                min_index = i;
                min_dist = dist;
            }
        }
        output.push_back(cpy_samples[min_index]);
        label_output.push_back(cpy_label[min_index]);
        cpy_samples.erase(cpy_samples.begin() + min_index);
        cpy_samples.shrink_to_fit();
        cpy_label.erase(cpy_label.begin() + min_index);
        cpy_label.shrink_to_fit();
    }
}

void GMM::_split(int sign, const std::vector<Eigen::VectorXd>& samples, const std::vector<double> &label){
    model_t components = sign > 0 ? _pos_components : _neg_components;
    double score = 0, intern_score = 0;
    std::vector<Eigen::VectorXd> knn_output;
    std::vector<double> lbl_output;
    std::vector<Component::Ptr> new_comps;
    for(auto& comp : components){
        if(comp->size() < 4)
            continue;
        knn_output.clear();
        lbl_output.clear();
        score = 0;
        knn(comp->get_mu(),samples,label,knn_output,lbl_output,comp->size());

        for(int i = 0; i < knn_output.size(); i++){
            score += (comp->get_sign()*comp->compute_multivariate_normal_dist(knn_output[i])
                      /comp->compute_multivariate_normal_dist(comp->get_mu())- lbl_output[i])*
                    (comp->get_sign()*comp->compute_multivariate_normal_dist(knn_output[i])
                     /comp->compute_multivariate_normal_dist(comp->get_mu()) - lbl_output[i]);
        }
        score = score/(double)knn_output.size();
        intern_score = comp->component_score();
//        std::cout << "score : " << score << " vs intern score : " << intern_score << std::endl;

        if(score > 2.*intern_score){
            Component::Ptr new_component = comp->split();
            if(new_component){
                std::cout << "-_- SPLIT _-_" << std::endl;
                new_comps.push_back(new_component);
            }
        }
    }
    for(auto& comp : new_comps)
        components.push_back(comp);

    if(sign > 0)
        _pos_components = components;
    else _neg_components = components;

    update_factors();
}

Eigen::VectorXd GMM::next_sample(int* space, Eigen::MatrixXd& means_entropy_map){

    std::vector<std::pair<double,Eigen::VectorXd>> list_candidate_samples;

    double actual_entropy, next_entropy, diameter;
    Component test_comp;
    Eigen::VectorXd proposed_sample(2);
    means_entropy_map = Eigen::MatrixXd::Constant(space[0],space[1],-10000.);
    Eigen::MatrixXd entropy_map(space[0],space[1]);
    for(const auto& comp : _pos_components){
        actual_entropy = comp->entropy();
        for(int i = 0; i < space[0]; i++){
            for(int j = 0; j < space[1]; j++){
                proposed_sample(0) = (double)i/(double)space[0];
                proposed_sample(1) = (double)j/(double)space[1];
                test_comp = Component(*comp);
                test_comp.add(proposed_sample);
                test_comp.update_parameters();
                test_comp.set_factor((double)test_comp.size()/(
                                         ((double)_pos_components.size()+(double)_neg_components.size())*
                                         ((double) number_of_samples() + 1)));
                entropy_map(i,j) = fabs(test_comp.entropy()-actual_entropy);
                means_entropy_map(i,j) = entropy_map(i,j) > means_entropy_map(i,j) ?
                            entropy_map(i,j) :  means_entropy_map(i,j) ;
            }
        }
        int r,c;
        next_entropy = entropy_map.maxCoeff(&r,&c);
        proposed_sample(0) = r;
        proposed_sample(1) = c;
        list_candidate_samples.push_back(std::make_pair(next_entropy,proposed_sample));

    }
    for(const auto& comp : _neg_components){
        actual_entropy = comp->entropy();
        for(int i = 0; i < space[0]; i++){
            for(int j = 0; j < space[1]; j++){
                proposed_sample(0) = (double)i/(double)space[0];
                proposed_sample(1) = (double)j/(double)space[1];
                test_comp = Component(*comp);
                test_comp.add(proposed_sample);
                test_comp.update_parameters();
                test_comp.set_factor((double)test_comp.size()/(
                                         ((double)_pos_components.size()+(double)_neg_components.size())*
                                         ((double) number_of_samples() + 1)));
                entropy_map(i,j) = fabs(test_comp.entropy()-actual_entropy);
                means_entropy_map(i,j) = entropy_map(i,j) > means_entropy_map(i,j) ?
                            entropy_map(i,j) :  means_entropy_map(i,j);

            }
        }
        int r,c;
        next_entropy = entropy_map.maxCoeff(&r,&c);
        proposed_sample(0) = r;
        proposed_sample(1) = c;
        list_candidate_samples.push_back(std::make_pair(next_entropy,proposed_sample));

    }



    if(list_candidate_samples.empty()){
        Eigen::VectorXd v(2);
        v << 0,0;
        return v;
    }

//    double min_entr = list_candidate_samples.front().first;
//    Eigen::VectorXd elected_sample = list_candidate_samples.front().second;
    for(const auto& candidate : list_candidate_samples){
        std::cout << candidate.second << std::endl;
        std::cout << " -- " << std::endl;
        //        if(min_entr > candidate.first)
//            elected_sample = candidate.second;
    }
    int r,c;
    next_entropy = means_entropy_map.maxCoeff(&r,&c);
    proposed_sample(0) = r;
    proposed_sample(1) = c;
    return proposed_sample;
}

void GMM::update_model(const std::vector<Eigen::VectorXd> &samples, const std::vector<double> &label){
    //CAUTION: This update make the assumption of one new sample at a time.

    std::vector<Eigen::VectorXd> new_sample;
    new_sample.push_back(samples.back());

    //Base cases
    if(_pos_components.empty() && _neg_components.empty()){
        _new_component(new_sample,label.back());
        update_factors();
        return;
    }
    //--

    std::cout << "nb of pos components : " << _pos_components.size() << std::endl;
    std::cout << "nb of neg components : " << _neg_components.size() << std::endl;

    if(_pos_components.empty() && label.back() > 0){
        _new_component(new_sample,label.back());
        update_factors();
        return;
    }
    if(_neg_components.empty() && label.back() < 0){
        _new_component(new_sample,label.back());
        update_factors();
        return;
    }

    //add the new sample
    int r,c;
    if(label.back() > 0){
        Eigen::MatrixXd d2((int)new_sample.size(), (int)_pos_components.size());

        for (int j = 0; j < _pos_components.size(); j++) {
            for (int k=0; k < new_sample.size(); k++) {
                d2(k,j) = (new_sample[k]-_pos_components[j]->get_mu()).squaredNorm();
            }
        }

        for (int k=0; k < new_sample.size(); k++) {
            d2.row(k).minCoeff(&r,&c);
            _pos_components[c]->add(new_sample[k]);
            _pos_components[c]->update_parameters();
        }
    }else{
        Eigen::MatrixXd d2((int)new_sample.size(), (int)_neg_components.size());
        for (int j = 0; j < _neg_components.size(); j++) {
            for (int k=0; k < new_sample.size(); k++) {
                d2(k,j) = (new_sample[k]-_neg_components[j]->get_mu()).squaredNorm();
            }
        }

        for (int k=0; k < new_sample.size(); k++) {
            d2.row(k).minCoeff(&r,&c);
            _neg_components[c]->add(new_sample[k]);
            _neg_components[c]->update_parameters();
        }
    }

    update_factors();
    //--

    _split(1,samples,label);
    _split(-1,samples,label);


//    if(_pos_comp_ind.empty() || _neg_comp_ind.empty())
//        return;


//    update_factors();



    for(auto& comp : _pos_components)
        comp->update_parameters();

    for(auto& comp : _neg_components)
        comp->update_parameters();

}

std::vector<int> GMM::find_closest_components(double& min_dist, double sign){
    model_t comp = sign > 0 ? _pos_components : _neg_components;

    std::vector<int> indexes(2);
    indexes[0] = 0;
    indexes[1] = 1;


    min_dist = (comp[0]->get_mu()-comp[1]->get_mu()).squaredNorm();

    double dist;
    for(int i = 1; i < comp.size(); i++){
        for(int j = i+1; j < comp.size(); j++){
            dist = (comp[i]->get_mu()-comp[j]->get_mu()).squaredNorm();
            if(dist < min_dist){
                indexes[0] = i;
                indexes[1] = j;
            }
        }
    }

    return indexes;
}

int GMM::find_closest(int i, double &min_dist, double sign){
    model_t comp = sign > 0 ? _pos_components : _neg_components;


    Eigen::VectorXd distances(comp.size()-1);
    int k = 0;
    for(int j = 0; j < comp.size(); j++){
        if(j == i)
            continue;

        distances(k) =  (comp[i]->get_mu() - comp[j]->get_mu()).squaredNorm();
        k++;
    }
    int r, c;
    min_dist = distances.minCoeff(&r,&c);


    if(r >= i) return r+1;
    else return r;
}

