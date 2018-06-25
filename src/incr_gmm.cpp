#include <iagmm/incr_gmm.hpp>
#include <chrono>

using namespace iagmm;

void IncrementalGMM::add(const Eigen::VectorXd &sample, int lbl){
    int r,c; //row and column indexes

    _samples.add(lbl,sample);

    if(_model[lbl].empty()){
        new_component(sample,lbl);
        _last_index = 0;
        _last_label = lbl;
        return;
    }
    Eigen::VectorXd distances(_model[lbl].size());

    for (int j = 0; j < _model[lbl].size(); j++) {
        distances(j) = _model[lbl][j]->distance(sample);
    }
    distances.minCoeff(&r,&c);
    _model[lbl][r]->_incr_parameters(sample);
    update_factors();

    _last_index = r;
    _last_label = lbl;

}

void IncrementalGMM::update(){
    int n,rand_ind/*,max_size,max_ind,min_ind,min_size*/;
    _estimate_training_dataset();


    n = _model[_last_label].size();
    if(!_split(_model[_last_label][_last_index]) && n > 1)
        _merge(_model[_last_label][_last_index]);

    for(int i = 0; i < _nbr_class; i++){
        n = _model[i].size();

        if(n < 2) break;
        _estimate_training_dataset();

        do
            rand_ind = rand()%n;
        while(rand_ind == _last_index);
        if(!_split(_model[i][rand_ind]))
            _merge(_model[i][rand_ind]);
    }
}

std::vector<double> IncrementalGMM::compute_estimation(const Eigen::VectorXd &X){
    if([&]() -> bool { for(int i = 0; i < _nbr_class; i++){if(!_model.at(i).empty()) return false;} return true;}())
        return std::vector<double>(_nbr_class,1./(double)_nbr_class);

    Estimator<IncrementalGMM> estimator(this, X);

    return estimator.estimation();
}

void IncrementalGMM::new_component(const Eigen::VectorXd& sample, int label){
    Component::Ptr component(new Component(_dimension,label));
    component->_incr_parameters(sample);
    _model[label].push_back(component);
    update_factors();
}

void IncrementalGMM::update_factors(){
    for(int i = 0; i < _nbr_class; i++)
        _update_factors(i);
}

void IncrementalGMM::_update_factors(int lbl){
    double sum_size = 0;
    for(auto& components : _model[lbl])
        sum_size += components->size();

    for(auto& c : _model[lbl]){
        c->set_factor((double)c->size()/((double)sum_size));
    }
}

bool IncrementalGMM::_split(const Component::Ptr& comp){
    //* If the component have to few samples abort
    if(comp->size() < 5)
        return false;
    //*/

    //*/ Retrieve the label and the indice of the component
    int lbl = comp->get_label();
    int ind;
    for(ind = 0; ind < _model[lbl].size(); ind++)
        if(_model[lbl][ind].get() == comp.get())
            break;
    if(ind == _model[lbl].size())
        return false;
    //*/

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

    //* Capture time.
#ifdef VERBOSE
    std::cout << "split function" << std::endl;
    std::cout << "component of class " << lbl << std::endl;
    std::chrono::system_clock::time_point timer;
    timer  = std::chrono::system_clock::now();
#endif
    //*/

    int nb_comp = 0;
    for(const auto & models : _model){
        if(models.first != lbl)
            nb_comp += models.second.size();
    }

    //* Search for the closest component of comp
    Eigen::VectorXd distances(nb_comp);
    int closest_comp_ind, c;

    int counter = 0;
    std::vector<int> labels(nb_comp);
    std::vector<int> real_ind(nb_comp);
    for(int l = 0; l < _nbr_class; l++){
        if(l == lbl) // only consider models of other classes
            continue;

        if(_model[l].empty()) // if the model of classes l is empty
            continue;

        // compute the distances the component candidate for splitting and the components of the model of class l
        for(int j = 0; j < _model[l].size(); j++){
            distances(counter) = comp->distance(_model[l][j]->get_mu());
            labels[counter] = l;
            real_ind[counter] = j;
            counter++;
        }
    }
    distances.minCoeff(&closest_comp_ind,&c); // take the indice of the closest component
    //*/

    if(comp->intersect(_model[labels[closest_comp_ind]][real_ind[closest_comp_ind]])){
#ifdef VERBOSE
                std::cout << "-_- SPLIT _-_" << std::endl;
#endif

        Component::Ptr new_component(new Component(_dimension,comp->get_label()));

        Eigen::VectorXd eigenvalues;
        Eigen::MatrixXd eigenvectors;
        Eigen::VectorXd princ_axis;
        comp->compute_eigenvalues(eigenvalues,eigenvectors);
        int r,c;
        eigenvalues.maxCoeff(&r,&c);
        princ_axis = eigenvectors.col(r)/2.;
        std::cout << eigenvectors << std::endl;
        std::cout << eigenvalues << std::endl;
        std::cout << std::endl;

        std::cout << princ_axis << std::endl << std::endl;

        std::cout << princ_axis*princ_axis.transpose() << std::endl  << std::endl;

        std::cout << comp->get_covariance() << std::endl << std::endl;

        new_component->set_size(comp->size()*(1-_alpha));
        comp->set_size(comp->size()*_alpha);


        new_component->set_mu(comp->get_mu() + sqrt(_alpha/(1-_alpha))*_u*princ_axis);
        comp->set_mu(comp->get_mu() - sqrt((1-_alpha)/_alpha)*_u*princ_axis);

        new_component->set_covariance(
                    _alpha/(1-_alpha)*comp->get_covariance()
                    + (_beta*_u*_u - _beta - _u*_u)*1/(1-_alpha)*princ_axis*princ_axis.transpose()
                    + princ_axis*princ_axis.transpose());


        std::cout << new_component->get_covariance() << std::endl << std::endl;

//        std::cout << comp->get_mu().transpose()*new_component->covariance_pseudoinverse()*comp->get_mu() << std::endl << std::endl;


        comp->set_covariance(
                    (1-_alpha)/_alpha*comp->get_covariance()
                    + (_beta - _beta*_u*_u-1)*1/_alpha*princ_axis*princ_axis.transpose()
                    + princ_axis*princ_axis.transpose());

        std::cout << comp->get_covariance() << std::endl << std::endl;

//        std::cout << comp->get_mu().transpose()*comp->covariance_pseudoinverse()*comp->get_mu() << std::endl;

        _model[lbl].push_back(new_component);

        update_factors();


#ifdef VERBOSE
        std::cout << "Split finish, time spent : "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(
                         std::chrono::system_clock::now() - timer).count() << std::endl;
#endif

        return true;
    }
    return false;
}

bool IncrementalGMM::_merge(const Component::Ptr &comp){

    // if comp have too few samples or there is only one component in lbl model abort
    if(comp->size() < 5 || _model[comp->get_label()].size() == 1)
        return false;

    // Retrieve indice and label of comp
    int lbl = comp->get_label();
    int ind;
    for(ind = 0; ind < _model[lbl].size(); ind++)
        if(_model[lbl][ind].get() == comp.get())
            break;
    if(ind == _model[lbl].size())
        return false;

    //* Capture time.
#ifdef VERBOSE
    std::cout << "merge function" << std::endl;
    std::chrono::system_clock::time_point timer;
    timer  = std::chrono::system_clock::now();
#endif
    //*/

    //* Search for the closest component in lbl model from comp
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

    if(comp->intersect(_model[lbl][r])){
        double w1 = comp->get_factor();
        double w2 = _model[lbl][r]->get_factor();
        comp->set_size(comp->size() + _model[lbl][r]->size());
        update_factors();
        double w = comp->get_factor();
        Eigen::VectorXd mu1 = comp->get_mu();
        Eigen::VectorXd mu2 = _model[lbl][r]->get_mu();
        Eigen::VectorXd mu = (mu1*w1 + mu2*w2)/w;
        Eigen::MatrixXd covar1 = comp->get_covariance();
        Eigen::MatrixXd covar2 = _model[lbl][r]->get_covariance();
        Eigen::MatrixXd covar =
                w1/w*(covar1 + mu1*mu1.transpose())
                + w2/w*(covar2 + mu2*mu2.transpose())
                - mu*mu.transpose();
        comp->set_mu(mu);
        comp->set_covariance(covar);
        _model[lbl].erase(_model[lbl].begin()+ r);
    }
}

double IncrementalGMM::confidence(const Eigen::VectorXd &sample) const{return 1;}
int IncrementalGMM::next_sample(const std::vector<std::pair<Eigen::VectorXd, std::vector<double> > > &samples, Eigen::VectorXd &choice_dist_map){return 0;}
