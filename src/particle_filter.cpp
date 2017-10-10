/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

const bool debug = false;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
    num_particles = 100;
    weights.resize(num_particles);

    std::default_random_engine gen;
    std::normal_distribution<double> nd_x(x,std[0]);
    std::normal_distribution<double> nd_y(y,std[1]);
    std::normal_distribution<double> nd_theta(theta, std[2]);
    
    for (int i = 0;i < num_particles; i++) {
        Particle p;
        p.x = nd_x(gen);
        p.y = nd_y(gen);
        p.theta = nd_theta(gen);
        p.weight = 1.0;
        
        particles.push_back(p);        
    }
    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
    std::default_random_engine gen;
    std::normal_distribution<double> nd_x(0.0,std_pos[0]);
    std::normal_distribution<double> nd_y(0.0,std_pos[1]);
    std::normal_distribution<double> nd_theta(0.0, std_pos[2]);

    for (int i = 0;i < num_particles; i++) {
        if(fabs(yaw_rate) < 0.00001){
            particles[i].x = particles[i].x + velocity * delta_t * cos(particles[i].theta); 
            particles[i].y = particles[i].y + velocity * delta_t * sin(particles[i].theta); 
        } else {
            particles[i].x = particles[i].x + velocity/yaw_rate*(sin(particles[i].theta+yaw_rate*delta_t)-sin(particles[i].theta));
            particles[i].y = particles[i].y + velocity/yaw_rate*(cos(particles[i].theta)-cos(particles[i].theta+yaw_rate*delta_t));
            particles[i].theta = particles[i].theta + yaw_rate*delta_t;
        }
        particles[i].x += nd_x(gen);
        particles[i].y += nd_y(gen);
        particles[i].theta += nd_theta(gen);        
    }

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
    for (int i = 0; i < observations.size(); i++) {
        float min_dist = std::numeric_limits<float>::max();
        for (int j = 0; j < predicted.size(); j++) {
            float d= dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);
            if (d < min_dist) {
                min_dist = d;                
                observations[i].id = predicted[j].id; 
            }
        }
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
    //return;
        
    double total_weights = 0.0;
    for (int i = 0;i < num_particles; i++) {
        Particle p = particles[i];

        if (debug) {
            std::cout << "particle[" << i << "]";           
            std::cout << " x: " << particles[i].x;           
            std::cout << " y:" << particles[i].y;           
            std::cout << " theta:" << particles[i].theta << std::endl;           
        }
        
        std::vector<LandmarkObs> observed;        
        for (int i = 0; i < observations.size(); i++) {
            LandmarkObs obs;
            obs.x = p.x + cos(p.theta)*observations[i].x - sin(p.theta)*observations[i].y;
            obs.y = p.y + sin(p.theta)*observations[i].x + cos(p.theta)*observations[i].y;

            if (debug) {            
                std::cout << "observation[" << i << "]";            
                std::cout << " observation.x:" <<  observations[i].x;
                std::cout << " observation.y:" <<  observations[i].y;
                std::cout << " tobs.x:" <<  obs.x;
                std::cout << " tobs.y:" <<  obs.y << std::endl;
            }
            observed.push_back(obs);
        }
        
        std::vector<LandmarkObs> predicted;        
        for (int i = 0; i < map_landmarks.landmark_list.size(); i++) {
            LandmarkObs landmark;
            landmark.id = map_landmarks.landmark_list[i].id_i;
            landmark.x = map_landmarks.landmark_list[i].x_f;
            landmark.y = map_landmarks.landmark_list[i].y_f;

            float d = dist(p.x, p.y, landmark.x, landmark.y);
            if (d < sensor_range)
                predicted.push_back(landmark);
        }

        if (predicted.size() == 0)
            continue;

        dataAssociation(predicted, observed);

        double total_prob = 1.0;
        double gauss_norm = (1.0/(2.0 * M_PI * std_landmark[0] * std_landmark[1]));

        for (int i = 0; i < observed.size(); i++) {
            const Map::single_landmark_s* landmark = get_landmark(observed[i].id, map_landmarks);
            if (landmark == NULL)
                std::cout << observed[i].id << ' ' << "NULL" << std::endl;

            if (debug) {
                std::cout << "observation[" << i << "]";            
                std::cout << " assosiation landmark id:" << observed[i].id;            
                std::cout << " assosiation landmark x:" << landmark->x_f;            
                std::cout << " assosiation landmark y:" << landmark->y_f << std::endl;            
            }
            
            double dist_x = (observed[i].x - landmark->x_f);
            double dist_y = (observed[i].y - landmark->y_f);
            double exponent= (dist_x*dist_x)/(2.0 * std_landmark[0]*std_landmark[0]) + (dist_y*dist_y)/(2.0 * std_landmark[1]*std_landmark[1]);
            
            double prob = gauss_norm * exp(-exponent);

            if (debug) {
                std::cout << "dist_x:" << dist_x << " dist_y:" << dist_y << " exponent:" << exponent << " norm: " << gauss_norm << std::endl;
            }            
            total_prob *= prob;
        }

        particles[i].weight = total_prob;

        if (debug) {        
            std::cout << "total_prob: " << total_prob << std::endl;
        }    
        total_weights += total_prob;
    }

    for (int i = 0;i < num_particles; i++) {
        Particle p = particles[i];
        
        particles[i].weight = p.weight/total_weights;
        if (debug) {
            std::cout << "particle[" << i << "].weight = " << particles[i].weight << std::endl;
        }
    }
}

void ParticleFilter::resample() {
    
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
    std::default_random_engine gen;
    std::vector<double> weights;
    for (int i = 0; i < particles.size(); i++) {
        Particle p = particles[i];
        weights.push_back(p.weight);
    }

    std::vector<Particle> resampled;
    std::discrete_distribution<> d(weights.begin(), weights.end());
    for (int i = 0; i < particles.size(); i++) {
        int idx = d(gen);
        resampled.push_back(particles[idx]);
    }

    particles = resampled;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
