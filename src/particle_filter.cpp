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

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	
	num_particles = 100;
	default_random_engine gen;
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);
	
	cout <<x<< "  " << y << "  " << theta << endl;
	
	for (int i = 0; i < num_particles; ++i) {
		
		Particle sample;

        sample.x = dist_x(gen);
		sample.y = dist_y(gen);
		sample.theta = dist_theta(gen);
		sample.weight = 1.0;
		// Print your samples to the terminal.
		particles.push_back(sample);
		weights.push_back(1.0);
		cout <<i<< "sample.x=" << sample.x << endl;
	}
	
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	
	default_random_engine gen;
	for (int i = 0; i < num_particles; ++i){
		Particle p = particles[i];
		double new_x;
		double new_y;
		double new_theta;
		if (fabs(yaw_rate)<0.0001){
			new_x = p.x + velocity * cos(p.theta) * delta_t;
			new_y = p.y + velocity * sin(p.theta) * delta_t;
			new_theta = p.theta;
		} else {
			new_x = p.x + velocity/yaw_rate * (sin(p.theta + yaw_rate * delta_t)-sin(p.theta));
			new_y = p.y + velocity/yaw_rate * (cos(p.theta)-cos(p.theta+yaw_rate * delta_t));
			new_theta = p.theta + yaw_rate * delta_t;
		}
		
		
		normal_distribution<double> dist_x(new_x, std_pos[0]);
		normal_distribution<double> dist_y(new_y, std_pos[1]);
		normal_distribution<double> dist_theta(new_theta, std_pos[2]);	
		
		p.x = dist_x(gen);
		p.y = dist_y(gen);
		p.theta = dist_theta(gen);
		
		cout<<i<<"p.x "<<p.x<<endl;
		particles[i] = p;
	
	}

}

void ParticleFilter::dataAssociation(Particle &p, double std_landmark[], std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	
	
	numeric_limits<double> nl;
	
	std::vector<int> associations(predicted.size());
	std::vector<double> sense_x(predicted.size());
	std::vector<double> sense_y(predicted.size());
	p.weight = 1.0;
	
	for (int i = 0; i < predicted.size(); i++){
		double min_distance = nl.max();
		LandmarkObs closest_landmark;
		for (int j = 0; j < observations.size(); j++){
			double distance = dist(observations[j].x, observations[j].y, predicted[i].x, predicted[i].y);
			if (distance < min_distance){
				min_distance = distance;
				closest_landmark = observations[j];
			}
		}
	
		
		associations[i] = closest_landmark.id;
		sense_x[i] = closest_landmark.x;
		sense_y[i] = closest_landmark.y;
		//sense_x[i] = predicted[i].x;
		//sense_y[i] = predicted[i].y;
		
		double w = multivariate_gaussian(predicted[i].x, predicted[i].y, 
			closest_landmark.x, closest_landmark.y, std_landmark[0], std_landmark[1]);
		
		p.weight *= w;
	}
	
	p.associations= associations;
    p.sense_x = sense_x;
    p.sense_y = sense_y;
	
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
	
	
	vector<LandmarkObs> map_real_landmarks(map_landmarks.landmark_list.size());
	for (int i = 0; i < map_real_landmarks.size(); i++){
		map_real_landmarks[i].id = map_landmarks.landmark_list[i].id_i;
		map_real_landmarks[i].x = map_landmarks.landmark_list[i].x_f;
		map_real_landmarks[i].y = map_landmarks.landmark_list[i].y_f;
	}
	
	
	for (int i = 0; i < particles.size(); i++){
		
		vector<LandmarkObs> particle_obs(observations.size());	
		
		for (int j = 0; j < observations.size();j++){
			double new_x = particles[i].x + (cos(particles[i].theta) * observations[j].x) - (sin(particles[i].theta) * observations[j].y);
			double new_y = particles[i].y + (sin(particles[i].theta) * observations[j].x) + (cos(particles[i].theta) * observations[j].y);
			particle_obs[j].x = new_x;
			particle_obs[j].y = new_y;
		}
		
		dataAssociation(particles[i],std_landmark, particle_obs, map_real_landmarks);
			
	}

}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	
		// First calculate the sum of all weights
	double w_sum = 0.0;
	for(unsigned int i = 0; i < particles.size(); ++i){
		w_sum += particles[i].weight;
	}	

	// Normalise our weights so that their sum cumulates to 1	
	for(unsigned int i = 0; i < particles.size(); ++i){
		Particle p = particles[i];
		p.weight = p.weight / w_sum;
		// particles[i] = p;

		weights[i] = p.weight;		
	}

	default_random_engine gen;
	discrete_distribution<unsigned int> dd(weights.begin(), weights.end());
	
	vector<Particle> survived_particles(particles.size());
	for(unsigned int i = 0; i < survived_particles.size(); ++i){
		int idx = dd(gen);		
		Particle survivor = particles[idx];
		survivor.id = i;
		survived_particles[i] = survivor;
	}

	// Our sampled (or survived particles) now become our new particles
	particles = survived_particles;

}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
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
