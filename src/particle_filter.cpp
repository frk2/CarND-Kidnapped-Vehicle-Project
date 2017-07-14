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
#include <random>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	std::default_random_engine gen;
	std::normal_distribution<double> dist_x(x, std[0]);
	std::normal_distribution<double> dist_y(y, std[1]);
	std::normal_distribution<double> dist_theta(theta, std[2]);
	Particle part;
	for (int i = 0; i < 1000; i++) {
		part.id = i++;
		part.x = dist_x(gen);
		part.y = dist_y(gen);
		part.theta = dist_theta(gen);
		part.weight = 1;
		particles.push_back(part);
	}
	is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	std::default_random_engine gen;

	for (auto & part : particles) {
		double v_part = velocity;
		double yaw_part = yaw_rate;
		if (fabs(yaw_part) > 0.00001) {
			part.x += v_part / yaw_part * (sin(part.theta + yaw_part*delta_t) - sin(part.theta));
			part.y += v_part / yaw_part * (cos(part.theta) - cos(part.theta + yaw_part * delta_t));
			part.theta += yaw_part * delta_t;
		} else {
			part.x += v_part * delta_t * cos(part.theta);
			part.y += v_part * delta_t * sin(part.theta);
		}
		std::normal_distribution<double> dist_x(part.x, std_pos[0]);
		std::normal_distribution<double> dist_y(part.y, std_pos[1]);
		std::normal_distribution<double> dist_theta(part.theta, std_pos[2]);
		part.x = dist_x(gen);
		part.y = dist_y(gen);
		part.theta = dist_theta(gen);

	}

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	LandmarkObs closest = predicted.front();
	double least_distance = INT32_MAX;
	for (auto & obs : observations) {
		closest = predicted.front();
		least_distance = INT32_MAX;
		for (auto pos : predicted) {
			double distance = sqrt(pow(pos.x - obs.x,2) + pow(pos.y - obs.y,2));
			if (distance < least_distance) {
				least_distance = distance;
				closest = pos;
			}
		}
		obs.id = closest.id;

	}
}


LandmarkObs ParticleFilter::translateCoords(LandmarkObs obs, Particle particle) {
	double xprime, yprime;
	xprime = obs.x * cos(particle.theta) - obs.y * sin(particle.theta);
	yprime = obs.x * sin(particle.theta) + obs.y * cos(particle.theta);
	obs.x = particle.x + xprime;
	obs.y	= particle.y + yprime;
	return obs;
}
void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	//filter out stuff
	double total_prob = 0;
	for (auto & part: particles) {
		std::vector<LandmarkObs> possible;
		std::vector<LandmarkObs> translated_obs;
		for (auto obs : observations) {
			translated_obs.push_back(translateCoords(obs, part));
		}


		for (auto lm : map_landmarks.landmark_list) {
			if (sqrt(pow(lm.x_f - part.x, 2) + pow(lm.y_f - part.y,2)) < sensor_range) {
				LandmarkObs plm;
				plm.id = lm.id_i;
				plm.x = lm.x_f;
				plm.y = lm.y_f;
				possible.push_back(plm);
			}
		}
		dataAssociation(possible, translated_obs);
		part.associations.clear();
		part.sense_x.clear();
		part.sense_y.clear();

		double weight = 1.0;
		for (auto obs : translated_obs) {
			// Guassian MV Prob
			double x_dist = ( pow(obs.x - map_landmarks.landmark_list[obs.id-1].x_f,2)) / (2 * pow(std_landmark[0],2));
			double y_dist = ( pow(obs.y - map_landmarks.landmark_list[obs.id-1].y_f,2)) / (2 * pow(std_landmark[1],2));
			weight *= exp(-(x_dist + y_dist)) / (2 * M_PI * std_landmark[0] * std_landmark[1]);
			if (weight == 0) {
				std::cout << "Weight went to zero" <<std::endl;
			}
			part.associations.push_back(obs.id);
			part.sense_x.push_back(obs.x);
			part.sense_y.push_back(obs.y);
		}
		part.weight = weight;
		total_prob += part.weight;
	}
//
//	for (auto & part : particles) {
//		part.weight /= total_prob;
//	}

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
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	std::default_random_engine gen;

	std::vector<double> all_weights;
	for (auto part : particles) {
		all_weights.push_back(part.weight);
	}

	std::discrete_distribution<> dist(all_weights.begin(), all_weights.end());
	std::vector<Particle> new_particles;
	for (int i = 0; i < particles.size(); i++) {
		new_particles.push_back(particles[dist(gen)]);
	}

	particles = new_particles;
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
