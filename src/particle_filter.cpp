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
#include "helper_functions.h"


#include "particle_filter.h"
using namespace std;


void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	num_particles = 20;

	default_random_engine gen;
	normal_distribution<double> N_x_init(x, std[0]);
	normal_distribution<double> N_y_init(y, std[1]);
	normal_distribution<double> N_theta_init(theta, std[2]);

	for (unsigned int i = 0; i < num_particles; ++i) {
		// initialize id, x, y, theta, and weight for each particle
        Particle p_temp;
        p_temp.id = i;
		p_temp.x = N_x_init(gen);
		p_temp.y = N_y_init(gen);
		p_temp.theta = N_theta_init(gen);
		p_temp.weight = 1.0;
        particles.push_back(p_temp);

        // initialize weights vector with normalized weights
        weights.push_back(1.0 / num_particles);
	}
    // set is_initialized to true
    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

    // Set up random engine and normal distribution
    default_random_engine gen;
    normal_distribution<double> N_x(0, std_pos[0]);
    normal_distribution<double> N_y(0, std_pos[1]);
    normal_distribution<double> N_theta(0, std_pos[2]);

	for (unsigned int i = 0; i < num_particles; ++i) {
		// Predict new position
		double x_f, y_f, theta_f;
		double x0 = particles[i].x;
		double y0 = particles[i].y;
		double theta0 = particles[i].theta;

        if (fabs(yaw_rate) < 0.001) { // yaw_rate close to zero
            x_f = x0 + velocity * delta_t * cos(theta0);
            y_f = y0 + velocity * delta_t * sin(theta0);
            theta_f = theta0;
        }
        else { // yaw rate is sufficiently large
            x_f = x0 + velocity / yaw_rate * (sin(theta0 + yaw_rate * delta_t) - sin(theta0));
            y_f = y0 + velocity / yaw_rate * (cos(theta0) - cos(theta0 + yaw_rate * delta_t));
            theta_f = theta0 + yaw_rate * delta_t;
        }

		// Add noise
		particles[i].x = x_f + N_x(gen);
		particles[i].y = y_f + N_y(gen);
		particles[i].theta = theta_f + N_theta(gen);
	}

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

    // This function is not used in the implementation.

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33. Note that you'll need to switch the minus sign in that equation to a plus to account 
	//   for the fact that the map's y-axis actually points downwards.)
	//   http://planning.cs.uiuc.edu/node99.html

    // record sum of weights for normalization
    double sum_prob = 0.0;

	// loop through all particles
	for (unsigned int i = 0; i < num_particles; ++i) {
        // initialize weight of each particle
        double prob = 1.0;

        // get particle position and yaw angle
        double x_p = particles[i].x;
        double y_p = particles[i].y;
        double theta = particles[i].theta;

		// select landmarks within sensor_range
		vector<Map::single_landmark_s> predicted_landmarks;
		for (unsigned int j = 0; j < map_landmarks.landmark_list.size(); ++j) {
            double x_landmark = map_landmarks.landmark_list[j].x_f;
            double y_landmark = map_landmarks.landmark_list[j].y_f;
            if (dist(x_landmark, y_landmark, x_p, y_p) <= sensor_range)
                predicted_landmarks.push_back(map_landmarks.landmark_list[j]);
        }

        // loop through all observations
        for (unsigned int j = 0; j < observations.size(); ++j) {
            // convert local coordinates to global coordinates
            double x_local = observations[j].x;
            double y_local = observations[j].y;
            double x_global = x_local * cos(theta) - y_local * sin(theta) + x_p;
            double y_global = x_local * sin(theta) + y_local * cos(theta) + y_p;

            // loop through predicted_landmarks and find the closest one
            double min_dist = sensor_range;
            double x_diff = sensor_range;
            double y_diff = sensor_range;
            for (unsigned int k = 0; k < predicted_landmarks.size(); ++k) {
                double x_landmark = predicted_landmarks[k].x_f;
                double y_landmark = predicted_landmarks[k].y_f;
                if (dist(x_landmark, y_landmark, x_global, y_global) < min_dist) {
                    min_dist = dist(x_landmark, y_landmark, x_global, y_global);
                    x_diff = x_landmark - x_global;
                    y_diff = y_landmark - y_global;
                }
            }

            // update probability wrt each observation
            prob *= (1 / (2 * M_PI * std_landmark[0] * std_landmark[1])) *
                    exp (-0.5 * ((x_diff * x_diff / (std_landmark[0] * std_landmark[0])) +
                                 (y_diff * y_diff / (std_landmark[1] * std_landmark[1]))));
        }
        // update un-normalized weight to each particle
        particles[i].weight = prob;
        // calculate sum_prob to normalize the weights in the weights vector
        sum_prob += prob;
	}

    // update the weights vector with normalized weights
    for (unsigned int i = 0; i < num_particles; ++i) weights[i] = particles[i].weight / sum_prob;
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
    vector<Particle> new_particles;
    default_random_engine gen;
    discrete_distribution<int> distribution(weights.begin(), weights.end());
    for (unsigned int i = 0; i < num_particles; ++i) {
        int weighted_index = distribution(gen);
        new_particles.push_back(particles[weighted_index]);
    }
    particles = new_particles;
}

void ParticleFilter::write(std::string filename) {
	// You don't need to modify this file.
	std::ofstream dataFile;
	dataFile.open(filename, std::ios::app);
	for (int i = 0; i < num_particles; ++i) {
		dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << "\n";
	}
	dataFile.close();
}
