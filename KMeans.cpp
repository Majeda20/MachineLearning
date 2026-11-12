#include "KMeans.h"
#include "../DataUtils/DataLoader.h"
#include "../Utils/SimilarityFunctions.h"
#include "../Evaluation/Metrics.h"
#include "../DataUtils/DataPreprocessor.h"
#include "../Utils/PCADimensionalityReduction.h"
#include <string>
#include <vector>
#include <utility>
#include <cmath>
#include <algorithm>
#include <limits>
#include <random> 
#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <unordered_map> 
using namespace System::Windows::Forms; // For MessageBox


///  KMeans class implementation  ///

// KMeans function: Constructor for KMeans class.//
KMeans::KMeans(int numClusters, int maxIterations)
	: numClusters_(numClusters), maxIterations_(maxIterations) {}


// fit function: Performs K-means clustering on the given dataset and return the centroids of the clusters.//
void KMeans::fit(const std::vector<std::vector<double>>& data) {
	// Create a copy of the data to preserve the original dataset
	std::vector<std::vector<double>> normalizedData = data;

	/* Implement the following:
		---	Initialize centroids randomly
		--- Randomly select unique centroid indices
		---	Perform K-means clustering
		--- Assign data points to the nearest centroid
		--- Calculate the Euclidean distance between the point and the current centroid
		--- Update newCentroids and clusterCounts
		--- Update centroids
		---  Check for convergence
	*/
	
	// TODO


	// Initialize centroids randomly
	srand(time(NULL));
	std::vector<std::vector<double>> centroids;

	// Randomly select unique centroid indices
	for (int i = 0; i < numClusters_; i++) {
		int rnd = rand() % normalizedData.size();
		centroids.push_back(normalizedData[rnd]);
	}
	std::vector<int> labels(data.size(), -1);
	bool has_converged = false;
	int itr = 0;

	//Perform K-means clustering
	while (!has_converged && itr < maxIterations_) {
		has_converged = true;
		std::vector<std::vector<std::vector<double>>> clusters(numClusters_);

		// Assign data points to the nearest centroid
		for (int i = 0; i < normalizedData.size(); i++) {
			int closest_centroid = -1;
			double min_dist = std::numeric_limits<double>::infinity();

			// Calculate the Euclidean distance between the point and the current centroid
			for (int j = 0; j < numClusters_; j++) {
				double dist = SimilarityFunctions::euclideanDistance(normalizedData[i], centroids[j]);

				if (dist < min_dist) {
					min_dist = dist;
					closest_centroid = j;
				}
			}
			if (labels[i] != closest_centroid) {
				labels[i] = closest_centroid;
				has_converged = false;
			}
			clusters[closest_centroid].push_back(normalizedData[i]);
		}

		// Update newCentroids and clusterCounts
		for (int i = 0; i < numClusters_; i++) {
			if (!clusters[i].empty()) {
				int num_points = clusters[i].size();
				int num_features = clusters[i][0].size();
				std::vector<double> mean(num_features, 0.0);

				for (const auto& point : clusters[i]) {
					for (int f = 0; f < num_features; f++) {
						mean[f] += point[f];
					}
				}
				for (int f = 0; f < num_features; f++) {
					mean[f] /= num_points;
				}

				//Update centroids
				centroids[i] = mean;
			}
		}

		// Check for convergence
		itr++;
		centroids_ = centroids;
	}
}


//// predict function: Calculates the closest centroid for each point in the given data set and returns the labels of the closest centroids.//
std::vector<int> KMeans::predict(const std::vector<std::vector<double>>& data) const {
	std::vector<int> labels;
	labels.reserve(data.size());
	
	/* Implement the following:
		--- Initialize the closest centroid and minimum distance to the maximum possible value
		--- Iterate through each centroid
		--- Calculate the Euclidean distance between the point and the centroid
		--- Add the closest centroid to the labels vector
    */
	
	// TODO

	for (const auto& point : data) {
		//  Initialize the closest centroid and minimum distance to the maximum possible value
		int closest_centroid = -1;
		double min_dist = std::numeric_limits<double>::infinity();

		// Iterate through each centroid
		for (int i = 0; i < centroids_.size(); i++) {
			//  Calculate the Euclidean distance between the point and the centroid
			double dist = SimilarityFunctions::euclideanDistance(point, centroids_[i]);

			if (dist < min_dist) {
				min_dist = dist;
				closest_centroid = i;
			}
		}

		//  Add the closest centroid to the labels vector
		labels.push_back(closest_centroid + 1);
	}
	return labels; // Return the labels vector

}





/// runKMeans: this function runs the KMeans clustering algorithm on the given dataset and 
/// then returns a tuple containing the evaluation metrics for the training and test sets, 
/// as well as the labels and predictions for the training and test sets.///
std::tuple<double, double, std::vector<int>, std::vector<std::vector<double>>>
KMeans::runKMeans(const std::string& filePath) {
	DataPreprocessor DataPreprocessor;
	try {
		// Check if the file path is empty
		if (filePath.empty()) {
			MessageBox::Show("Please browse and select the dataset file from your PC.");
			return {}; // Return an empty vector since there's no valid file path
		}

		// Attempt to open the file
		std::ifstream file(filePath);
		if (!file.is_open()) {
			MessageBox::Show("Failed to open the dataset file");
			return {}; // Return an empty vector since file couldn't be opened
		}

		std::vector<std::vector<double>> dataset; // Create an empty dataset vector
		DataLoader::loadAndPreprocessDataset(filePath, dataset);

		// Use the all dataset for training and testing sets.
		double trainRatio = 1.0;

		std::vector<std::vector<double>> trainData;
		std::vector<double> trainLabels;
		std::vector<std::vector<double>> testData;
		std::vector<double> testLabels;

		DataPreprocessor::splitDataset(dataset, trainRatio, trainData, trainLabels, testData, testLabels);

		// Fit the model to the training data
		fit(trainData);

		// Make predictions on the training data
		std::vector<int> labels = predict(trainData);

		// Calculate evaluation metrics
		// Calculate Davies BouldinIndex using the actual features and predicted cluster labels
		double daviesBouldinIndex = Metrics::calculateDaviesBouldinIndex(trainData, labels);

		// Calculate Silhouette Score using the actual features and predicted cluster labels
		double silhouetteScore = Metrics::calculateSilhouetteScore(trainData, labels);

		// Create an instance of the PCADimensionalityReduction class
		PCADimensionalityReduction pca;

		// Perform PCA and project the data onto a lower-dimensional space
		int num_dimensions = 2; // Number of dimensions to project onto
		std::vector<std::vector<double>> reduced_data = pca.performPCA(trainData, num_dimensions);

		MessageBox::Show("Run completed");
		return std::make_tuple(daviesBouldinIndex, silhouetteScore, std::move(labels), std::move(reduced_data));
	}
	catch (const std::exception& e) {
		// Handle the exception
		MessageBox::Show("Not Working");
		std::cerr << "Exception occurred: " << e.what() << std::endl;
		return std::make_tuple(0.0, 0.0, std::vector<int>(), std::vector<std::vector<double>>());
	}
}