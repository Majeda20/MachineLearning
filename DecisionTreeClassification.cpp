#include "DecisionTreeClassification.h"
#include "../DataUtils/DataLoader.h"
#include "../Evaluation/Metrics.h"
#include "../Utils/EntropyFunctions.h"
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <iostream>
#include <unordered_map>
#include <set>
#include <unordered_set>
#include <utility>
#include <fstream>
#include <sstream>
#include <map>
#include <random>
#include <limits>
#include "../DataUtils/DataPreprocessor.h"
using namespace System::Windows::Forms; // For MessageBox

// Add these lines to avoid Windows min/max macros
#undef min
#undef max

// DecisionTreeClassification class implementation //

// DecisionTreeClassification is a constructor for DecisionTree class.//
DecisionTreeClassification::DecisionTreeClassification(int min_samples_split, int max_depth, int n_feats)
    : min_samples_split(min_samples_split), max_depth(max_depth), n_feats(n_feats), root(nullptr) {
}

// Fit is a function to fits a decision tree to the given data.//
void DecisionTreeClassification::fit(std::vector<std::vector<double>>& X, std::vector<double>& y) {
    n_feats = (n_feats == 0) ? X[0].size() : (std::min)(n_feats, static_cast<int>(X[0].size()));
    root = growTree(X, y);
}

// Predict is a function that Traverses the decision tree and returns the prediction for a given input vector.//
std::vector<double> DecisionTreeClassification::predict(std::vector<std::vector<double>>& X) {
    std::vector<double> predictions;

    // For each input sample, traverse the tree to get prediction
    for (auto& sample : X) {
        predictions.push_back(traverseTree(sample, root));
    }

    return predictions;
}

// growTree function: This function grows a decision tree using the given data and labelsand  return a pointer to the root node of the decision tree.//
Node* DecisionTreeClassification::growTree(std::vector<std::vector<double>>& X, std::vector<double>& y, int depth) {

    /* Implement the following:
        --- define stopping criteria
        --- Loop through candidate features and potential split thresholds.
        --- greedily select the best split according to information gain
        ---grow the children that result from the split
    */

    // Get the number of samples and number of features
    int n_samples = X.size();
    int n_features = X.empty() ? 0 : X[0].size();

    // Calculate current entropy for stopping criteria
    double current_entropy = EntropyFunctions::entropy(y);

    // Stopping criteria: check if we should stop growing the tree
    if (depth >= max_depth || n_samples < min_samples_split ||
        current_entropy < 0.05) 
    { // Menos estricto: 0.05 en lugar de 0.01
        // Create a leaf node with the most common label
        double leaf_value = mostCommonlLabel(y);
        return new Node(-1, -1.0, nullptr, nullptr, leaf_value);
    }

    // Check if all labels are the same (pure node)
    bool all_same = true;
    double first_label = 0.0;
    if (!y.empty()) {
        first_label = y[0];
        for (double label : y) {
            if (label != first_label) {
                all_same = false;
                break;
            }
        }
    }
    else {
        all_same = false;
    }

    if (all_same) {
        return new Node(-1, -1.0, nullptr, nullptr, first_label);
    }

    double best_gain = -1.0; // set the best gain to -1
    int best_feature = -1; // split index
    double best_threshold = 0.0; // split threshold

    // Randomly select features to consider for splitting
    std::vector<int> feature_indices(n_features);
    for (int i = 0; i < n_features; i++) {
        feature_indices[i] = i;
    }

    // Shuffle and select n_feats features
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(feature_indices.begin(), feature_indices.end(), g);
    int n_features_to_try = (std::min)(n_feats, static_cast<int>(feature_indices.size()));

    // Try each selected feature
    for (int i = 0; i < n_features_to_try; i++) {
        int feat_idx = feature_indices[i];
        // Extract the feature column
        std::vector<double> feature_values;
        for (auto& row : X) {
            feature_values.push_back(row[feat_idx]);
        }

        // Get unique values as potential thresholds
        std::set<double> unique_vals(feature_values.begin(), feature_values.end());

        // Balanced threshold selection
        std::vector<double> candidate_thresholds;

        if (unique_vals.size() > 8) 
        {
            // Use percentiles for larger sets
            auto it_median = unique_vals.begin();
            std::advance(it_median, unique_vals.size() / 2);
            candidate_thresholds.push_back(*it_median);

            auto it_quartile1 = unique_vals.begin();
            std::advance(it_quartile1 , unique_vals.size() / 4);
            candidate_thresholds.push_back(*it_quartile1);

            auto it_quartile3 = unique_vals.begin();
            std::advance(it_quartile3, 3 * unique_vals.size() / 4);
            candidate_thresholds.push_back(*it_quartile3);
        }
        else 
        {
            // If few unique values, use all
            candidate_thresholds.assign(unique_vals.begin(), unique_vals.end());
        }

        // Loop through candidate split thresholds
        for (double threshold : candidate_thresholds) {
            // Calculate information gain for this split
            double gain = informationGain(y, feature_values, threshold);

            // Update best gain if this split is better
            if (gain > best_gain) {
                best_gain = gain;
                best_feature = feat_idx;
                best_threshold = threshold;
            }
        }
    }

    // If no split improves information gain or gain is too small, create a leaf node
    if (best_gain <= 0.03 || best_feature == -1) { // Balanced: 0.03
        double leaf_value = mostCommonlLabel(y);
        return new Node(-1, -1.0, nullptr, nullptr, leaf_value);
    }

    // Split the data based on the best split found
    std::vector<std::vector<double>> X_left, X_right;
    std::vector<double> y_left, y_right;

    for (int i = 0; i < n_samples; i++) {
        if (X[i][best_feature] <= best_threshold) {
            X_left.push_back(X[i]);
            y_left.push_back(y[i]);
        }
        else {
            X_right.push_back(X[i]);
            y_right.push_back(y[i]);
        }
    }

    // Check if split creates nodes that are too small
    if (X_left.size() < 2 || X_right.size() < 2) { // Menos estricto: 2 en lugar de 3
        double leaf_value = mostCommonlLabel(y);
        return new Node(-1, -1.0, nullptr, nullptr, leaf_value);
    }

    // Recursively grow children
    Node* left_child = growTree(X_left, y_left, depth + 1);
    Node* right_child = growTree(X_right, y_right, depth + 1);

    return new Node(best_feature, best_threshold, left_child, right_child, -1.0);
}

/// informationGain function: Calculates the information gain of a given split threshold for a given feature column.
double DecisionTreeClassification::informationGain(std::vector<double>& y, std::vector<double>& X_column, double split_thresh) {
    // parent loss // You need to caculate entropy using the EntropyFunctions class//
    double parent_entropy = EntropyFunctions::entropy(y);

    /* Implement the following:
       --- generate split
       --- compute the weighted avg. of the loss for the children
       --- information gain is difference in loss before vs. after split
    */
    // Split the labels based on the threshold
    std::vector<double> left_labels, right_labels;

    for (int i = 0; i < X_column.size(); i++) {
        if (X_column[i] <= split_thresh) {
            left_labels.push_back(y[i]);
        }
        else {
            right_labels.push_back(y[i]);
        }
    }

    // If either child is empty, return 0 gain
    if (left_labels.empty() || right_labels.empty()) {
        return 0.0;
    }

    // Calculate weighted average of child entropies
    double n_total = y.size();
    double n_left = left_labels.size();
    double n_right = right_labels.size();

    double left_entropy = EntropyFunctions::entropy(left_labels);
    double right_entropy = EntropyFunctions::entropy(right_labels);

    double weighted_entropy = (n_left / n_total) * left_entropy + (n_right / n_total) * right_entropy;

    // Information gain is difference between parent entropy and weighted child entropy
    return parent_entropy - weighted_entropy;
}

// mostCommonlLabel function: Finds the most common label in a vector of labels.//
double DecisionTreeClassification::mostCommonlLabel(std::vector<double>& y) {
    if (y.empty()) {
        return 0.0;
    }

    // Count frequency of each label
    std::unordered_map<double, int> count_map;

    for (double label : y) {
        count_map[label]++;
    }

    // Find the label with maximum frequency
    double most_common = y[0];
    int max_count = 0;

    for (auto& pair : count_map) {
        if (pair.second > max_count) {
            max_count = pair.second;
            most_common = pair.first;
        }
    }

    return most_common;
}

// traverseTree function: Traverses a decision tree given an input vector and a node.//
double DecisionTreeClassification::traverseTree(std::vector<double>& x, Node* node) {

    /* Implement the following:
        --- If the node is a leaf node, return its value
        --- If the feature value of the input vector is less than or equal to the node's threshold, traverse the left subtree
        --- Otherwise, traverse the right subtree
    */

    // If the node is a leaf node, return its value
    if (node->isLeafNode()) {
        return node->value;
    }

    // If the feature value of the input vector is less than or equal to the node's threshold,
    // traverse the left subtree, otherwise traverse the right subtree
    if (x[node->feature] <= node->threshold) {
        return traverseTree(x, node->left);
    }
    else {
        return traverseTree(x, node->right);
    }
}

/// runDecisionTreeClassification: this function runs the decision tree classification algorithm on the given dataset and
/// then returns a tuple containing the evaluation metrics for the training and test sets,
/// as well as the labels and predictions for the training and test sets.///
std::tuple<double, double, std::vector<double>, std::vector<double>, std::vector<double>, std::vector<double>>
DecisionTreeClassification::runDecisionTreeClassification(const std::string& filePath, int trainingRatio) {
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

        // Split the dataset into training and test sets (e.g., 80% for training, 20% for testing)
        double trainRatio = trainingRatio * 0.01;

        std::vector<std::vector<double>> trainData;
        std::vector<double> trainLabels;
        std::vector<std::vector<double>> testData;
        std::vector<double> testLabels;

        DataPreprocessor::splitDataset(dataset, trainRatio, trainData, trainLabels, testData, testLabels);

        // Use balanced parameters
        min_samples_split = 9;    // Balanced
        max_depth = 500;           // Balanced depth
        n_feats = trainData[0].size(); // Usar todas las características

        // Fit the model to the training data
        fit(trainData, trainLabels);//

        // Make predictions on the test data
        std::vector<double> testPredictions = predict(testData);

        // Calculate accuracy using the true labels and predicted labels for the test data
        double test_accuracy = Metrics::accuracy(testLabels, testPredictions);

        // Make predictions on the training data
        std::vector<double> trainPredictions = predict(trainData);

        // Calculate accuracy using the true labels and predicted labels for the training data
        double train_accuracy = Metrics::accuracy(trainLabels, trainPredictions);

        MessageBox::Show("Run completed");
        return std::make_tuple(train_accuracy, test_accuracy,
            std::move(trainLabels), std::move(trainPredictions),
            std::move(testLabels), std::move(testPredictions));
    }
    catch (const std::exception& e) {
        // Handle the exception
        MessageBox::Show("Not Working");
        std::cerr << "Exception occurred: " << e.what() << std::endl;
        return std::make_tuple(0.0, 0.0, std::vector<double>(),
            std::vector<double>(), std::vector<double>(),
            std::vector<double>());
    }
}