#include "LogisticRegression.h"
#include "../DataUtils/DataLoader.h"
#include "../Evaluation/Metrics.h"
#include "../DataUtils/DataPreprocessor.h"
#include <string>
#include <vector>
#include <utility>
#include <set>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <random>
#include <unordered_map>
#include <numeric>      // iota
#include <stdexcept>    // invalid_argument

using namespace System::Windows::Forms; // For MessageBox


///  LogisticRegression class implementation  ///

// Constructor
LogisticRegression::LogisticRegression(double learning_rate, int num_epochs)
    : learning_rate(learning_rate), num_epochs(num_epochs) {
}

// Fit method for training the logistic regression model
void LogisticRegression::fit(const std::vector<std::vector<double>>& X_train, const std::vector<double>& y_train) {
    // Validations
    if (X_train.empty() || y_train.empty()) {
        throw std::invalid_argument("fit: X_train or y_train is empty.");
    }
    if (X_train.size() != y_train.size()) {
        throw std::invalid_argument("fit: X_train and y_train must have the same number of samples.");
    }

    int num_samples = static_cast<int>(X_train.size());
    int num_features = static_cast<int>(X_train[0].size());

    // Check consistent feature sizes
    for (size_t i = 0; i < X_train.size(); ++i) {
        if (static_cast<int>(X_train[i].size()) != num_features) {
            throw std::invalid_argument("fit: inconsistent feature vector sizes in X_train.");
        }
    }

    // Number of classes inferred from labels (assumes labels are already 0..C-1)
    std::set<double> unique_labels(y_train.begin(), y_train.end());
    int num_classes = static_cast<int>(unique_labels.size());
    if (num_classes < 2) {
        throw std::invalid_argument("fit: need at least two classes to train.");
    }

    // Initialize weights: num_classes x (num_features + 1) (bias included)
    weights.clear();
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<double> dis(-0.01, 0.01);

    weights.resize(num_classes, std::vector<double>(num_features + 1, 0.0));
    for (int c = 0; c < num_classes; ++c) 
    {
        for (int j = 0; j < num_features + 1; ++j) 
        {
            weights[c][j] = dis(gen);
        }
    }

    // SGD with shuffling per epoch
    std::mt19937 shuffle_gen(std::random_device{}());
    std::vector<int> indices(num_samples);
    std::iota(indices.begin(), indices.end(), 0);

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        std::shuffle(indices.begin(), indices.end(), shuffle_gen);

        for (int idx : indices) {
            // add bias term at front
            std::vector<double> x_with_bias;
            x_with_bias.reserve(num_features + 1);
            x_with_bias.push_back(1.0); // bias
            x_with_bias.insert(x_with_bias.end(), X_train[idx].begin(), X_train[idx].end());

            // compute logits (linear scores) for each class
            std::vector<double> logits(num_classes, 0.0);
            for (int c = 0; c < num_classes; ++c) {
                double s = 0.0;
                for (size_t j = 0; j < x_with_bias.size(); ++j) {
                    s += weights[c][j] * x_with_bias[j];
                }
                logits[c] = s;
            }

            // softmax (stable)
            double max_logit = *std::max_element(logits.begin(), logits.end());
            double sum_exp = 0.0;
            for (int c = 0; c < num_classes; ++c) {
                logits[c] = std::exp(logits[c] - max_logit);
                sum_exp += logits[c];
            }
            for (int c = 0; c < num_classes; ++c) {
                logits[c] /= sum_exp; // probability for class c
            }

            // gradient step: for each class c, error = p_c - y_c
            int true_c = static_cast<int>(y_train[idx]); // must be 0..num_classes-1
            for (int c = 0; c < num_classes; ++c) {
                double target = (c == true_c) ? 1.0 : 0.0;
                double error = logits[c] - target; // (p - y)
                for (size_t j = 0; j < x_with_bias.size(); ++j) {
                    weights[c][j] -= learning_rate * error * x_with_bias[j];
                }
            }
        } // end samples
    } // end epochs
}

// Predict method to predict class labels for test data
std::vector<double> LogisticRegression::predict(const std::vector<std::vector<double>>& X_test) {
    std::vector<double> predictions;

    if (X_test.empty()) {
        return predictions; // empty result
    }
    if (weights.empty()) {
        throw std::runtime_error("predict: model weights are empty (train the model first).");
    }

    int num_classes = static_cast<int>(weights.size());
    int expected_feature_count = static_cast<int>(weights[0].size()) - 1;

    for (const auto& x : X_test) {
        if (static_cast<int>(x.size()) != expected_feature_count) {
            throw std::invalid_argument("predict: test sample has different number of features than trained model.");
        }

        // add bias term
        std::vector<double> x_with_bias;
        x_with_bias.reserve(expected_feature_count + 1);
        x_with_bias.push_back(1.0);
        x_with_bias.insert(x_with_bias.end(), x.begin(), x.end());

        // compute logits
        std::vector<double> logits(num_classes, 0.0);
        for (int c = 0; c < num_classes; ++c) {
            double s = 0.0;
            for (size_t j = 0; j < x_with_bias.size(); ++j) {
                s += weights[c][j] * x_with_bias[j];
            }
            logits[c] = s;
        }

        // softmax (we only need argmax; softmax for numerical stability still applied)
        double max_logit = *std::max_element(logits.begin(), logits.end());
        double sum_exp = 0.0;
        for (int c = 0; c < num_classes; ++c) {
            logits[c] = std::exp(logits[c] - max_logit);
            sum_exp += logits[c];
        }
        for (int c = 0; c < num_classes; ++c) {
            logits[c] /= sum_exp;
        }

        int best_idx = static_cast<int>(std::distance(logits.begin(), std::max_element(logits.begin(), logits.end())));
        predictions.push_back(static_cast<double>(best_idx)); // return index (0..C-1)
    }

    return predictions;
}

/// runLogisticRegression: this function runs the logistic regression algorithm on the given dataset and
/// then returns a tuple containing the evaluation metrics for the training and test sets,
/// as well as the labels and predictions for the training and test sets.///
std::tuple<double, double, std::vector<double>, std::vector<double>, std::vector<double>, std::vector<double>>
LogisticRegression::runLogisticRegression(const std::string& filePath, int trainingRatio) {

    DataPreprocessor DataPreprocessor;
    try {
        if (filePath.empty()) {
            MessageBox::Show("Please browse and select the dataset file from your PC.");
            return std::make_tuple(0.0, 0.0, std::vector<double>(), std::vector<double>(), std::vector<double>(), std::vector<double>());
        }

        std::ifstream file(filePath);
        if (!file.is_open()) {
            MessageBox::Show("Failed to open the dataset file");
            return std::make_tuple(0.0, 0.0, std::vector<double>(), std::vector<double>(), std::vector<double>(), std::vector<double>());
        }

        std::vector<std::vector<double>> dataset;
        DataLoader::loadAndPreprocessDataset(filePath, dataset);

        if (dataset.empty()) {
            MessageBox::Show("Dataset loaded but it's empty.");
            return std::make_tuple(0.0, 0.0, std::vector<double>(), std::vector<double>(), std::vector<double>(), std::vector<double>());
        }

        double trainRatio = trainingRatio * 0.01;

        std::vector<std::vector<double>> trainData;
        std::vector<double> trainLabels;
        std::vector<std::vector<double>> testData;
        std::vector<double> testLabels;

        DataPreprocessor::splitDataset(dataset, trainRatio, trainData, trainLabels, testData, testLabels);

        if (trainData.empty() || trainLabels.empty()) {
            MessageBox::Show("Training set is empty after splitting. Check the trainingRatio or dataset.");
            return std::make_tuple(0.0, 0.0, std::vector<double>(), std::vector<double>(), std::vector<double>(), std::vector<double>());
        }
        if (testData.empty() || testLabels.empty()) {
            MessageBox::Show("Test set is empty after splitting. Check the trainingRatio or dataset.");
            return std::make_tuple(0.0, 0.0, std::vector<double>(), std::vector<double>(), std::vector<double>(), std::vector<double>());
        }

        // Map labels to 0..C-1 by subtracting the minimum label (works if labels are e.g. 1,2,3)
        double min_label = *std::min_element(trainLabels.begin(), trainLabels.end());
        // build zero-based copies for training
        std::vector<double> trainLabelsZero(trainLabels.size());
        for (size_t i = 0; i < trainLabels.size(); ++i) {
            trainLabelsZero[i] = trainLabels[i] - min_label;
        }

        // Train model (labels are now 0..C-1)
        fit(trainData, trainLabelsZero);

        // Predict on test data: returns predicted indices 0..C-1
        std::vector<double> testPredictionsZero = predict(testData);
        // Map predictions back to original label values
        std::vector<double> testPredictions(testPredictionsZero.size());
        for (size_t i = 0; i < testPredictionsZero.size(); ++i) {
            testPredictions[i] = testPredictionsZero[i] + min_label;
        }

        // Compute test accuracy comparing original testLabels with mapped predictions
        double test_accuracy = Metrics::accuracy(testLabels, testPredictions);

        // Predict on training data and map back
        std::vector<double> trainPredictionsZero = predict(trainData);
        std::vector<double> trainPredictions(trainPredictionsZero.size());
        for (size_t i = 0; i < trainPredictionsZero.size(); ++i) {
            trainPredictions[i] = trainPredictionsZero[i] + min_label;
        }
        double train_accuracy = Metrics::accuracy(trainLabels, trainPredictions);

        MessageBox::Show("Run completed");
        return std::make_tuple(train_accuracy, test_accuracy,
            std::move(trainLabels), std::move(trainPredictions),
            std::move(testLabels), std::move(testPredictions));
    }
    catch (const std::exception& e) {
        // Show a helpful message to the user and the console for debugging
        try {
            MessageBox::Show(gcnew System::String(e.what()));
        }
        catch (...) {
            MessageBox::Show("An exception occurred (and its message could not be shown).");
        }
        std::cerr << "Exception occurred: " << e.what() << std::endl;
        return std::make_tuple(0.0, 0.0, std::vector<double>(),
            std::vector<double>(), std::vector<double>(),
            std::vector<double>());
    }
}