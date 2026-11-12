#include "DecisionTreeRegression.h"
#include "../DataUtils/DataLoader.h"
#include "../Evaluation/Metrics.h"
#include "../DataUtils/DataPreprocessor.h"
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <iostream>
#include <unordered_map>
#include <set>
#include <numeric>
#include <unordered_set>
#include <random>
#include <limits>
using namespace System::Windows::Forms; // For MessageBox

#undef min
#undef max

// Constructor
DecisionTreeRegression::DecisionTreeRegression(int min_samples_split, int max_depth, int n_feats)
    : min_samples_split(min_samples_split), max_depth(max_depth), n_feats(n_feats), root(nullptr) {
}

// fit
void DecisionTreeRegression::fit(std::vector<std::vector<double>>& X, std::vector<double>& y) {
    n_feats = (n_feats == 0) ? X[0].size() : (std::min)(n_feats, static_cast<int>(X[0].size()));
    root = growTree(X, y, 0);
}

// predict
std::vector<double> DecisionTreeRegression::predict(std::vector<std::vector<double>>& X) {
    std::vector<double> predictions;
    predictions.reserve(X.size());
    for (auto& sample : X)
        predictions.push_back(traverseTree(sample, root));
    return predictions;
}

// growTree
Node* DecisionTreeRegression::growTree(std::vector<std::vector<double>>& X, std::vector<double>& y, int depth) {
    int n_samples = X.size();
    int n_features = X[0].size();

    // stop por profundidad o pocas muestras
    if (depth >= max_depth || n_samples < min_samples_split)
        return new Node(-1, -1.0, nullptr, nullptr, mean(y));

    // stop si todos los valores iguales
    bool all_same = std::all_of(y.begin(), y.end(), [&](double v) { return std::abs(v - y[0]) < 1e-6; });
    if (all_same) return new Node(-1, -1.0, nullptr, nullptr, y[0]);

    // stop si varianza demasiado baja
    double y_mean = mean(y);
    double y_var = 0.0;
    for (double val : y) y_var += (val - y_mean) * (val - y_mean);
    y_var /= y.size();
    if (y_var < 1e-6) return new Node(-1, -1.0, nullptr, nullptr, y_mean);

    double best_mse = std::numeric_limits<double>::max();
    int best_feature = -1;
    double best_threshold = 0.0;

    std::vector<int> feature_indices(n_features);
    std::iota(feature_indices.begin(), feature_indices.end(), 0);
    std::random_device rd; std::mt19937 g(rd());
    std::shuffle(feature_indices.begin(), feature_indices.end(), g);
    int n_features_to_try = (std::min)(n_feats, static_cast<int>(feature_indices.size()));

    for (int i = 0; i < n_features_to_try; i++) {
        int feat_idx = feature_indices[i];
        std::vector<double> feature_values(n_samples);
        for (int j = 0; j < n_samples; j++) feature_values[j] = X[j][feat_idx];

        // obtener puntos únicos
        std::set<double> unique_vals(feature_values.begin(), feature_values.end());
        std::vector<double> thresholds(unique_vals.begin(), unique_vals.end());
        if (thresholds.size() > 20) {
            // sample 20 thresholds
            std::vector<double> sampled;
            double step = static_cast<double>(thresholds.size()) / 20.0;
            for (int k = 1; k < 20; k++)
                sampled.push_back(thresholds[int(k * step)]);
            thresholds = sampled;
        }

        for (double threshold : thresholds) {
            std::vector<double> left_y, right_y;
            int left_count = 0, right_count = 0;
            for (int j = 0; j < n_samples; j++) {
                if (feature_values[j] <= threshold) { left_y.push_back(y[j]); left_count++; }
                else { right_y.push_back(y[j]); right_count++; }
            }
            // asegurar que cada lado tenga suficientes muestras
            if (left_count < min_samples_split || right_count < min_samples_split) continue;

            double mse = meanSquaredError(y, feature_values, threshold);
            if (mse < best_mse) {
                best_mse = mse;
                best_feature = feat_idx;
                best_threshold = threshold;
            }
        }
    }

    // stop si no hay mejora
    double curr_mse = 0.0;
    for (double val : y) curr_mse += (val - y_mean) * (val - y_mean);
    curr_mse /= y.size();
    if (best_feature == -1 || (curr_mse - best_mse) < 1e-6)
        return new Node(-1, -1.0, nullptr, nullptr, y_mean);

    // dividir dataset
    std::vector<std::vector<double>> X_left, X_right;
    std::vector<double> y_left, y_right;
    for (int i = 0; i < n_samples; i++) {
        if (X[i][best_feature] <= best_threshold) { X_left.push_back(X[i]); y_left.push_back(y[i]); }
        else { X_right.push_back(X[i]); y_right.push_back(y[i]); }
    }

    // fallback si algún lado vacío
    if (X_left.empty() || X_right.empty()) return new Node(-1, -1.0, nullptr, nullptr, y_mean);

    Node* left_child = growTree(X_left, y_left, depth + 1);
    Node* right_child = growTree(X_right, y_right, depth + 1);
    return new Node(best_feature, best_threshold, left_child, right_child, -1.0);
}

// meanSquaredError
double DecisionTreeRegression::meanSquaredError(std::vector<double>& y, std::vector<double>& X_column, double split_thresh) {
    std::vector<double> left_y, right_y;
    for (int i = 0; i < X_column.size(); i++) {
        if (X_column[i] <= split_thresh) left_y.push_back(y[i]);
        else right_y.push_back(y[i]);
    }
    if (left_y.empty() || right_y.empty()) return std::numeric_limits<double>::max();
    double left_mean = mean(left_y), right_mean = mean(right_y);
    double mse = 0.0;
    for (double v : left_y) mse += (v - left_mean) * (v - left_mean);
    for (double v : right_y) mse += (v - right_mean) * (v - right_mean);
    return mse / (left_y.size() + right_y.size());
}

// mean
double DecisionTreeRegression::mean(std::vector<double>& values) {
    if (values.empty()) return 0.0;
    double sum = std::accumulate(values.begin(), values.end(), 0.0);
    return sum / values.size();
}

// traverseTree
double DecisionTreeRegression::traverseTree(std::vector<double>& x, Node* node) {
    if (node->isLeafNode()) return node->value;
    if (x[node->feature] <= node->threshold) return traverseTree(x, node->left);
    else return traverseTree(x, node->right);
}

// runDecisionTreeRegression
std::tuple<double, double, double, double, double, double,
    std::vector<double>, std::vector<double>,
    std::vector<double>, std::vector<double>>
    DecisionTreeRegression::runDecisionTreeRegression(const std::string& filePath, int trainingRatio) {
    try {
        if (filePath.empty()) { MessageBox::Show("Please select dataset"); return {}; }
        std::ifstream file(filePath);
        if (!file.is_open()) { MessageBox::Show("Failed to open file"); return {}; }
        std::vector<std::vector<std::string>> data = DataLoader::readDatasetFromFilePath(filePath);

        std::vector<std::vector<double>> dataset;
        bool isFirstRow = true;
        for (const auto& row : data) {
            if (isFirstRow) { isFirstRow = false; continue; }
            std::vector<double> conv;
            for (const auto& cell : row) {
                try { conv.push_back(std::stod(cell)); }
                catch (...) { conv.push_back(0.0); }
            }
            dataset.push_back(conv);
        }

        double trainRatio = trainingRatio * 0.01;
        std::vector<std::vector<double>> trainData, testData;
        std::vector<double> trainLabels, testLabels;
        DataPreprocessor::splitDataset(dataset, trainRatio, trainData, trainLabels, testData, testLabels);

        fit(trainData, trainLabels);

        std::vector<double> testPredictions = predict(testData);
        double test_mae = Metrics::meanAbsoluteError(testLabels, testPredictions);
        double test_rmse = Metrics::rootMeanSquaredError(testLabels, testPredictions);
        double test_rsquared = Metrics::rSquared(testLabels, testPredictions);

        std::vector<double> trainPredictions = predict(trainData);
        double train_mae = Metrics::meanAbsoluteError(trainLabels, trainPredictions);
        double train_rmse = Metrics::rootMeanSquaredError(trainLabels, trainPredictions);
        double train_rsquared = Metrics::rSquared(trainLabels, trainPredictions);

        MessageBox::Show("Run completed");
        return std::make_tuple(test_mae, test_rmse, test_rsquared,
            train_mae, train_rmse, train_rsquared,
            std::move(trainLabels), std::move(trainPredictions),
            std::move(testLabels), std::move(testPredictions));
    }
    catch (const std::exception& e) {
        MessageBox::Show("Not Working");
        std::cerr << "Exception: " << e.what() << std::endl;
        return std::make_tuple(0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            std::vector<double>(), std::vector<double>(),
            std::vector<double>(), std::vector<double>());
    }
}