#ifndef LINEARREGRESSION_H
#define LINEARREGRESSION_H

#include "../DataUtils/DataLoader.h"
#include "../DataUtils/DataPreprocessor.h"
#include "../Utils/SimilarityFunctions.h"
#include "../Evaluation/Metrics.h"
#include <string>
#include <vector>
#include <tuple>
#include <iostream>
#include <Eigen/Dense>

/// LinearRegression class definition ///
class LinearRegression {
public:
    // --- Forma matricial ---
    void fit(const std::vector<std::vector<double>>& trainData,
        const std::vector<double>& trainLabels);

    std::vector<double> predict(const std::vector<std::vector<double>>& testData);

    // --- Gradient Descent (overload) ---
    void fit(const std::vector<std::vector<double>>& trainData,
        const std::vector<double>& trainLabels,
        double learning_rate,
        int num_epochs);

    std::vector<double> predict(const std::vector<std::vector<double>>& testData,
        bool useGradientDescentModel);

    // --- Función principal para correr el experimento ---
    std::tuple<double, double, double, double, double, double,
        std::vector<double>, std::vector<double>,
        std::vector<double>, std::vector<double>>
        runLinearRegression(const std::string& filePath, int trainingRatio);

private:
    Eigen::VectorXd m_coefficients;              // Para Matrix Form
    Eigen::VectorXd m_coefficients_gd;           // Para Gradient Descent
    bool use_gd_model = false;                   // Flag para saber cuál modelo usar
};

#endif // LINEARREGRESSION_H