#include "LinearRegression.h"
#include "../DataUtils/DataLoader.h"
#include "../Utils/SimilarityFunctions.h"
#include "../Evaluation/Metrics.h"
#include "../DataUtils/DataPreprocessor.h"
#include "../Utils/SimilarityFunctions.h"
#include "../Evaluation/Metrics.h"
#include <cmath>
#include <string>
#include <algorithm>
#include <utility>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <map>
#include <random>
#include <unordered_map>
#include <msclr\marshal_cppstd.h>
#include <stdexcept>
#include "../MainForm.h"
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
using namespace System::Windows::Forms; // For MessageBox



///  LinearRegression class implementation  ///


// Function to fit the linear regression model to the training data //
void LinearRegression::fit(const std::vector<std::vector<double>>& trainData, const std::vector<double>& trainLabels) {

    // This implementation is using Matrix Form method
    /*
     Implement the following :
          --- Check if the sizes of trainData and trainLabels match
          --- Convert trainData to matrix representation
          --- Construct the design matrix X
            --- Convert trainLabels to matrix representation
            --- Calculate the coefficients using the least squares method
            --- Store the coefficients for future predictions
      */

      // TODO

    // --- Check if the sizes of trainData and trainLabels match ---
    // Verificamos que cada fila de datos tenga su correspondiente etiqueta
    if (trainData.size() != trainLabels.size()) {
        throw std::invalid_argument("sizes of trainData and trainLabels does not match");
    }

    // --- Convert trainData to matrix representation ---
    // Creamos una matriz X de Eigen con dimensiones: (número de muestras) x (número de características + 1)
    // El +1 es para la columna de bias (término independiente)
    Eigen::MatrixXd X(trainData.size(), trainData[0].size() + 1);

    // Construimos la matriz de diseño X
    for (int i = 0; i < trainData.size(); ++i) {
        X(i, 0) = 1.0; // Primera columna siempre es 1.0 para el término bias (intercept). Así la línea no está forzada por el origen; se ajusta mejor
        for (int j = 0; j < trainData[0].size(); ++j) {
            X(i, j + 1) = trainData[i][j]; // Las siguientes columnas son las características originales
        }
    }

    // --- Convert trainLabels to matrix representation---
    // Convertimos las etiquetas de entrenamiento a un vector Eigen
    Eigen::VectorXd y(trainLabels.size());

    for (size_t i = 0; i < trainLabels.size(); ++i) {
        y(i) = trainLabels[i]; // Cada etiqueta se asigna al vector y
    }

    // --- Calculate the coefficients using the least squares method ---
    // Fórmula de mínimos cuadrados β = (XᵀX)⁻¹Xᵀy
    // Donde:
    // - Xᵀ es la transpuesta de X
    // - (XᵀX)⁻¹ es la inversa de XᵀX
    // - β son los coeficientes del modelo (incluyendo el bias)
    m_coefficients = (X.transpose() * X).inverse() * X.transpose() * y;

    // --- Store the coefficients for future predictions ---
    // Los coeficientes quedan almacenados en m_coefficients para usar en predict()
    // m_coefficients[0] es el bias, m_coefficients[1...] son los pesos de cada característica
}


// Function to make predictions on new data //
std::vector<double> LinearRegression::predict(const std::vector<std::vector<double>>& testData) {

    // This implementation is using Matrix Form method    
  /* Implement the following
            --- Check if the model has been fitted
            --- Convert testData to matrix representation
            --- Construct the design matrix X
            --- Make predictions using the stored coefficients
            --- Convert predictions to a vector
      */

      // TODO

    // --- Check if the model has been fitted
    // Verificamos que el modelo haya sido entrenado (que existan coeficientes)
    if (m_coefficients.size() == 0) {
        throw std::runtime_error("model has not been fitted");
    }

    // --- Convert trainData to matrix representation ---
    // Creamos una matriz X para los datos de prueba, con la misma estructura que en fit()
    Eigen::MatrixXd X(testData.size(), testData[0].size() + 1);

    // --- Construct the design matrix X ---
    // Construimos la matriz de diseño para los datos de prueba
    for (int i = 0; i < testData.size(); ++i) {
        X(i, 0) = 1.0; // Columna de bias siempre 1.0
        for (int j = 0; j < testData[0].size(); ++j) {
            X(i, j + 1) = testData[i][j]; // Caracteísticas de prueba
        }
    }

    // --- Make predictions using the stored coefficients ---
    // Realizamos la predicción: ŷ = X * β
    // Donde:
    // - X es la matriz de diseño de prueba
    // - β son los coeficientes aprendidos durante el entrenamiento
    // - ŷ son las predicciones resultantes
    Eigen::VectorXd predictions = X * m_coefficients;

    // --- Convert predictions to a vector ---
   // Convertimos las predicciones de Eigen::VectorXd a std::vector<double>
    std::vector<double> result(testData.size());
    for (size_t i = 0; i < predictions.size(); ++i) {
        result[i] = predictions[i]; // Cada precicción se copia al vector result
    }

    return result; // Return the vector of predictions
}


// === GRADIENT DESCENT OVERLOAD === //

// Gradient Descent fit method (overload)
void LinearRegression::fit(const std::vector<std::vector<double>>& trainData,
    const std::vector<double>& trainLabels,
    double learning_rate,
    int num_epochs) {

    // Validaciones
    if (trainData.empty() || trainLabels.empty()) {
        throw std::invalid_argument("fit: trainData or trainLabels is empty.");
    }
    if (trainData.size() != trainLabels.size()) {
        throw std::invalid_argument("fit: trainData and trainLabels must have the same number of samples.");
    }

    int num_samples = static_cast<int>(trainData.size());
    int num_features = static_cast<int>(trainData[0].size());

    // Check consistent feature sizes
    for (size_t i = 0; i < trainData.size(); ++i) {
        if (static_cast<int>(trainData[i].size()) != num_features) {
            throw std::invalid_argument("fit: inconsistent feature vector sizes in trainData.");
        }
    }

    // Initialize weights: (num_features + 1) para incluir bias
    m_coefficients_gd.resize(num_features + 1);

    // Random initialization (mejor que inicializar con ceros)
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<double> dis(-0.01, 0.01);

    for (int j = 0; j < num_features + 1; ++j) {
        m_coefficients_gd(j) = dis(gen);
    }

    // SGD with shuffling per epoch
    std::mt19937 shuffle_gen(std::random_device{}());
    std::vector<int> indices(num_samples);
    std::iota(indices.begin(), indices.end(), 0);

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        std::shuffle(indices.begin(), indices.end(), shuffle_gen);

        for (int idx : indices) {
            // Crear vector x con bias term al frente
            std::vector<double> x_with_bias;
            x_with_bias.reserve(num_features + 1);
            x_with_bias.push_back(1.0); // bias
            x_with_bias.insert(x_with_bias.end(), trainData[idx].begin(), trainData[idx].end());

            // Prediction: ŷ = w₀ + w₁x₁ + w₂x₂ + ... + wₙxₙ
            double prediction = 0.0;
            for (size_t j = 0; j < x_with_bias.size(); ++j) {
                prediction += m_coefficients_gd(j) * x_with_bias[j];
            }

            // Error: e = ŷ - y (para Linear Regression)
            double error = prediction - trainLabels[idx];

            // Gradient step: w = w - learning_rate * error * x
            for (size_t j = 0; j < x_with_bias.size(); ++j) {
                m_coefficients_gd(j) -= learning_rate * error * x_with_bias[j];
            }
        } // end samples
    } // end epochs

    use_gd_model = true; // Indicar que usamos el modelo GD
}


// Gradient Descent predict method (overload)
std::vector<double> LinearRegression::predict(const std::vector<std::vector<double>>& testData,
    bool useGradientDescentModel) {
    std::vector<double> predictions;

    if (testData.empty()) {
        return predictions; // empty result
    }

    // Elegir cuál coeficientes usar según el flag
    Eigen::VectorXd* coeffs_to_use = nullptr;

    if (useGradientDescentModel) {
        if (m_coefficients_gd.size() == 0) {
            throw std::runtime_error("predict: GD model weights are empty (train the model first with GD).");
        }
        coeffs_to_use = &m_coefficients_gd;
    }
    else {
        if (m_coefficients.size() == 0) {
            throw std::runtime_error("predict: Matrix model weights are empty (train the model first).");
        }
        coeffs_to_use = &m_coefficients;
    }

    int expected_feature_count = static_cast<int>(coeffs_to_use->size()) - 1;

    for (const auto& x : testData) {
        if (static_cast<int>(x.size()) != expected_feature_count) {
            throw std::invalid_argument("predict: test sample has different number of features than trained model.");
        }

        // Crear vector con bias term
        std::vector<double> x_with_bias;
        x_with_bias.reserve(expected_feature_count + 1);
        x_with_bias.push_back(1.0);
        x_with_bias.insert(x_with_bias.end(), x.begin(), x.end());

        // Prediction: ŷ = w₀ + w₁x₁ + w₂x₂ + ... + wₙxₙ
        double prediction = 0.0;
        for (size_t j = 0; j < x_with_bias.size(); ++j) {
            prediction += (*coeffs_to_use)(j) * x_with_bias[j];
        }

        predictions.push_back(prediction);
    }

    return predictions;
}


/// runLinearRegression: this function runs the Linear Regression algorithm on the given dataset and
/// then returns a tuple containing the evaluation metrics for the training and test sets,
/// as well as the labels and predictions for the training and test sets. ///

std::tuple<double, double, double, double, double, double,
    std::vector<double>, std::vector<double>,
    std::vector<double>, std::vector<double>>
    LinearRegression::runLinearRegression(const std::string& filePath, int trainingRatio) {
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
        // Load the dataset from the file path
        std::vector<std::vector<std::string>> data = DataLoader::readDatasetFromFilePath(filePath);

        // Convert the dataset from strings to doubles
        std::vector<std::vector<double>> dataset;
        bool isFirstRow = true; // Flag to identify the first row

        for (const auto& row : data) {
            if (isFirstRow) {
                isFirstRow = false;
                continue; // Skip the first row (header)
            }

            std::vector<double> convertedRow;
            for (const auto& cell : row) {
                try {
                    double value = std::stod(cell);
                    convertedRow.push_back(value);
                }
                catch (const std::exception& e) {
                    // Handle the exception or set a default value
                    std::cerr << "Error converting value: " << cell << std::endl;
                    // You can choose to set a default value or handle the error as needed
                }
            }
            dataset.push_back(convertedRow);
        }

        // Split the dataset into training and test sets (e.g., 80% for training, 20% for testing)
        double trainRatio = trainingRatio * 0.01;

        std::vector<std::vector<double>> trainData;
        std::vector<double> trainLabels;
        std::vector<std::vector<double>> testData;
        std::vector<double> testLabels;

        DataPreprocessor::splitDataset(dataset, trainRatio, trainData, trainLabels, testData, testLabels);

        // === DECISIÓN: Gradient Descent (true) o Matrix (false) ===
        bool useGradientDescent = true; // ← CAMBIA AQUÍ: true para Gradient Descent, false para Matrix

        // Fit the model based on the decision
        if (useGradientDescent) {
            // Parámetros para Gradient Descent
            double learning_rate = 0.000001;
            int num_epochs = 5000;
            fit(trainData, trainLabels, learning_rate, num_epochs);
        }
        else {
            // Método matricial
            fit(trainData, trainLabels);
        }

        // Make predictions based on the method used
        std::vector<double> testPredictions;
        std::vector<double> trainPredictions;

        if (useGradientDescent) {
            testPredictions = predict(testData, true);
            trainPredictions = predict(trainData, true);
        }
        else {
            testPredictions = predict(testData);
            trainPredictions = predict(trainData);
        }

        // Calculate evaluation metrics (e.g., MAE, MSE)
        double test_mae = Metrics::meanAbsoluteError(testLabels, testPredictions);
        double test_rmse = Metrics::rootMeanSquaredError(testLabels, testPredictions);
        double test_rsquared = Metrics::rSquared(testLabels, testPredictions);

        // Calculate evaluation metrics for training data
        double train_mae = Metrics::meanAbsoluteError(trainLabels, trainPredictions);
        double train_rmse = Metrics::rootMeanSquaredError(trainLabels, trainPredictions);
        double train_rsquared = Metrics::rSquared(trainLabels, trainPredictions);



        return std::make_tuple(test_mae, test_rmse, test_rsquared,
            train_mae, train_rmse, train_rsquared,
            std::move(trainLabels), std::move(trainPredictions),
            std::move(testLabels), std::move(testPredictions));
    }
    catch (const std::exception& e) {
        // Handle the exception
        MessageBox::Show("Not Working");
        std::cerr << "Exception occurred: " << e.what() << std::endl;
        return std::make_tuple(0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            std::vector<double>(), std::vector<double>(),
            std::vector<double>(), std::vector<double>());
    }
}