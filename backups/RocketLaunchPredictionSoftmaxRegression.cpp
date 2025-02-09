#include <mlpack/core.hpp>
#include <mlpack/methods/softmax_regression.hpp>
#include <iostream>
#include <mlpack/core/data/split_data.hpp>
#include <fstream>
#include <sstream>
#include <vector>

struct CSVData {
    std::vector<std::string> labels;   // First column (S/F/PF)
    arma::mat features;                // Columns 5-13 (numerical features)
};

CSVData ReadCSV(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    std::vector<std::string> labels;
    std::vector<std::vector<double>> features;
    std::string line;

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string cell;
        std::vector<std::string> row;

        // Split line into cells
        while (std::getline(ss, cell, ',')) {
            row.push_back(cell);
        }

        // Check if row has fewer than 14 columns; if so, skip it.
        if (row.size() < 14) {
            std::cerr << "Skipping row due to insufficient columns: " << line << std::endl;
            continue;
        }

        // Extract label (first column)
        labels.push_back(row[0]);

        // Extract features (columns 5-13)
        std::vector<double> featureRow;
        for (size_t j = 5; j <= 13; ++j) {
            try {
                featureRow.push_back(std::stod(row[j]));
            }
            catch (const std::invalid_argument&) {
                throw std::runtime_error("Non-numeric value in feature column: " + row[j]);
            }
        }
        features.push_back(featureRow);
    }

    // Convert features to Armadillo matrix
    CSVData result;
    result.labels = labels;
    result.features = arma::mat(features[0].size(), features.size());
    for (size_t i = 0; i < features.size(); ++i) {
        result.features.col(i) = arma::vec(features[i]);
    }

    return result;
}

int main() {
    CSVData csvData;

    // Load the data
    try {
        csvData = ReadCSV("1970data.csv");
    }
    catch (const std::exception& e) {
        std::cerr << "Error loading data: " << e.what() << std::endl;
        return -1;
    }

    // ============================================
    // 1. Convert Labels to Numeric (0, 1, 2)
    // ============================================
    arma::Row<size_t> labels(csvData.labels.size());
    for (size_t i = 0; i < csvData.labels.size(); ++i) {
        std::string label = csvData.labels[i];
        std::cout << "label = " << label << std::endl;

        if (label == "S") {
            labels[i] = 0;
        }
        else if (label == "F") {
            labels[i] = 1;
        }
        else if (label == "PF") {
            labels[i] = 2;
        }
        else {
            std::cerr << "Unknown label: " << label << " (defaulting to S)" << std::endl;
            labels[i] = 0;
        }
    }

    // ============================================
    // 2. Print Label Mappings
    // ============================================
    std::cout << "=== Label Mappings ===" << std::endl;
    std::cout << "Class 0: S\nClass 1: F\nClass 2: PF" << std::endl;

    // ============================================
    // 3. Normalize and Split Data
    // ============================================
    arma::mat features = csvData.features;
    features = arma::normalise(features, 2, 0); // L2 normalization

    arma::mat trainFeatures, testFeatures;
    arma::Row<size_t> trainLabels, testLabels;
    mlpack::data::Split(features, labels, trainFeatures, testFeatures, trainLabels, testLabels, 0.2);

    // ============================================
    // 4. Train and Evaluate the Model
    // ============================================
    mlpack::SoftmaxRegression sr;
    sr.Train(trainFeatures, trainLabels, 3);

    arma::Row<size_t> predictions;
    sr.Classify(testFeatures, predictions);

    // ============================================
    // 5. Confusion Matrix and Metrics
    // ============================================
    size_t numClasses = 3;
    arma::Mat<size_t> confusion(numClasses, numClasses, arma::fill::zeros);
    for (size_t i = 0; i < testLabels.n_elem; ++i) {
        confusion(testLabels(i), predictions(i))++;
    }

    std::vector<std::string> classNames = { "S", "F", "PF" };

    std::cout << "\n=== Confusion Matrix ===" << std::endl;
    std::cout << "          ";
    for (size_t j = 0; j < numClasses; ++j) {
        std::cout << "Predicted " << classNames[j] << "\t";
    }
    std::cout << std::endl;

    for (size_t i = 0; i < numClasses; ++i) {
        std::cout << "Actual " << classNames[i] << ":\t";
        for (size_t j = 0; j < numClasses; ++j) {
            std::cout << confusion(i, j) << "\t\t";
        }
        std::cout << std::endl;
    }

    // Metrics Calculations
    arma::vec precision(numClasses), recall(numClasses), f1(numClasses);
    for (size_t c = 0; c < numClasses; ++c) {
        double tp = confusion(c, c);
        double fp = arma::accu(confusion.col(c)) - tp;
        double fn = arma::accu(confusion.row(c)) - tp;

        // Handle division by zero safely
        precision[c] = (tp + fp > 0) ? tp / (tp + fp) : 0.0;
        recall[c] = (tp + fn > 0) ? tp / (tp + fn) : 0.0;
        f1[c] = (precision[c] + recall[c] > 0)
            ? 2 * (precision[c] * recall[c]) / (precision[c] + recall[c])
            : 0.0;
    }

    // Print classification report
    std::cout << "\n=== Classification Report ===" << std::endl;
    for (size_t c = 0; c < numClasses; ++c) {
        std::cout << "Class " << classNames[c] << " (" << c << "):\n"
            << "  Precision: " << precision[c] << "\n"
            << "  Recall:    " << recall[c] << "\n"
            << "  F1-Score:  " << f1[c] << "\n"
            << "----------------------------" << std::endl;
    }

    // Overall accuracy
    double accuracy = arma::accu(predictions == testLabels)
        / static_cast<double>(testLabels.n_elem);
    std::cout << "\nOverall Accuracy: " << accuracy << std::endl;

    return 0;
}