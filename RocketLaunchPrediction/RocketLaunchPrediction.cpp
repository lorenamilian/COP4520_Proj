#include <mlpack/core.hpp>
#include <mlpack/methods/random_forest/random_forest.hpp>
#include <iostream>
#include <mlpack/core/data/split_data.hpp>
#include <fstream>
#include <sstream>
#include <vector>


struct CSVData {
    // First column* (S/F) --- *column because csv file gets transposed when read by mlpack
	// Meaning rows become columns and columns become rows
    std::vector<std::string> labels;
    // Columns 5-13 (numerical features) --- skipping the text based data
    arma::mat features;
};

// This function reads the data from the CSV file and returns it as a CSVData struct.
// Necessary because there was a problem reading directly from the file using mlpacks data functions.
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
		// This is a simple way to handle any samples with missing data.
        if (row.size() < 14) {
            continue;
        }

        // Extract label (first column)
        labels.push_back(row[0]);

        // Extract features (columns 5-13)
		// Skip the first 5 columns because they contain text data.
        std::vector<double> featureRow;
        for (size_t j = 5; j <= 13; ++j) {
            try {
                featureRow.push_back(std::stod(row[j]));
            }
            // Basically we skip the sample if it's going to cause an issue for the rest of the data.
            catch (const std::invalid_argument&) {
                std::cerr << "Skipping row due to non-numeric value in feature column: " << row[j] << std::endl;
                featureRow.clear();
                break; // Skip the rest of this row.
            }
        }

        // Only add the feature row if it was successfully processed.
        if (!featureRow.empty()) {
            features.push_back(featureRow);
        }
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


// Main function
int main() {
    // Declare the CSVData struct
    CSVData csvData;

    // Call the ReadCSV function to load the data from the CSV file
    try {
        // The data file we are using is set here
        csvData = ReadCSV("1970DuplicatedFailTest.csv");
    }
    catch (const std::exception& e) {
        std::cerr << "Error loading data: " << e.what() << std::endl;
        return -1;
    }

    // 1. Convert Labels to Numeric (0, 1)
    // Since we are predicting a binary outcome, we need to convert the labels to numeric form.
    arma::Row<size_t> labels(csvData.labels.size());
    for (size_t i = 0; i < csvData.labels.size(); ++i) {
        std::string label = csvData.labels[i];
        if (label == "S") {
            labels[i] = 0;
        }
        else if (label == "F") {
            labels[i] = 1;
        }

        else {
            labels[i] = 0;
        }
    }

    // 2. Print Label Mappings
    // Just for reference
    std::cout << "=== Label Mappings ===" << std::endl;
    std::cout << "Class 0: S\nClass 1: F\n" << std::endl;

    // 3. Normalize and Split Data
    // Normalizing the data may not be entirely necessary for Random Forests, but it improves the acuracy of the model.
    arma::mat features = csvData.features;
    features = arma::normalise(features, 2, 0); // L2 normalization

    // Declare variables to store the training and testing data
    arma::mat trainFeatures, testFeatures;
    arma::Row<size_t> trainLabels, testLabels;
    // Split the data into training and testing sets (80% training, 20% testing)
    mlpack::data::Split(features, labels, trainFeatures, testFeatures, trainLabels, testLabels, 0.2);


    // 4. Train and Evaluate the Random Forest Model
    // Define Random Forest parameters.
    const size_t numClassesForest = 2;
    // You can adjust the number of trees, but the model seems accurate as it is.
    const size_t numTrees = 50;
    // Minimum number of points in a leaf.
    const size_t minimumLeafSize = 1;

    // Create and train the random forest classifier.
    mlpack::RandomForest<> rf;
    rf.Train(trainFeatures, trainLabels, numClassesForest, numTrees, minimumLeafSize);

    // Classify the test set.
    arma::Row<size_t> predictionsForest;
    rf.Classify(testFeatures, predictionsForest);


    // 5. Confusion Matrix and Metrics
    // This is a table of our model's performance.
    arma::Mat<size_t> confusionForest(numClassesForest, numClassesForest, arma::fill::zeros);
    for (size_t i = 0; i < testLabels.n_elem; ++i) {
        confusionForest(testLabels(i), predictionsForest(i))++;
    }

    std::vector<std::string> classNames = { "S", "F" };

    std::cout << "\n=== Confusion Matrix - Forest ===" << std::endl;
    std::cout << "          ";
    for (size_t j = 0; j < numClassesForest; ++j) {
        std::cout << "Predicted " << classNames[j] << "\t";
    }
    std::cout << std::endl;

    for (size_t i = 0; i < numClassesForest; ++i) {
        std::cout << "Actual " << classNames[i] << ":\t";
        for (size_t j = 0; j < numClassesForest; ++j) {
            std::cout << confusionForest(i, j) << "\t\t";
        }
        std::cout << std::endl;
    }


    // Metrics Calculations
    arma::vec precisionForest(numClassesForest), recallForest(numClassesForest), f1Forest(numClassesForest);
    for (size_t c = 0; c < numClassesForest; ++c) {
        double tp = confusionForest(c, c);
        double fp = arma::accu(confusionForest.col(c)) - tp;
        double fn = arma::accu(confusionForest.row(c)) - tp;

        // Handle division by zero
        if (tp + fp > 0) {
            precisionForest[c] = tp / (tp + fp);
        }
        else {
            precisionForest[c] = 0.0;
        }

        if (tp + fn > 0) {
            recallForest[c] = tp / (tp + fn);
        }
        else {
            recallForest[c] = 0.0;
        }

        if (precisionForest[c] + recallForest[c] > 0) {
            f1Forest[c] = 2 * (precisionForest[c] * recallForest[c]) / (precisionForest[c] + recallForest[c]);
        }
        else {
            f1Forest[c] = 0.0;
        }
    }

    // Print classification report
    std::cout << "\n=== Forest Classification Report ===" << std::endl;
    for (size_t c = 0; c < numClassesForest; ++c) {
        std::cout << "Class " << classNames[c] << " (" << c << "):\n"
            << "  Precision: " << precisionForest[c] << "\n"
            << "  Recall:    " << recallForest[c] << "\n"
            << "  F1-Score:  " << f1Forest[c] << "\n"
            << "----------------------------" << std::endl;
    }

    // Overall accuracy
    double accuracyForest = arma::accu(predictionsForest == testLabels)
        / static_cast<double>(testLabels.n_elem);
    std::cout << "\nOverall Accuracy - Forest: " << accuracyForest << std::endl;

    // Wait for user input before exiting
    std::cout << "Press Enter to exit..." << std::endl;
    std::cin.get();


    return 0;
}