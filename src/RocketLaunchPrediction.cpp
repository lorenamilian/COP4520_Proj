#include <format>
#include <mlpack/core.hpp>
#include <mlpack/methods/random_forest/random_forest.hpp>
#include <mlpack/core/data/split_data.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>
#include <thread>
#include <mutex>
#include <bitset>
#include <chrono>

#include <cpr/cpr.h>
#include <nlohmann/json.hpp>
using json = nlohmann::json;

using namespace mlpack;
using namespace std;


//=========================== Utility Functions =================================//

// Get the current Time as an ISO8601 string
std::string GetCurrentISO8601() {
    std::time_t now = std::time(nullptr);
    std::tm* gmt = std::gmtime(&now);
    char buf[30];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S", gmt);
    return std::string(buf);
}


// Get the time 10 days from now
std::string GetFutureISO8601(int daysAhead) {
    std::time_t now = std::time(nullptr);
	now += daysAhead * 24 * 60 * 60; // Add daysAhead days
	std::tm* gmt = std::gmtime(&now);
	char buf[30];
	std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S", gmt);
	return std::string(buf);
}


// Function to print the equals sign for binary representation (double)
void PrintEqualsForBinary(double decimalNumber, std::vector<string>& training_output) {
  int integerPart = static_cast<int>(decimalNumber); // Get the integer part
  stringstream ss;
  ss << "|";
  for (int i = 0; i < integerPart; ++i) {
    ss << "===";
  }
  ss << "|";
  training_output.push_back("");
  training_output.push_back(ss.str());
}


// Function to print the equals sign for binary representation (int)
void PrintEqualsForBinary(int decimalNumber) {
    // Convert decimal number to binary using std::bitset (assuming a 32-bit number)
    std::bitset<32> binary(decimalNumber);

    // Print '=' for each bit in the binary representation
    for (size_t i = 0; i < binary.size(); ++i) {
        std::cout << "=";
    }
    std::cout << std::endl;
}


// Function to compare sequential and parallel execution times
void CompareTimes(double seqTime, double parTime, std::vector<string>& training_output) {
  if (parTime < seqTime) {
    training_output.push_back(std::format("Parallel was approximately {}x faster.", seqTime/parTime));
    training_output.push_back("");
  }
  else {
    training_output.push_back(std::format("Sequential was approximately {}x faster.", parTime/ seqTime));
    training_output.push_back("");
  }
}


//=========================== Data Structures =================================//

// Structure to store the data read from the CSV file
struct CSVData {
    // First column* (S/F) --- *column because csv file gets transposed when read by mlpack
	// Meaning rows become columns and columns become rows
    std::vector<std::string> labels;
    // Columns 5-13 (numerical features) --- skipping the text based data
    arma::mat features;
};

// Structure to store the best hyperparameters found
struct BestHyperparams {
    size_t numClassesForest = 2;
    size_t numTrees = 50;
    size_t minimumLeafSize = 1;
    double bestAccuracy = 0.0;
};


//=========================== CSV Loading and Model Training =================================//

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


std::mutex mtx; // Mutex for safe updating of shared variables in hyperparm search

// Function to evaluate hyperparameters in parallel
void EvaluateHyperparameters(const arma::mat& trainFeatures,
                             const arma::Row<size_t>& trainLabels,
                             const arma::mat& testFeatures,
                             const arma::Row<size_t>& testLabels,
                             size_t numTrees, size_t minimumLeafSize,
                             BestHyperparams& bestParams)
{
    // Train a Random Forest with given hyperparameters
    mlpack::RandomForest<> rf;
    rf.Train(trainFeatures, trainLabels, 2, numTrees, minimumLeafSize);

    // Classify test set
    arma::Row<size_t> predictions;
    rf.Classify(testFeatures, predictions);

    // Compute accuracy
    double accuracy = arma::accu(predictions == testLabels) / static_cast<double>(testLabels.n_elem);

    // Safely update best hyperparameters if accuracy is higher
    std::lock_guard<std::mutex> lock(mtx);
    if (accuracy > bestParams.bestAccuracy) {
        bestParams.bestAccuracy = accuracy;
        bestParams.numTrees = numTrees;
        bestParams.minimumLeafSize = minimumLeafSize;
    }
}


// Function to evaluate hyperparameters sequentially
double TimerForSequentialHyperparameterSearch(const arma::mat& trainFeatures,
    const arma::Row<size_t>& trainLabels,
    const arma::mat& testFeatures,
    const arma::Row<size_t>& testLabels,
    std::vector<string>& training_output)
{
    // Record start time
    auto start = std::chrono::high_resolution_clock::now();

    BestHyperparams bestParams;

    std::vector<size_t> numTreesList = { 10, 30, 50 };
    std::vector<size_t> minLeafSizeList = { 1, 2, 3, 4 };

    // Sequential hyperparameter search
    for (size_t trees : numTreesList) {
        for (size_t leafSize : minLeafSizeList) {
            EvaluateHyperparameters(trainFeatures, trainLabels, testFeatures, testLabels, trees, leafSize, bestParams);
        }
    }

    // Record end time
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    training_output.push_back(std::format("Sequential hyperparameter search: ({}s)", duration.count()));
    // PrintEqualsForBinary(duration.count(), training_output);
    return duration.count();

}







// MAIN FUNCTION TO TRAIN MODEL
// Trains the Random Forest model using the CSV data, performs hyperparmeter search,
// prints performance metrics, and returns the trained model
std::tuple<mlpack::RandomForest<>, vector<string>> TrainModel() {

  // WARN: testing to see if I can use vector of strings to redirect output.
  std::vector<std::string> training_output;

    // std::cout << "Loading training data..." << std::endl;
	CSVData csvData = ReadCSV("1970DuplicatedFailTest.csv");


    // Convert Labels ("S" -> 0, "F" -> 1; unknown defaults to 0)
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
  training_output.push_back("--------------------------------------- Label Mappings --------------------------------------\n");
  training_output.push_back("Class 0: S\nClass 1: F\n");


    // 3. Normalize and Split Data
    // Normalizing the data may not be entirely necessary for Random Forests, but it improves the acuracy of the model.
    arma::mat features = csvData.features;
    features = arma::normalise(features, 2, 0); // L2 normalization

    // Scale the latitude and longitude features
    // We scale latitude and longitude to improve the prediction
    features.row(0) *= 0.1; // Scale the latitude feature
    features.row(1) *= 0.1; // Scale the longitude feature

    // Declare variables to store the training and testing data
    arma::mat trainFeatures, testFeatures;
    arma::Row<size_t> trainLabels, testLabels;
    // Split the data into training and testing sets (80% training, 20% testing)
    mlpack::data::Split(features, labels, trainFeatures, testFeatures, trainLabels, testLabels, 0.2);


    // 4. Parallel hyperparameter search
    // Start timer for hyperparameter search
    auto start = std::chrono::high_resolution_clock::now();
    BestHyperparams bestParams;
    std::vector<std::thread> threads;

    // Hyperparameter ranges
    std::vector<size_t> numTreesList = { 10, 30, 50 };
    std::vector<size_t> minLeafSizeList = { 1, 2, 3, 4 };

    // Launch threads for each hyperparameter combination
    for (size_t trees : numTreesList) {
        for (size_t leafSize : minLeafSizeList) {
            threads.emplace_back(EvaluateHyperparameters, trainFeatures, trainLabels,
                testFeatures, testLabels, trees, leafSize, std::ref(bestParams));
        }
    }

    // Wait for all threads to finish
    for (auto& t : threads) {
        t.join();
    }

    // End timer for hyperparameter search
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    // Assign the best found hyperparameters
    const size_t numClassesForest = bestParams.numClassesForest;
    const size_t numTrees = bestParams.numTrees;
    // Minimum number of points in a leaf.
    const size_t minimumLeafSize = bestParams.minimumLeafSize;

  training_output.push_back(std::format("--------------------------------- Best Hyperparameters Found --------------------------------"));
  training_output.push_back(std::format("Number of Trees: {}\nMinimum Leaf Size: {}", numTrees, minimumLeafSize));


    // 5. Train and Evaluate the Random Forest Model
    mlpack::RandomForest<> rf;
    rf.Train(trainFeatures, trainLabels, numClassesForest, numTrees, minimumLeafSize);

    // Classify the test set.
    arma::Row<size_t> predictionsForest;
    rf.Classify(testFeatures, predictionsForest);


    // 6. Confusion Matrix and Metrics
    // This is a table of our model's performance.
    arma::Mat<size_t> confusionForest(numClassesForest, numClassesForest, arma::fill::zeros);
    for (size_t i = 0; i < testLabels.n_elem; ++i) {
        confusionForest(testLabels(i), predictionsForest(i))++;
    }

    std::vector<std::string> classNames = { "S", "F" };

  training_output.push_back(std::format("\n--------------------------------- Confusion Matrix - Forest ---------------------------------\n\n          "));
  for (size_t j = 0; j < numClassesForest; ++j) {
    training_output.push_back(std::format("Predicted {}", classNames[j]));
  }

  training_output.push_back("\n");

  for (size_t i = 0; i < numClassesForest; ++i) {

    training_output.push_back(string("Actual " + classNames[i] + ": \t"));
    for (size_t j = 0; j < numClassesForest; ++j) {
      training_output.push_back(std::format("{}\t\t", confusionForest(i, j)));
    }
    training_output.push_back("\n");
  }

  // Metrics Calculations
  arma::vec precisionForest(numClassesForest, arma::fill::zeros);
  arma::vec recallForest(numClassesForest, arma::fill::zeros);
  arma::vec f1Forest(numClassesForest, arma::fill::zeros);

  for (size_t c = 0; c < numClassesForest; ++c) {
    double tp = confusionForest(c, c);
    double fp = arma::accu(confusionForest.col(c)) - tp;
    double fn = arma::accu(confusionForest.row(c)) - tp;
    double tn = arma::accu(confusionForest) - tp - fp - fn;

    // Calculate precision, recall, and F1 score for each class
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
    training_output.push_back("\n------------------------------- Forest Classification Report --------------------------------\n");
    for (size_t c = 0; c < numClassesForest; ++c) {

    training_output.push_back(std::format("Class {} ({})",classNames[c], c));
    training_output.push_back(std::format("Precision: {}\n  Recall: {}\n", precisionForest[c], recallForest[c]));
    training_output.push_back(std::format("F1-Score: {}\n", f1Forest[c]));
    }

    // Overall accuracy
    double accuracyForest = arma::accu(predictionsForest == testLabels)
        / static_cast<double>(testLabels.n_elem);
    training_output.push_back(string("\nOverall Accuracy - Forest: {}\n", accuracyForest));


  training_output.push_back("\n-----------------Parallel vs. Sequential: Hyperparameter Search Time Analysis-----------------\n\n");
    // Print the duration of the hyperparameter search
    training_output.push_back(std::format("Parallel hyperparameter search: ({}s)", duration.count()));


  // TODO: Need to figure this out
    // PrintEqualsForBinary(duration.count(), training_output);

    // TODO: Figure this out too
    double SeqTime = TimerForSequentialHyperparameterSearch(trainFeatures, trainLabels, testFeatures, testLabels, training_output);
    CompareTimes(SeqTime, duration.count(), training_output);
    return make_tuple(rf, training_output);
}










//=========================== Weather and Launch API Functions =================================//

// MENU OPTION 1
// Call open meteo to get weather for launch pads at
// Cape Canaveral, Vandenberg, or Baikonur
arma::vec GetWeatherFeaturesForLaunchPad(int pad) {
    // Coordinates for the launch pads.
    double lat, lon;
    if (pad == 0) { // Cape Canaveral
        lat = 28.5623;
        lon = -80.5774;
    }
    else if (pad == 1) { // Vandenberg Air Force Base
        lat = 34.7420;
        lon = -120.5724;
    }
    else if (pad == 2) { // Baikonur Cosmodrome
        lat = 45.9654;
        lon = 63.3056;
    }
    else {
        throw std::invalid_argument("Invalid launch pad selection.");
    }

    // Build API request URL for Open-Meteo.
    std::string url = "https://api.open-meteo.com/v1/forecast?latitude=" +
        std::to_string(lat) + "&longitude=" + std::to_string(lon) +
        "&hourly=dewpoint_2m,pressure_msl,cloudcover,windspeed_10m,windspeed_100m,windgusts_10m,boundary_layer_height" +
        "&timezone=auto";

    // Make the GET request.
    cpr::Response r = cpr::Get(cpr::Url{ url });
    if (r.status_code != 200) {
        std::cerr << "Error: HTTP status code " << r.status_code << std::endl;
        throw std::runtime_error("Failed to fetch weather data.");
    }

    // DEBUG: Print the raw JSON response.
    //std::cout << "API response from Open-Meteo:" << std::endl;
    //std::cout << r.text << std::endl;

    // Parse the JSON response.
    json j = json::parse(r.text);

    // Verify that the response contains the expected 'hourly' object.
    if (!j.contains("hourly")) {
        throw std::runtime_error("Response does not contain 'hourly' data.");
    }

    // Get the hourly data.
    json hourly = j["hourly"];
    // We assume that each array (e.g., "dewpoint_2m") has at least one element.
    // In a robust application, you'd match the current time.
    if (hourly["dewpoint_2m"].empty() || hourly["pressure_msl"].empty() ||
        hourly["cloudcover"].empty() || hourly["windspeed_10m"].empty() ||
        hourly["windspeed_100m"].empty() || hourly["windgusts_10m"].empty() ||
        hourly["boundary_layer_height"].empty()) {
        throw std::runtime_error("Incomplete hourly weather data.");
    }

    // Prepare a feature vector with 9 elements.
    arma::vec features(9);
    features(0) = lat; // Use longitude as a feature.
    features(1) = lon;
    features(2) = hourly["dewpoint_2m"][0];      // dew point at 2m
    features(3) = hourly["pressure_msl"][0];       // surface pressure
    features(4) = hourly["cloudcover"][0];         // cloud cover percentage
    features(5) = hourly["windspeed_10m"][0];        // wind speed at 10m
    features(6) = hourly["windspeed_100m"][0];       // wind speed at 100m
    features(7) = hourly["windgusts_10m"][0];        // wind gusts
    features(8) = hourly["boundary_layer_height"][0];  // boundary layer height

    return features;
}


// Get weather at specific latitude and longitude obtained
// from location of launch pad reported by The Space Devs API
arma::vec GetWeatherFeaturesForLocation(double lat, double lon, const std::string& launchTime) {
    std::string url = "https://api.open-meteo.com/v1/forecast?latitude=" +
        std::to_string(lat) + "&longitude=" + std::to_string(lon) +
        "&hourly=dewpoint_2m,pressure_msl,cloudcover,windspeed_10m,windspeed_100m,windgusts_10m,boundary_layer_height" +
        "&timezone=auto";
    // Make the GET request
    cpr::Response r = cpr::Get(cpr::Url{ url });
    if (r.status_code != 200) {
        std::cerr << "Error: HTTP status code " << r.status_code << std::endl;
        throw std::runtime_error("Failed to fetch weather data.");
    }

    // DEBUG - print raw JSON response
    //std::cout << "Weather API response:" << std::endl;
    //std::cout << r.text << std::endl;


	// Parse the JSON response
    json j = json::parse(r.text);
    if (!j.contains("hourly")) {
        throw std::runtime_error("Response does not contain 'hourly' data.");
    }
    json hourly = j["hourly"];

	// We need to find the forecast index corresponsindg to the launch time
    int index = -1;
    std::string launchTimeTrim = launchTime.substr(0, 13); // "YYYY-MM-DDTHH"
    for (size_t i = 0; i < hourly["time"].size(); ++i) {
        std::string forecastTime = hourly["time"][i];
        // Trim forecast time to 16 characters (YYYY-MM-DDTHH)
        if (forecastTime.size() >= 13) {
            forecastTime = forecastTime.substr(0, 13);
        }

        if (forecastTime == launchTimeTrim) {
            index = static_cast<int>(i);
            break;
        }
    }

    if (index == -1) {
        std::cerr << "No forecast found matching launch time " << launchTime << ". Using first forecast." << std::endl;
		index = 0;
    }


    auto getFirstValue = [&](const std::string& key, double defaultVal) -> double {
        if (hourly.contains(key) && !hourly[key].empty())
            return hourly[key][0];
        else {
            std::cerr << "Warning: '" << key << "' data missing; using default value " << defaultVal << std::endl;
            return defaultVal;
        }
        };

    // Prepare a feature vector with 9 elements.
    arma::vec features(9);
    features(0) = lon;                         // Use longitude as a feature.
    features(1) = lat;                         // Include latitude if desired.
    features(2) = hourly["dewpoint_2m"][index]; // dew point at 2m.
    features(3) = hourly["pressure_msl"][index];  // surface pressure.
    features(4) = hourly["cloudcover"][index];    // cloud cover percentage.
    features(5) = hourly["windspeed_10m"][index];   // wind speed at 10m.
    features(6) = hourly["windspeed_100m"][index];  // wind speed at 100m.
    features(7) = hourly["windgusts_10m"][index];   // wind gusts at 10m.
    features(8) = hourly["boundary_layer_height"][index]; // boundary layer height.

    return features;
}









// MENU OPTION 2
// Call The Space Devs API to find the next 5 upcoming launches
// that fall within the next 10 days (10 days or less to ensure
// that we can get a weather prediction for launch date)
std::vector<json> GetUpcomingLaunches() {
    // Build a time window: now to 10 days from now.
    std::string startTime = GetCurrentISO8601();
    std::string endTime = GetFutureISO8601(10);

    // Construct the URL with query parameters.
    std::string url = "https://ll.thespacedevs.com/2.2.0/launch/upcoming/"
        "?window_start__gte=" + startTime +
        "&window_start__lte=" + endTime +
        "&limit=5";  // up to 5 launches

    cpr::Response r = cpr::Get(cpr::Url{ url });
    if (r.status_code != 200) {
        std::cerr << "Error fetching launches. HTTP status: " << r.status_code << std::endl;
        throw std::runtime_error("Failed to fetch upcoming launches.");
    }

    // DEBUG
    //std::cout << "Launches API response:" << std::endl;
    //std::cout << r.text << std::endl;

    // Parse the JSON.
    json j = json::parse(r.text);

    // The launches in results array
    std::vector<json> launches;
    if (j.contains("results")) {
        for (const auto& item : j["results"]) {
            launches.push_back(item);
        }
    }
    else {
        throw std::runtime_error("No 'results' found in launch API response.");
    }
    return launches;
}


// Function to extract launch pad coordinates from JSON obj
std::pair<double, double> GetPadCoordinates(const json& launch) {
    if (launch.contains("pad") && launch["pad"].contains("latitude") && launch["pad"].contains("longitude")) {
        double lat = std::stod(launch["pad"]["latitude"].get<std::string>());
        double lon = std::stod(launch["pad"]["longitude"].get<std::string>());
        return { lat, lon };
    }
    throw std::runtime_error("Launch JSON does not contain pad coordinates.");
}







//=========================== Menu Options Functions =================================//

// Option 1: Test current weather for a specific launch pad
void menuOption1(mlpack::RandomForest<>& rf, int pad_choice, vector<string>& output) {
    arma::mat liveFeatures = GetWeatherFeaturesForLaunchPad(pad_choice);
    if (liveFeatures.n_elem == 0) {
        std::cout << "Failed to retrieve weather data. Please try again later." << std::endl;
        return;
    }

    // --- Display the weather data (un-normalized) ---
    output.push_back(std::format("\nCurrent Weather Data:"));
    output.push_back(std::format("Latitude: {}", liveFeatures[0]));
    output.push_back(std::format("Longitude: {}", liveFeatures[1]));
    output.push_back(std::format("Dew Point (2m): {}", liveFeatures[2]));
    output.push_back(std::format("Surface Pressure: {}", liveFeatures[3]));
    output.push_back(std::format("Cloud Cover: {}", liveFeatures[4]));
    output.push_back(std::format("Wind Speed (10m): {}", liveFeatures[5]));
    output.push_back(std::format("Wind Speed (100m): {}", liveFeatures[6]));
    output.push_back(std::format("Wind Gusts (10m): {}", liveFeatures[7]));
    output.push_back(std::format("Boundary Layer Height: {}", liveFeatures[8]));

    // Normalize the live features in the same way as training.
    liveFeatures = arma::normalise(liveFeatures, 2, 0);

    // scaling the latitude and longitude features improves the prediction
    liveFeatures.row(0) *= 0.1; // Scale the latitude feature
    liveFeatures.row(1) *= 0.1; // Scale the longitude feature
    // Convert to an Armadillo matrix with one column.
    //
    // DEBUG
    //
    //arma::mat liveMat = liveFeatures;

    // Use the trained Random Forest to classify the live data.
    arma::Row<size_t> livePrediction;
    rf.Classify(liveFeatures, livePrediction);


  output.push_back(std::format("\nPrediction for the selected launch pad: "));
  if (livePrediction(0) == 0)
    output.push_back(std::format("Launch Likely"));
  else
    output.push_back(std::format("Launch Scrubbed"));
}


// FIXME: There is option to pick a scheduled launch this definitely won't work with current menu setup
// Option 2: Select a scheduled launch from The Space Devs API
void menuOption2(mlpack::RandomForest<>& rf, vector<string> output) {

    output.push_back(std::format("\nFetching upcoming launches from The Space Devs API..."));
    std::vector<json> launches = GetUpcomingLaunches();

    if (launches.empty()) {
        std::cout << "No scheduled launches found in the next 10 days." << std::endl;
        return;
    }

    std::cout << "\nUpcoming Launches:" << std::endl;
    for (size_t i = 0; i < launches.size(); ++i) {
        // For each launch, print details.
        std::string agency = launches[i].value("launch_service_provider", json()).value("name", "N/A");
        std::string rocket = launches[i].value("rocket", json()).value("configuration", json()).value("name", "N/A");
        std::string padName = launches[i].contains("pad") ? launches[i]["pad"].value("name", "N/A") : "N/A";
        std::string windowStart = launches[i].value("window_start", "N/A");
        std::cout << i + 1 << ") " << agency << " | " << rocket << " | Launch Pad: " << padName << " | " << "Window Start: " << windowStart << std::endl;
    }

    std::cout << "Enter the number of the launch to test (or 0 to go cancel): ";
    int launchChoice;
    std::cin >> launchChoice;
    if (launchChoice == 0) {
        return;
    }
    if (launchChoice < 1 || launchChoice > static_cast<int>(launches.size())) {
        std::cout << "Invalid choice. Returning to main menu." << std::endl;
        return;
    }
    // Get the selected launch.
    json selectedLaunch = launches[launchChoice - 1];
    // Extract pad coordinates from the selected launch.
    std::pair<double, double> padCoords = GetPadCoordinates(selectedLaunch);
    std::cout << "Selected launch pad coordinates: Latitude = " << padCoords.first << ", Longitude = " << padCoords.second << std::endl;

    std::string windowStart = selectedLaunch.value("window_start", "");
    if (windowStart.size() >= 16)
        windowStart = windowStart.substr(0, 16);

    // Fetch weather data for these coordinates.
    arma::vec liveFeatures = GetWeatherFeaturesForLocation(padCoords.first, padCoords.second, windowStart);
    if (liveFeatures.n_elem == 0) {
        std::cout << "Failed to retrieve weather data." << std::endl;
        return;
    }
    // Display raw weather data.
    std::cout << "\nCurrent Weather Forecast Data for Selected Launch Pad:" << std::endl;
    std::cout << "Longitude: " << liveFeatures(0) << std::endl;
    std::cout << "Latitude: " << liveFeatures(1) << std::endl;
    std::cout << "Dew Point: " << liveFeatures(2) << std::endl;
    std::cout << "Pressure: " << liveFeatures(3) << std::endl;
    std::cout << "Cloud Cover: " << liveFeatures(4) << std::endl;
    std::cout << "Wind Speed (10m): " << liveFeatures(5) << std::endl;
    std::cout << "Wind Speed (100m): " << liveFeatures(6) << std::endl;
    std::cout << "Wind Gust (10m): " << liveFeatures(7) << std::endl;
    std::cout << "Boundary Layer Height: " << liveFeatures(8) << std::endl;

    // Normalize the live features.
    arma::mat liveFeaturesMat = liveFeatures;
    liveFeaturesMat.row(0) *= 0.01; // Scale the latitude feature
    liveFeaturesMat.row(1) *= 0.01; // Scale the longitude feature
    liveFeatures = arma::normalise(liveFeaturesMat, 2, 0);

    // DEBUG
    //std::cout << "Scaled Lat: " << liveFeaturesMat(0) << std::endl;
    //std::cout << "Scaled Lon: " << liveFeaturesMat(1) << std::endl;


    // Run prediction using the trained Random Forest.
    arma::Row<size_t> livePrediction;
    rf.Classify(liveFeatures, livePrediction);

    std::cout << "\nPrediction for the selected scheduled launch: ";
    if (livePrediction(0) == 0)
        std::cout << "Launch Likely" << std::endl;
    else
        std::cout << "Launch Scrubbed" << std::endl;

}

// call GetUpcomingLaunches() and build a list of launches that returns
// a vector of formatted strings each with a list of launch details
// and the corresponding vector of JSON objects
std::tuple<std::vector<std::string>, std::vector<json>> BuildLaunchList() {
    std::vector<json> launches = GetUpcomingLaunches();
    std::vector<std::string> launchList;
    for (size_t i = 0; i < launches.size(); ++i) {
        std::string agency = launches[i].value("launch_service_provider", json())
                                 .value("name", "N/A");
        std::string rocket = launches[i].value("rocket", json())
                                 .value("configuration", json())
                                 .value("name", "N/A");
        std::string padName = launches[i].contains("pad") 
                                ? launches[i]["pad"].value("name", "N/A")
                                : "N/A";
        std::string windowStart = launches[i].value("window_start", "N/A");
        std::string formatted = std::to_string(i + 1) + 
            ") " + agency + " | " + rocket + " | Pad: " + padName +
            " | Window: " + windowStart;
        launchList.push_back(formatted);
    }
    return std::make_tuple(launchList, launches);
}

// take trained model and selected launch JSON and extract pad coords
// and launch window start time, then call GetWeatherFeaturesForLocation()
// to get the weather features for the launch pad at the time of launch
// and run the prediction using the trained model
std::string GetScheduledLaunchPrediction(mlpack::RandomForest<>& rf, const json& launch) {
    // get pad coords
    auto [lat, lon] = GetPadCoordinates(launch);
    // get launch window start time
    std::string windowStart = launch.value("window_start", "");
    if (windowStart.size() >= 16)
        windowStart = windowStart.substr(0, 16);
    
    // get forecast weather features
    arma::vec features = GetWeatherFeaturesForLocation(lat, lon, windowStart);

    //scale lat and lon to minimize impact on prediction
    features(0) *= 0.01; // Scale the latitude feature
    features(1) *= 0.01; // Scale the longitude feature

    // normalize in same way as training data
    features = arma::normalise(features, 2, 0);

    // run prediction
    arma::Row<size_t> prediction;
    arma::mat featureMat = features; // convert vector to matrix
    rf.Classify(featureMat, prediction);

    std::string predStr = (prediction(0) == 0) ? "Launch Likely" : "Launch Scrubbed";
    return "Scheduled lauinch at " + windowStart + ": " + predStr;

}











// Main function
// int main() {
//     try {
//         // Train the model and get the trained Random Forest.
//         mlpack::RandomForest<> rf = TrainModel();
//
//         // Main menu loop.
//         bool exitInterface = false;
//         while (!exitInterface) {
//             cout << "\n=== Main Menu ===" << endl;
//             cout << "1) Test Launch Pad (live weather)" << endl;
//             cout << "2) Select Scheduled Launch" << endl;
//             cout << "0) Exit" << endl;
//             cout << "Enter your choice: ";
//             int mainChoice;
//             cin >> mainChoice;
//             switch (mainChoice) {
//             case 0:
//                 exitInterface = true;
//                 break;
//             case 1:
//                 menuOption1(rf);
//                 break;
//             case 2:
//                 menuOption2(rf);
//                 break;
//             default:
//                 cout << "Invalid choice. Please try again." << endl;
//                 break;
//             }
//         }
//         cout << "Exiting program." << endl;
//     }
//     catch (const std::exception& ex) {
//         cerr << "Fatal error: " << ex.what() << endl;
//         return -1;
//     }
//     return 0;
// }
