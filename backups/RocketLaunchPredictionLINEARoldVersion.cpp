
#include <mlpack/core.hpp>
#include <mlpack/methods/logistic_regression/logistic_regression.hpp>
#include <iostream>
#include <mlpack/core/data/split_data.hpp>

int main()
{
    using namespace mlpack;
    // Load dataset from CSV.
    arma::mat data;
    mlpack::data::DatasetInfo info;



    try {
        // Transpose the data to get features as rows, samples as columns
        mlpack::data::Load("data.csv", data, info, true);
    }
    catch (const std::exception& e) {
        std::cerr << "Error loading data: " << e.what() << std::endl;
        return -1;
    }

    std::cout << "Loaded dataset dimensions (features x examples): "
        << data.n_rows << " x " << data.n_cols << std::endl;






    // Print label mappings
    if (info.Type(0) == mlpack::data::Datatype::categorical)
    {
        std::cout << "=== Label Mappings ===\n";
        for (size_t i = 0; i < info.NumMappings(0); ++i)
        {
            std::cout << "Class " << i << ": "
                << info.UnmapString(0, i) << "\n";
        }
    }





    // 1. Extract the Label.
    // (row 0) is our label (Status: S or F).
    // mlpack has converted these categorical values to integers.

    arma::Row<size_t> labels = arma::conv_to<arma::Row<size_t>>::from(data.row(0));

    if (info.Type(0) == data::Datatype::categorical)
        {
            std::cout << "Label (column 0) is categorical with "
                << info.NumMappings(0) << " mappings." << std::endl;
            // For example, if the mapping is alphabetical, then "F" might map to 0 and "S" to 1.
        }


     // 2. Extract the Numerical Weather Features.
     // We want to ignore the non-numeric reference columns (Provider, Rocket, Date, Time)
     // which are rows 1-4. The numerical weather data are in rows 5 to 13.
    if (data.n_rows < 14)
    {
        std::cerr << "Error: Expected at least 14 rows in the data, got "
            << data.n_rows << std::endl;
        return -1;
    }
    arma::mat features = data.submat(5, 0, 13, data.n_cols - 1);



    // (Optional) Normalize the numerical features.
    features = arma::normalise(features, 2, 0);  // Normalize each column (example using L2 norm).



     // 3. Split the Data into Training and Test Sets.
    arma::mat trainFeatures, testFeatures;
    arma::Row<size_t> trainLabels, testLabels;
    // Use 20% of the data for testing.
    data::Split(features, labels, trainFeatures, testFeatures, trainLabels, testLabels, 0.2);



    // 4. Train a Logistic Regression Model.
    // Note: Logistic regression in mlpack requires numeric features.
    // Our features are now the numerical weather data.
    mlpack::LogisticRegression<> lr;
    lr.Train(trainFeatures, trainLabels);


    // 5. Make predictions
    arma::Row<size_t> predictions;
    lr.Classify(testFeatures, predictions);



    // 6. Detailed evaluation
    std::cout << "\n=== Evaluation Metrics ===\n";


    // Basic accuracy.
    double accuracy = arma::accu(predictions == testLabels) / double(testLabels.n_elem);
    std::cout << "Accuracy: " << accuracy << std::endl;



    // Confusion matrix
    arma::Mat<size_t> confusion(2, 2, arma::fill::zeros);
    for (size_t i = 0; i < testLabels.n_elem; ++i)
    {
        confusion(testLabels(i), predictions(i))++;
    }




    std::cout << "\nConfusion Matrix:\n"
        << "               Predicted 0  Predicted 1\n"
        << "Actual 0       " << confusion(0, 0) << "           " << confusion(0, 1) << "\n"
        << "Actual 1       " << confusion(1, 0) << "           " << confusion(1, 1) << "\n";




    // Precision/Recall
    double precision = confusion(1, 1) / double(confusion(0, 1) + confusion(1, 1));
    double recall = confusion(1, 1) / double(confusion(1, 0) + confusion(1, 1));
    double f1 = 2 * (precision * recall) / (precision + recall);

    std::cout << "\nPrecision: " << precision
        << "\nRecall: " << recall
        << "\nF1 Score: " << f1 << "\n";




    // 7. Print sample predictions
    std::cout << "\n=== Sample Predictions ===\n";
    size_t num_samples_to_show = 10;
    for (size_t i = 0; i < num_samples_to_show && i < testLabels.n_elem; ++i)
    {
        std::cout << "Sample " << i << ": "
            << "Predicted: " << predictions(i)
            << ", Actual: " << testLabels(i)
            << " -> " << (predictions(i) == testLabels(i) ? "CORRECT" : "WRONG")
            << "\n";
    }

    return 0;
}
