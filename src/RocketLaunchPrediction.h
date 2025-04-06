// RocketLaunchPrediction.h : Include file for standard system include files,
// or project specific include files.
#include <mlpack/core.hpp>
#include <mlpack/methods/random_forest/random_forest.hpp>
#include <mlpack/core/data/split_data.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
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
#pragma once

#include <iostream>

// TODO: Reference additional headers your program requires here.
mlpack::RandomForest<> TrainModel();
