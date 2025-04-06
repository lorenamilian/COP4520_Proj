#include <ftxui/dom/flexbox_config.hpp>
#include <functional>
#include <memory>
#include <string>
#include <vector>
#include "ftxui/component/component.hpp"
#include "ftxui/component/component_base.hpp"
#include "ftxui/component/screen_interactive.hpp"
#include "ftxui/dom/elements.hpp"
#include "ftxui/screen/color.hpp"
#include "RocketLaunchPrediction.h"

using namespace ftxui;

// Enum to track current UI state
enum class UIState {
  MAIN_MENU,
  PREDICTION_SUBMENU,
  EVALUATION_SUBMENU
};

int main() {
  auto screen = ScreenInteractive::Fullscreen();
  std::tuple<mlpack::RandomForest<>, std::vector<std::string>> model_tuple;
  std::vector<std::string> training_output;
  mlpack::RandomForest<> rf;
  bool model_trained = false;

  // Track the current UI state
  UIState current_state = UIState::MAIN_MENU;

  // Main menu setup
  std::vector<std::string> menu_entries = {
    "Train Model",
    "Make Predictions",
    "Evaluate Model",
    "Quit"
  };
  int selected = 0;
  int menu_selector = 0;
  auto main_menu = Menu(&menu_entries, &selected);

  // Prediction submenu setup
  std::vector<std::string> launch_pad_entries = {
    "Cape Canaveral",
    "Vandenberg Air Force Base",
    "Baikonur Cosmodrome",
    "Back to Main Menu"
  };
  int sub_selected = 0;
  auto prediction_submenu = Menu(&launch_pad_entries, &sub_selected);

  // Output box for displaying results
  auto output_box = Renderer([&] {
    std::vector<Element> lines;
    for (const auto& line : training_output) {
      lines.push_back(text(line));
    }
    return vbox({
      text("Model Output") | center | color(Color::Green),
      separator(),
      vbox(std::move(lines)) | flex | vscroll_indicator | frame,
    });
  });

  // Container for selecting which menu to show
  auto menu_container = Container::Tab(
    {
      Renderer(main_menu, [&] {
        return vbox({
          text("MAIN MENU") | bold | center | color(Color::Yellow),
          separator(),
          main_menu->Render(),
          filler(),
        });
      }),

      Renderer(prediction_submenu, [&] {
        return vbox({
          text("PREDICTION SUBMENU") | bold | center | color(Color::Cyan),
          text("Select Launch Pad:") | center,
          separator(),
          prediction_submenu->Render(),
          filler(),
        });
      })
    },
    &menu_selector
    // Use the state to determine which tab is active
    // [&] { return current_state == UIState::MAIN_MENU ? 0 : 1; }
  );

  // Handle main menu selection
  auto menu_with_action = CatchEvent(menu_container, [&](Event event) {
    if (event == Event::Return) {
      // If we're in the main menu
      if (current_state == UIState::MAIN_MENU) {
        switch (selected) {
          case 0: // Train Model
            if (!model_trained) {
              model_trained = true;
              model_tuple = TrainModel();
              training_output = get<1>(model_tuple);
              rf = get<0>(model_tuple);
            } else {
              training_output.clear();
              training_output.push_back("Model is already trained");
            }
            break;

          case 1: // Make Predictions
            if (!model_trained) {
              training_output.clear();
              training_output.push_back("Error: Please train the model first.");
            } else {
              // Switch to prediction submenu
              training_output.clear();
              training_output.push_back("Select a launch pad for prediction");
              current_state = UIState::PREDICTION_SUBMENU;
              sub_selected = 0; // Reset submenu selection
            }
            break;

          case 2: // Evaluate Model
            if (!model_trained) {
              training_output.clear();
              training_output.push_back("Error: Please train the model first.");
            } else {
              training_output.clear();
              training_output.push_back("Evaluation mode selected");
              // Add evaluation functionality
            }
            break;

          case 3: // Quit
            screen.ExitLoopClosure()();
            break;
        }
      }
      // If we're in the prediction submenu
      else if (current_state == UIState::PREDICTION_SUBMENU) {
        if (sub_selected == launch_pad_entries.size() - 1) {
          // Last option is "Back to Main Menu"
          current_state = UIState::MAIN_MENU;
          training_output.clear();
        } else {
          // Handle launch pad selection
          training_output.clear();
          training_output.push_back("Selected launch pad: " + launch_pad_entries[sub_selected]);

          // Here you would add code to make predictions based on the selected launch pad
          // For example:
          if (model_trained) {
            training_output.push_back("Making predictions for " + launch_pad_entries[sub_selected] + "...");
            training_output.push_back("Weather conditions: Favorable");
            training_output.push_back("Success probability: 85%");
            // You could call a function like MakePrediction(rf, sub_selected) here
          }
        }
      }
      return true;
    }

    // Handle Escape key to return to main menu
    if (event == Event::Escape && current_state != UIState::MAIN_MENU) {
      current_state = UIState::MAIN_MENU;
      return true;
    }

    return false;
  });

  // Layout containers
  auto left_container = Container::Vertical({
    menu_with_action
  });

  auto right_container = Container::Vertical({
    output_box
  });

  auto main_container = Container::Horizontal({
    left_container,
    right_container
  });

  auto split_view = Renderer(main_container, [=] {
    return hbox({
      left_container->Render() | size(WIDTH, LESS_THAN, 30) | border,
      right_container->Render() | flex | border,
    });
  });

  screen.Loop(split_view);
}
