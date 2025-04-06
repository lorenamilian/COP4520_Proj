#include <ftxui/component/component.hpp>
#include <ftxui/component/screen_interactive.hpp>
#include <ftxui/dom/elements.hpp>
#include <ftxui/screen/color.hpp>
#include "RocketLaunchPrediction.h"
#include <mlpack/methods/random_forest/random_forest.hpp>
#include <tuple>
#include <vector>
#include <string>

using namespace ftxui;

// Enum to track current UI state
enum class UIState {
  MAIN_MENU,
  PREDICTION_SUBMENU
};

int main() {
  auto screen = ScreenInteractive::Fullscreen();
  std::tuple<mlpack::RandomForest<>, std::vector<std::string>> model_tuple;
  std::vector<std::string> model_output;
  mlpack::RandomForest<> rf;
  bool model_trained = false;

  int selected_tab = 0;

  std::vector<std::string> menu_entries = {
    "Train Model",
    "Make Predictions",
    "Evaluate Model",
    "Quit"
  };
  int main_menu_selected = 0;
  auto main_menu = Menu(&menu_entries, &main_menu_selected);

  std::vector<std::string> launch_pad_entries = {
    "Cape Canaveral",
    "Vandenberg Air Force Base",
    "Baikonur Cosmodrome",
    "Back to Main Menu"
  };
  int prediction_selected = 0;
  auto prediction_menu = Menu(&launch_pad_entries, &prediction_selected);

  // Renderer for the output box
  auto output_box = Renderer([&] {
    std::vector<Element> lines;
    for (const auto& line : model_output) {
      lines.push_back(text(line));
    }
    return vbox({
      text("Model Output") | center | color(Color::Green),
      separator(),
      vbox(std::move(lines)) | flex | vscroll_indicator | frame,
    });
  });

  // Menu container using Tab switching
  auto menu_container = Container::Tab({
    Renderer(main_menu, [&] {
      return vbox({
        text("MAIN MENU") | bold | center | color(Color::Yellow),
        separator(),
        main_menu->Render(),
        filler(),
      });
    }),
    Renderer(prediction_menu, [&] {
      return vbox({
        text("PREDICTION SUBMENU") | bold | center | color(Color::Cyan),
        text("Select Launch Pad:") | center,
        separator(),
        prediction_menu->Render(),
        filler(),
      });
    })
  }, &selected_tab);

  // Handle tab key events and menu selections
  auto menu_with_action = CatchEvent(menu_container, [&](Event event) {
    if (event == Event::Tab && model_trained) {
      selected_tab = (selected_tab + 1) % 2;
      return true;
    }
    if (event == Event::TabReverse) {
      selected_tab = (selected_tab - 1 + 2) % 2;
      return true;
    }
    if (event == Event::Return) {
      if (selected_tab == 0) {
        switch (main_menu_selected) {
          case 0:
            if (!model_trained) {
              model_trained = true;
              model_tuple = TrainModel();
              model_output = get<1>(model_tuple);
              rf = get<0>(model_tuple);
            } else {
              model_output.clear();
              model_output.push_back("Model is already trained");
            }
            break;
          case 1:
            if (!model_trained) {
              model_output.clear();
              model_output.push_back("Error: Please train the model first.");
            } else {
              model_output.clear();

              selected_tab = 1;
            }
            break;
          case 2:
            if (!model_trained) {
              model_output.clear();
              model_output.push_back("Error: Please train the model first.");
            } else {
              model_output.clear();
              model_output.push_back("Evaluation mode selected");
            }
            break;
          case 3:
            screen.ExitLoopClosure();
            break;
        }
      } else if (selected_tab == 1) {
        if (prediction_selected == launch_pad_entries.size() - 1) {
          selected_tab = 0;
          model_output.clear();
        } else {
          model_output.clear();
          model_output.push_back("Selected launch pad: " + launch_pad_entries[prediction_selected]);
          if (model_trained) {
            model_output.push_back("Making predictions for " + launch_pad_entries[prediction_selected] + "...");
            menuOption1(rf, prediction_selected, model_output);
          }
        }
      }
      return true;
    }
    return false;
  });

  // Layout containers
  auto left_container = Container::Vertical({ menu_with_action });
  auto right_container = Container::Vertical({ output_box });
  auto main_container = Container::Horizontal({ left_container, right_container });

  auto split_view = Renderer(main_container, [=] {
    return hbox({
      left_container->Render() | size(WIDTH, LESS_THAN, 30) | border,
      right_container->Render() | flex | border,
    });
  });

  screen.Loop(split_view);
}
