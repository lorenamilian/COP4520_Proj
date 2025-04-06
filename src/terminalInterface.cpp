#include <algorithm>
#include <ftxui/component/component.hpp>
#include <ftxui/component/screen_interactive.hpp>
#include <ftxui/dom/elements.hpp>
#include <ftxui/screen/color.hpp>
#include "RocketLaunchPrediction.h"
// #include <future>
#include <mlpack/methods/random_forest/random_forest.hpp>
#include <random>
#include <tuple>
#include <vector>
#include <string>
#include <nlohmann/json.hpp>

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
  std::tuple<std::vector<std::string>, std::vector<json>> tuple_tmp = BuildLaunchList();
  std::vector<nlohmann::json> json;


  int selected_tab = 0;

  std::vector<std::string> menu_entries = {
    "Train Model",
    "Make Predictions",
    "Get Upcoming Launches",
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

  std::vector<std::string> upcoming_launch_entries = get<0>(tuple_tmp);
  upcoming_launch_entries.push_back("Back to Main Menu");

  int launch_selected = 0;
  auto launch_menu = Menu(&upcoming_launch_entries, &launch_selected);

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
        text("SELECT PREDICTION") | bold | center | color(Color::Cyan),
        separator(),
        prediction_menu->Render(),
        filler(),
      });
    }),
    Renderer(launch_menu, [&] {
      return vbox({
        text("SELECT LAUNCH") | bold | center | color(Color::OrangeRed1),
        separator(),
        launch_menu->Render(),
        filler(),
      });
    })
  }, &selected_tab);

  // Handle tab key events and menu selections
  auto menu_with_action = CatchEvent(menu_container, [&](Event event) {
    if (event == Event::Tab) {
      selected_tab = (selected_tab + 1) % 3;
      return true;
    }
    // FIXME: accessing the prediction menu should without trained model should be removed before release
    if (event == Event::TabReverse) {
      selected_tab = (selected_tab - 1 + 3) % 3;
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
              model_output.push_back("WARN: Predictions unavailable Model must be trained.");
              selected_tab = 2;
            } else {
              model_output.clear();
              json = get<1>(tuple_tmp);
              selected_tab = 2;
            }
            break;
          case 3:
            screen.ExitLoopClosure()();
            break;
        }
      } else if (selected_tab == 1) {
        if (prediction_selected == launch_pad_entries.size() - 1) {
          selected_tab = 0;
          model_output.clear();
        } else {
          model_output.clear();
          // model_output.push_back("Selected launch pad: " + launch_pad_entries[prediction_selected]);
          if (model_trained) {
            model_output.push_back("Making predictions for " + launch_pad_entries[prediction_selected] + "...");
            menuOption1(rf, prediction_selected, model_output);
          }
        }
      } else if (selected_tab == 2) {
        if (launch_selected == upcoming_launch_entries.size() - 1) {
          selected_tab = 0;
          model_output.clear();
        } else {
          if (model_trained) {
            model_output.push_back("Making predictions for " + upcoming_launch_entries[launch_selected] + "...");
            model_output.push_back(GetScheduledLaunchPrediction(rf, json[launch_selected]));
          } else {
            model_output.push_back("WARN: Predictions unavailable Model must be trained.");
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

 Decorator left_con_size;
  if (screen.dimx() < 180) {
    left_con_size = size(WIDTH, ftxui::LESS_THAN, 30);
  } else {
    left_con_size = size(WIDTH, ftxui::GREATER_THAN, 180);
  }

  auto split_view = Renderer(main_container, [=] {
    return hbox({
      left_container->Render() | left_con_size | border,
      right_container->Render() | flex | border,
    });
  });

  screen.Loop(split_view);
}
