#include <ftxui/dom/elements.hpp>
#include <ftxui/component/screen_interactive.hpp>
#include <ftxui/component/component.hpp>
#include <ftxui/component/component_options.hpp>
#include <vector>

using namespace ftxui;

int main() {

  auto screen = ScreenInteractive::TerminalOutput();

  std::vector<std::string> entries = {
    "Entry 1",
    "Entry 2",
    "Entry 3"
  };

  int selected = 0;

  MenuOption option;
  option.on_enter = screen.ExitLoopClosure();
  auto menu = Menu(&entries, &selected, option);

  screen.Loop(menu);

  std::cout << "Selected element = " << selected << std::endl;

  return EXIT_SUCCESS;
}

