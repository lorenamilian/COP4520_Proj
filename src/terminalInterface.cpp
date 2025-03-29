// TODO: build a tui
#include <iostream>
#include <string>
#include "ftxui/component/component.hpp"
#include "ftxui/component/screen_interactive.hpp"
#include "ftxui/dom/elements.hpp"

using namespace ftxui;

int main() {
    // The counter value
    int counter = 0;

    // Components
    Component increment_button = Button("Increment", [&] { counter++; });
    Component decrement_button = Button("Decrement", [&] { counter--; });
    Component quit_button = Button("Quit", screen.ExitLoopClosure());

    // Arrange components in a horizontal layout
    auto buttons = Container::Horizontal({
        increment_button,
        decrement_button,
        quit_button,
    });

    // Define the rendering function - how the UI will look
    auto renderer = Renderer(buttons, [&] {
        return vbox({
            text("FTXUI Counter Example") | bold | center,
            text("Value: " + std::to_string(counter)) | center,
            hbox({
                increment_button->Render(),
                separator(),
                decrement_button->Render(),
                separator(),
                quit_button->Render(),
            }),
        }) | border;
    });

    // Start the interactive screen
    auto screen = ScreenInteractive::TerminalOutput();
    screen.Loop(renderer);

    return 0;
}
