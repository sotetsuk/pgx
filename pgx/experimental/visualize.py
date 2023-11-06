from typing import List, Union

from pgx.core import State


def show_state_seq(
    states: Union[State, List[State]],
) -> None:
    """Visualize list of states on notebook."""
    import sys

    if "ipykernel" in sys.modules:
        # Jupyter Notebook

        if isinstance(states, list):
            _show_states_in_widgets(
                states=states,
            )

        else:
            assert isinstance(states, State)
            from IPython.display import display_svg  # type:ignore

            display_svg(
                states._repr_html_(),
                raw=True,
            )
    else:
        # Not Jupyter
        sys.stdout.write("This function only works in Jupyter Notebook.")


def _show_states_in_widgets(
    states: List[State],
):
    import ipywidgets as widgets  # type:ignore
    from IPython.core.display import display_svg
    from IPython.display import display

    N = len(states)
    i = [-1]  # TODO: fix me using global

    def _on_click(button: widgets.Button):
        output.clear_output(True)
        with output:
            if button.description == "next":
                i[0] = (i[0] + 1) % N
            else:
                i[0] = (i[0] - 1) % N
            print(i[0])
            display_svg(
                states[i[0]]._repr_html_(),
                raw=True,
            )

    button1 = widgets.Button(description="next")
    button1.on_click(_on_click)

    button2 = widgets.Button(description="back")
    button2.on_click(_on_click)

    output = widgets.Output()
    box = widgets.Box([button2, button1])

    display(box, output)
    button1.click()
