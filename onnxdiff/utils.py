from enum import Enum
from tabulate import tabulate
from dataclasses import dataclass
from colorama import init as colorama_init
from colorama import Fore

colorama_init()


@dataclass
class SummaryResults:
    exact_match: bool  # The input models are exactly the same.
    score: float  # Graph kernel score to estimate shape similarity.
    a_valid: bool  # True when model A passes ONNX checker.
    b_valid: bool  # True when model B passes ONNX checker.
    graph_matches: dict  # Items exactly the same, for all fields in graph.
    root_matches: dict  # Items exactly the same, for the fields in root (excluding the graph)

class Status(Enum):
    Success = 0
    Warning = 1
    Error = 2

color_map = {
    Status.Success: Fore.GREEN,
    Status.Warning: Fore.YELLOW,
    Status.Error: Fore.RED,
}

def color(text: str, status: Status) -> str:
    return f"{color_map[status]}{text}{Fore.RESET}"


def matches_string(count: int, total: int):
    text = f"{count}/{total}"
    status = Status.Success if count == total else Status.Error
    return color(text=text, status=status)


def print_summary(results: SummaryResults) -> None:
    text = (
        "Exact Match"
        if results.exact_match and results.score == 1.0
        else "Difference Detected"
    )
    print("")
    print(f" {text} ({round(results.score * 100, 6)}%)")
    print("")

    data = []
    for key, matches in results.graph_matches.items():
        data.append(
            [
                f"Graph.{key.capitalize()}",
                matches_string(matches.same, matches.a_total),
                matches_string(matches.same, matches.b_total),
            ]
        )
    for key, matches in results.root_matches.items():
        data.append(
            [
                f"{key.capitalize()}",
                matches_string(matches.same, matches.a_total),
                matches_string(matches.same, matches.b_total),
            ]
        )
    print(tabulate(data, headers=["Matching Fields", "A", "B"], tablefmt="rounded_outline"))
