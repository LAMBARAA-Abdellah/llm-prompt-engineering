"""
Beautiful console output utilities using Rich library.

Provides banners, tables, and formatted sections for CLI display.
"""

from __future__ import annotations

from typing import Any, Dict, List

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich import box

console = Console()


def print_banner(title: str, subtitle: str = "") -> None:
    """Display a styled project banner."""
    text = f"[bold cyan]{title}[/bold cyan]"
    if subtitle:
        text += f"\n[dim]{subtitle}[/dim]"
    console.print(Panel(text, border_style="cyan", box=box.DOUBLE))


def print_section(title: str) -> None:
    """Print a section header."""
    console.print(f"\n[bold yellow]{'=' * 60}[/bold yellow]")
    console.print(f"[bold white]{title}[/bold white]")
    console.print(f"[bold yellow]{'=' * 60}[/bold yellow]\n")


def print_results_table(
    title: str,
    headers: List[str],
    rows: List[List[Any]],
) -> None:
    """Render a formatted results table."""
    table = Table(title=title, box=box.ROUNDED, show_header=True, header_style="bold magenta")
    for header in headers:
        table.add_column(header, justify="center")
    for row in rows:
        table.add_row(*[str(cell) for cell in row])
    console.print(table)


def print_metrics_summary(metrics: Dict[str, float]) -> None:
    """Display evaluation metrics in a panel."""
    lines = "\n".join(f"  {k}: [green]{v:.4f}[/green]" for k, v in metrics.items())
    console.print(Panel(lines, title="Evaluation Metrics", border_style="green"))


def create_progress_bar(description: str = "Processing") -> Progress:
    """Return a configured Rich progress bar."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    )
