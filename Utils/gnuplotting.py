import numpy as np
import subprocess

# CMU Sans Serif
# CMU Bright



gnuplot_script1 = f"""
# Set output and terminal type
set terminal pngcairo enhanced font 'CMU Bright,14' size 1000,600
set output 'plot.png'

# Titles and labels
set title "Training Loss Over Time" font 'CMU Bright,18'
set xlabel "Iterations" font 'CMU Bright,14'
set ylabel "Loss Value" font 'CMU Bright,14'

# Grid and border styling
set grid lw 1 lc rgb "#BBBBBB"
set border lw 2 lc rgb "black"

# Line styles
set style line 1 lc rgb "#1f77b4" lw 3 pt 7 ps 1.5   # Blue line, thicker, with points
set style line 2 lc rgb "#ff7f0e" lw 3 dt 2          # Orange dashed line
set style line 3 lc rgb "#2ca02c" lw 3               # Green line

# Key/Legend settings
set key top right box enhanced font 'CMU Bright,12'

# Axis styling
set tics font 'CMU Bright,12'
set xtics rotate by -45
set ytics format "%.2f"  # Format Y-axis values with 2 decimal places

# Plot data
plot 'loss_data.dat' using 1:2 with lines linestyle 1 title "Loss Curve"
"""

gnuplot_script2 = f"""set terminal pdfcairo enhanced font 'CMU Sans Serif,12' size 6,4
set output 'minimalist_plot.pdf'

# Titles and labels
set title "Loss vs. Iterations" font 'CMU Sans Serif,14'
set xlabel "Iterations"
set ylabel "Loss"

# No grid, minimal border
unset grid
set border 3

# Simple line style
set style line 1 lc rgb "black" lw 2

# No legend
unset key

# Plot data
plot 'loss_data.dat' using 1:2 with lines linestyle 1"""


def gnuplot_losses(losses, output_file="loss_plot.png"):
    # Create a temporary data file
    data_file = "loss_data.dat"
    with open(data_file, "w") as f:
        for i, loss in enumerate(losses):
            f.write(f"{i} {loss}\n")
    
    # Gnuplot script to visualize the loss
    gnuplot_script = f"""
    set terminal pngcairo enhanced font 'CMU Bright,14' size 800,600
    set output '{output_file}'
    set title "Training Loss vs Iterations" font 'CMU Bright,18' textcolor rgb "black" fontweight "bold"
    set xlabel "Iterations" offset 0,-0.25
    set ylabel "Loss" offset 0,-0.25
    set grid
    unset key
    plot '{data_file}' using 1:2 with lines title "Training Loss" lw 2 linecolor rgb "blue"
    """
    #gnuplot_script = gnuplot_script2

    # Run the Gnuplot script
    process = subprocess.Popen(["gnuplot"], stdin=subprocess.PIPE)
    process.communicate(gnuplot_script.encode())

    print(f"Plot saved as {output_file}")

# Example usage
losses = np.random.rand(100) * 0.1  # Simulating 100 iterations of random loss values
gnuplot_losses(losses)

