all: plot

count: 
	python wathamm.py

plot: count 
	python print_points.py
