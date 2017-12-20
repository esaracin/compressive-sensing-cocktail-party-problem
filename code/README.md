# Cocktail Party Problem Code

## Dependencies
The following Python packages are utilized in the scripts provided in the 'code/' subdirectory. Python 3 is expected to run all code:

	logging: to provide useful debugging information
	numpy: for the ease of the many of the mathematical computations	 
	scipy: for sparse matrix representation, as well as the reading in and writing of .wav files
	sklearn: for k-means clustering as well as Orthogonal Matching Pursuit

## Development

## Usage
reconstruct_signal.py:
	Main utility script for reading in a mixed signal and writing its separated and reconstructed output signals to the local directory. Take 3 parameters:

		path/to/soundFile.wav: a relative or absolute path to a mixed sound .wav file to be read in and separated
		numSources: the number of distinct sources in that sound file (i.e. the number of distinct output files to be written containing separated audio)
		windowSize: the size of the patches used to reconstruct the signals during the OMP portion of the code.

		python3 reconstruct_signal.py ../data/mixes/eastwood_graham_mix.wav 2 500		
