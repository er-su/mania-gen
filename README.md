This project is for learning purposes!

The purpose of this program is to create a model to generate beatmaps in the rhythm game osu! mania from a given audio file (in addition to bpm and offset). The model 
uses a convolutional neural network with mel spectrograms combined with extra contextual information to predict where musical onsets occur. This is then passed into
a recurrent neural network to convert onsets to combinations of key presses.
