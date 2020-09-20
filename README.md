# Notes

## Dataset


The whole dataset is available here: https://drive.google.com/open?id=1UlOk6EtiaXTHylRUx2mySgvJX9ycoeBp

It's gitignored due to the size but if you fetch this repo and then extract the zip at that link, then the paths will all match up

## Project status

I've got the accuracy to 0.48. That's terrible, but it's a start I suppose.

The network is composed of a series of convolutional layers, pooling layers, and dropout layers (to reduce overfitting).

There are 90k images in the dataset which are composed of pictures of the faces of various chinese celebrities.

There are also about 2k images of the same people wearing face masks. So far, I haven't attempted to tackle the face mask part.

## Running it

The file names are self explanatory, basically 

```bash
python3 generateTrainingData.py
```

and then

```bash
python3 trainModel.py
```

will get you going.

You can also do 

```bash
pip3 install -r requirements.txt
```

to get all the libraries you need.
