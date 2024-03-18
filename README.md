# Silent-Speech-Sleuth (S3)

A few weeks ago, La Liga hired a professional lip reader to decide whether Jude Bellingham called Mason Greenwood a “rapist” after the two players clashed during a football (not soccer) match.
Now I am not going to look much into the details of the incident, instead what I am going to try is to build a deep learning model that is able to read lips since not many of us are lunatics (only lunatics can read lips), or has pockets deep enouhg for a pro lip reader (ours, yes, mine and yours are only as deep because they are empty).

Lip reading, a skill crucial for those with hearing impairments, has long been a challenging task due to its complexity and nuances. However, with the advent of advanced technologies like LipNet, a deep learning model designed to interpret lip movements, there is newfound hope for enhanced communication accessibility. LipNet showcases the potential of artificial intelligence in bridging communication gaps and empowering individuals with hearing impairments to engage more fully in conversations. As we delve deeper into the realm of assistive technologies, LipNet stands as a promising tool, offering a glimpse into a future where communication barriers are minimized, and inclusivity thrives.

I'll be using a range of technologies; OpenCV to read our videos and TensorFlow to build the model.

# Workflow

1. Build Data Loading Function
2. Create Data Pipeline
3. Design the Deep Neural Network
4. Setup Training Options and Train
5. Make a Prediction
6. Test on a Sample Video
7. Create an Interface for Realtime Interaction With The Model


### Data Loading Function

This code performs several tasks related to video processing and alignment loading. Let's break it down step by step:

1. **Downloading and extracting data**:
   - The code begins by importing the `gdown` library. This library is used to download files from Google Drive.
   - A URL pointing to a zip file is provided, and it is downloaded using `gdown.download()`.
   - Then, `gdown.extractall()` is called to extract the contents of the downloaded zip file.

2. **Loading video frames** (`load_video` function):
   - This function takes a path to a video file as input and returns a list of preprocessed video frames.
   - It uses OpenCV (`cv2`) to read the video file and extract frames.
   - Each frame is converted to grayscale using TensorFlow's `tf.image.rgb_to_grayscale()` function.
   - Specific regions of interest (ROI) within each frame are extracted using array slicing.
   - The pixel values of the frames are standardized by subtracting the mean and dividing by the standard deviation.
   - The preprocessed frames are returned as a TensorFlow float tensor.

3. **Defining character vocabulary**:
   - A vocabulary list containing lowercase letters, some special characters, and digits is created.
   - Two string lookup layers (`char_to_num` and `num_to_char`) are defined using TensorFlow's `StringLookup` layer. These layers map characters to numbers and vice versa.

4. **Loading alignments** (`load_alignments` function):
   - This function takes a path to an alignment file as input and returns a list of aligned characters.
   - It reads the alignment file, extracts relevant tokens, and converts them into character sequences.
   - The character sequences are converted into numerical representations using `char_to_num`.

5. **Loading data** (`load_data` function):
   - This function takes a path to a video file and loads both video frames and alignment data.
   - It first extracts the filename from the provided path.
   - Then, it constructs paths for both the video file and the alignment file.
   - Video frames are loaded using the `load_video` function, and alignments are loaded using the `load_alignments` function.
   - Finally, it returns the frames and alignments.

6. **Testing the data loading**:
   - A test path to a video file is provided.
   - The `load_data` function is called with this test path, and the frames and alignments are obtained.

7. **Visualization and processing**:
   - The code visualizes a specific frame (`plt.imshow(frames[40])`).
   - It also converts alignment data back to strings using `num_to_char` and `tf.strings.reduce_join()`.

8. **Mapping function** (`mappable_function`):
   - This function is defined to be used with TensorFlow's `Dataset.map()` function.
   - It takes a path as input and returns the result of calling `load_data` on that path.
This is a pipeline for loading video data, processing frames, handling alignments, and preparing them for further processing, likely for tasks like speech recognition or similar applications.


### The Data Pipeline

```python
from matplotlib import pyplot as plt

data = tf.data.Dataset.list_files('./data/s1/*.mpg')
data = data.shuffle(500, reshuffle_each_iteration=False)
data = data.map(mappable_function)
data = data.padded_batch(2, padded_shapes=([75,None,None,None],[40]))
data = data.prefetch(tf.data.AUTOTUNE)
# Added for split 
train = data.take(450)
test = data.skip(450)

len(test)

frames, alignments = data.as_numpy_iterator().next()

len(frames)

sample = data.as_numpy_iterator()

val = sample.next(); val[0]

imageio.mimsave('./animation.gif', val[0][0], fps=10)

# 0:videos, 0: 1st video out of the batch,  0: return the first frame in the video 
plt.imshow(val[0][0][35])



```
### Designing the Deep Neural Network

```python

```
### Setup Training Options and Train

```python

```
### Make a Prediction

```python

```
### Test on a Sample Video

```python

```
### Create an Interface for Realtime Interaction With The Model

```python

```

## Usefull Resources

https://www.tensorflow.org/api_docs/python/tf/data

Original Paper: https://arxiv.org/abs/1611.01599 

Associated Code for Paper: https://github.com/rizkiarm/LipNet 

ASR Tutorial: https://keras.io/examples/audio/ctc_asr/#model


