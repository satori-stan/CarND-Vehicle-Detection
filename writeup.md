## Project Writeup

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_not_car.png
[image2]: ./output_images/HOG_example.png
[image3]: ./output_images/sliding_windows.png
[image4]: ./output_images/bboxes_and_heat.png
[image5]: ./output_images/labels_map.png
[image6]: ./output_images/output_bboxes.png
[video1]: ./project_video_out.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

This is it! I will eventually migrate this writeup file to the repo's README for a cleaner structure.

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The HOG feature extraction function is `get_hog_features` in cell 5. The code is the function defined in the lesson quiz, with two branches to return or not an image representation. It uses skimage's hog function as its basis.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

The images are read using matplotlib since it outputs an RGB image (the same format as moviepy, later used to process the video). There is no file to get the labels from, so the images are labelled based on their location in the file system (either inside the 'vehicles' folder or not). A few key values are extracted, like the data type and the maximum value of an image to get a sense of the data we will be working with, as well as the total number of examples per class, to make sure our dataset is balanced.

The datasets were then joined and shuffled to attempt to reduce overfitting (cell 3) since some of the training images are in a time series. 

Functions are defined to standardize the value range of the input, convert between colorspaces, calculate color histograms, build an object template and extract HOG features (cell 5).

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YUV` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and my selected values display what I think is a rather clear outline of a vehicle and very differentiable from non-vehicles. I actually am using a single channel for the HOG (Y) which can be obtained from "YUV" and "YcrCb" color spaces. I finally settled in "YUV" since I felt like it was more descriptive for the histograms and color binning, which means a single image and color conversion can be used for all the feature extractions.

More orientations did nothing to improve the result. Less orientations started to loose expressiveness. A similar thing happened when the pixels per cell were increased to 16: a larger value leads to less expressive outlines, smaller values caused the code to be extremely slow.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained an SVM with a polynomial kernel with C=1 and gamma=0.1. I arrived at these parameters after performing a grid search of parameter values (which ran for 86k seconds on an AWS instance) with my features.

Before training, the dataset is normalized and split 80-20 into train and test sets (with shuffling). When validating with the test set, the accuracy is above 97%.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I based my code on the lesson's code for sliding window with sub-sampling to reduce the overhead of extracting the HOG features on each window. Instead of returning the image with the boxes painted, I am returning the boxes in order to allow for joining the results of more than one search and facilitate the creation of a heatmap. The code to slide the windows is in function `find_cars` in cell 14.

I tested from one to three searches per image with different cropping (ystart/ystop) values, scale and cells per step (which relates to the amount of overlap). Here are the values I am using:

| Scale | Y Start | Y Stop | Cells/Step |
|-------|---------|--------|------------|
| 3     | 400     | 680    | 12         |
| 2     | 400     | 600    | 8          |
| 1     | 380     | 500    | 4          |

The logic for the top and bottom cropping is that closer vehicles are larger so they need a larger space, as the scale of the search becomes smaller, so does the space we need to look in since smaller sized vehicles would only appear closer to the top of the image.

Bigger cars in the video are easier to find, so we can overlap less, as the search window becomes smaller, we need to be more thorough in our search.

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on three scales using YUV single channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image3]

The way I ended up optimizing the performance was running each search (at different scales) in a separate thread. To do this, I used the ThreadPool class from the multiprocessing standard library. At first, I increased the pixels per cell in the HOG feature extractor, but this produced outlines that were not clear enough and decided to keep the default of 8 (as in the lesson). The improvement didn't become significant until I added more searches and while it is still under the desired speed (I blame it on my hardware), running three searches per frame took as long as running a single search.

I think there is still room for improvement. I am certain that parts of the code can be vectorized to make the code more efficient instead of just throwing hardware at it.

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my video result](./project_video_out.mp4). One key thing to consider is that when processing the video, images must first be undistorted, which requires reading in the chessboard images and using `cv2.calibrateCamera` to get an our distortion correction parameters.


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

When each frame is processed, the coordinates of the windows where a vehicle is found are returned to the calling function. The function (`process_image_w_heatmaps`) has a six frame memory. The pixels inside the bounding boxes add one intensity point to each of the memory frames. The oldest frame in the memory is thresholded to identify the areas that are most likely cars. The resulting image is passed to the `label` scipy function in order to distinguish each car in the image and then a bounding box is calculated around each labeled patch.

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding thresholded heatmaps:

![alt text][image4]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image5]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image6]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I have a number of doubts about the usability of the techniques in a real-life scenario. I remember a quote that says that humans are classification machines. We subconsciously classify everything we see (the keyword being subconsciously) and the procedures used in this project consume quite a lot of resources. That being said, I do understand that the main point is to understand what happens under the hood when we use neural network classifiers.

I would like to explore for example, what happens when we have vehicles that mostly obstruct our view (trucks for instance). For efficiency, we are defining ranges in which to search for vehicles, but it may be the case that one vehicle is all we see.

Also relevant are vehicles on the other side of the road. Not for the main project video, mind you, but at least one of the challenge videos is for a two way road. All the training examples used are for the backs of cars.

Furthermore, in night time situations we may only have head and tail lights to guide us. All the HOG features to tell a car apart by its shape will very much become useless.

Using more examples to train the classifier would definitively help with identifying other vehicle types and those that are in different directions of travel. It may very well get rid of some of my false positives. Implementing a procedure to read pre-labeled pictures like those in the Udacity datasets would be very valuable indeed. In real life situations I would expect to have a way to mark videoclips for review when a car is unsure of a classification.

Additionally, training an SVM is hard work with figuring out the set of parameters to use. An automatic grid search with 5 variations of each parameter resulted in a search of 24 hours in an AWS instance! That is, even with a three process parallel search it still took a significant amount of time... I couldn't find any parameters to use the GPU instead of the CPU for this calculation. There is an SVM class in TensorFlow, so it might be a better alternative.

