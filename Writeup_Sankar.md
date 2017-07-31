## Behavioral Cloning Project
---

The primary goals of this project are as follows:
* Use the udacity provided simulator to collect data of good driving behavior
* Build a convolution neural network in Keras that mimics good driving behavior
* Based on image obtained from center, left and right cameras, steer angle is predicted
* Test that the model successfully drives car around the track without leaving the road in fully autonomous fashion

[//]: # (Image References)

[image1]: ./examples/FlipDemonstrate.PNG "ImageAugmentation"
[image2]: ./examples/NvidiaCNNarch.PNG "NvidiaCNNArchitecture"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

**Description of files**

* model.ipynb contains the script to create and train the model
* model.h5 contains the trained convolution neural network
* drive.py for driving the car in autonomous mode on udacity simulator
* MyRun.mp4 is a video demonstrating the car driving in fully autonomous mode using the convolutional network values contained in model.h5

**Building the driver model**
---

As a first step, the car was driven in training mode using the Udacity simulator to record data for optimal driving behavior. Optimal in this case is to try to keep the car in the center of the lane as much as possible. Not being a gamer myself plus not having a joystick, it was actually very hard to drive around the test track being in the center of the lane :-) 

A sample training data set was provided by Udacity. Since bad training data would result in a garbage in - garbage out case and to get to the crux of the problem, I  relied on the Udacity provided data-set for the project. The dataset primarily consists of two files:

1. An IMG directory that captures left, center and right camera images mounted on the car while driving around the track
2. A drivinglog.csv file that includes path to images captured above along with measurements like steer angle, brake position, pedal and vehicle speed

As mentioned before, this project primarily focuses on using an image to predict what the steer angle needs to be. Simply put, 

```sh
X_train=images
Y_train=steering_angle
```

**Data Augmentation**

While initial testing with center images alone was done to verify basic functionality, it became obvious that more training data is needed to make the network predict driving behavior better. A simple way to do that was to use images from all cameras. 

While images from left and right camera were being analyzed, a small correction factor indicating steer angle that will drive it to the center was added. For example, for image from a left image, the correction factor for steer would be one that will turn the car slightly to the right. Code that performed this correction is shown below

```sh

for line in lines:
    for i in range(3):
        source_path=line[i]
        filename=source_path.split('/')[-1]
        current_path='/home/carnd/P3_sankar/myData/data/IMG/'+ filename
        image=cv2.imread(current_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if image == None:
            print("Invalid image:" , current_path)
        else:
            images.append(image)
            measurement = float(line[3])
            if i==1:
                measurements.append(measurement+0.2) ## for a left image, steer right a bit
            elif i==2:
                measurements.append(measurement-0.2) ## for a right image, steer left a bit
            else:
                measurements.append(measurement)   ## for a center image, do nothing     
```

An interesting thing to note was the driving direction on the track - clockwise vs anti-clockwise could bias the steer towards left or right. It is important to add this data to the training set to help the network perform better. One way to acquire this data is to actually drive around the track in anti-clockwise fashion. Cv2 has a very useful feature in the "flip" method that performs the same task in software. The figure below demonstrates use of this technique. The steer angle being a mirror image could simply be negated to create the correct label.

```sh
for image,measurement in zip(images,measurements):
    images_aug.append(image)    
    measurements_aug.append(measurement)
    images_aug.append(cv2.flip(image,1))
    measurements_aug.append(measurement*-1.0)
```

![alt text][image1]


In total, the baseline data set size was 24108. With image augmentation via the flip technique, the size doubled to 48216. This was sufficient to train the network.

**Network Architecture**
---

Various network architectures were tested all the way from simple linear models to slightly complex architectures via convolutions. Nvidia published a paper that details their convolutional network architecture for mimicing human behavior.

https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf

The architecture had the non-linearities needed for solving the problem and being tried and tested, I implemented this architecture for the problem. 

**Pre-Processing**

A "lambda" layer was added to normalize the image before training the model. The lambda layer in keras is essentially similar to adding python code that does the normalizing. An important advantage to using the lambda layer is that while testing the network on validation images, it goes through the same pre-processing without having to explicitly pre-process the feed images again. Normalizing code shown below:

```sh
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
```

In addition to normalizing, image was also cropped to remove surrounding environment data that just added to noise. The base input image was of shape 160x320x3 (RGB). The keras cropping2D function was used to reduce the image down to 70x25x3. 

```sh
model.add(Cropping2D(cropping=((70,25),(0,0))))
```

The Nvidia CNN architecture is shown below.

![alt text][image1]

The architecture was implemented in keras. 

```sh
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
```

Post normalizing, the network consists of 4 convolutional layers and 3 fully connected layers. The training parameters chosen were:

```sh
Optimizer=Adam with no specific learning rate
Loss Function='mse' as in mean squared error
validation split=0.2
Num of epochs=3

```
All the training was done on Amazon Web Sever using a GPU. Therefore, no generators were used. If done on a local machine without GPU, generators would have been necessary.

**Final Results**
---

The trained model parameters were saved and transferred to the local machine. By using the drive.py function and using "autonomous mode" on the simulator, the car was driven on the track using network predicted steer angles. The results were good and the vehicle did not leave the track even once. 

The video.py function was used to create the "MyRun".mp4 video. The frames per second parameter was set at 30.














####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py lines 18-24) 

The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 18). 

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
