## Project Overview

Welcome to the Convolutional Neural Networks (CNN) project!
In this project, you will learn how to build a pipeline to process real-world, user-supplied images and to put your model into an app.
Given an image, your app will predict the most likely locations where the image was taken.

By completing this lab, you demonstrate your understanding of the challenges involved in piecing together a series of models designed to perform various tasks in a data processing pipeline. 

Each model has its strengths and weaknesses, and engineering a real-world application often involves solving many problems without a perfect answer.

### Why We're Here

Photo sharing and photo storage services like to have location data for each photo that is uploaded. With the location data, these services can build advanced features, such as automatic suggestion of relevant tags or automatic photo organization, which help provide a compelling user experience. Although a photo's location can often be obtained by looking at the photo's metadata, many photos uploaded to these services will not have location metadata available. This can happen when, for example, the camera capturing the picture does not have GPS or if a photo's metadata is scrubbed due to privacy concerns.

If no location metadata for an image is available, one way to infer the location is to detect and classify a discernable landmark in the image. Given the large number of landmarks across the world and the immense volume of images that are uploaded to photo sharing services, using human judgement to classify these landmarks would not be feasible.

In this project, you will take the first steps towards addressing this problem by building a CNN-powered app to automatically predict the location of the image based on any landmarks depicted in the image. At the end of this project, your app will accept any user-supplied image as input and suggest the top k most relevant landmarks from 50 possible landmarks from across the world.


## Project Instructions

### Getting started

You have two choices for completing this project. You can work locally on your machine (NVIDIA GPU highly recommended), or you can work in the provided Udacity workspace that you can find in your classroom.

#### Setting up in the Udacity Project Workspace
You can find the Udacity Project Workspace in your Udacity classroom, in the Project section.

1. Start the workspace by clicking on `Project Workspace` in the left menu in the page
2. When prompted on whether you want a GPU or not, please ANSWER YES (the GPU is going to make everything several times faster)

The environment is already setup for you, including the starter code, so you can jump right into building the project!

#### Setting up locally

This setup requires a bit of familiarity with creating a working deep learning environment. While things should work out of the box, in case of problems you might have to do operations on your system (like installing new NVIDIA drivers) that are not covered in the class. Please do this if you are at least a bit familiar with these subjects, otherwise please consider using the provided Udacity workspace that you find in the classroom.

1. Open a terminal and clone the repository, then navigate to the downloaded folder:
	
	```	
		git clone https://github.com/udacity/cd1821-CNN-project-starter.git
		cd cd1821-CNN-project-starter
	```
    
2. Create a new conda environment with python 3.7.6:

    ```
        conda create --name udacity_cnn_project -y python=3.7.6
        conda activate udacity_cnn_project
    ```
    
    NOTE: you will have to execute `conda activate udacity_cnn_project` for every new terminal session.
    
3. Install the requirements of the project:

    ```
        pip install -r requirements.txt
    ```

4. Install and open Jupyter lab:
	
	```
        pip install jupyterlab
		jupyter lab
	```

### Developing your project

Now that you have a working environment, execute the following steps:

>**Note:** Complete the following notebooks in order, do not move to the next step if you didn't complete the previous one.

1. Open the `cnn_from_scratch.ipynb` notebook and follow the instructions there
2. Open `transfer_learning.ipynb` and follow the instructions
3. Open `app.ipynb` and follow the instructions there

## Evaluation

Your project will be reviewed by a Udacity reviewer against the CNN project rubric.  Review this rubric thoroughly and self-evaluate your project before submission.  All criteria found in the rubric must meet specifications for you to pass.

## Project Submission

Your submission should consist of the github link to your repository.  Your repository should contain:
- The `landmark.ipynb` file with fully functional code, all code cells executed and displaying output, and all questions answered.
- An HTML or PDF export of the project notebook with the name `report.html` or `report.pdf`.

Please do __NOT__ include any of the project data sets provided in the `landmark_images/` folder.

### Ready to submit your project?

Click on the "Submit Project" button in the classroom and follow the instructions to submit!

## Dataset Info

The landmark images are a subset of the Google Landmarks Dataset v2.
