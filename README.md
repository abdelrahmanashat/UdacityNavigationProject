# Udacity Navigation Project

In this project, we will implement a DQN agent with Unity's Banana Collector environment.

## 1. Environment Details: 

The simulation contains a single agent that navigates a large environment.  At each time step, it has four actions at its disposal:
- `0` - walk forward 
- `1` - walk backward
- `2` - turn left
- `3` - turn right
The state space has `37` dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  A reward of `+1` is provided for collecting a yellow banana, and a reward of `-1` is provided for collecting a blue banana. The environment is considered solved when we collect a reward of `13 or higher`.

## 2. Requirements for running the code:

#### Step 1: Install Numpy
Follow the instructions [here](https://numpy.org/install/) to install the latest version of Numpy.

#### Step 2: Install Pytorch and ML-Agents
If you haven't already, please follow the instructions in the [DRLND (Deep Reinforcement Learning Nano Degree) GitHub repository](https://github.com/udacity/deep-reinforcement-learning#dependencies) to set up your Python environment. These instructions can be found in `README.md` at the root of the repository. By following these instructions, you will install PyTorch, the ML-Agents toolkit, and a few more Python packages required to complete the project.

_(For Windows users)_ The ML-Agents toolkit supports Windows 10. While it might be possible to run the ML-Agents toolkit using other versions of Windows, it has not been tested on other versions. Furthermore, the ML-Agents toolkit has not been tested on a Windows VM such as Bootcamp or Parallels.

#### Step 3: Download the Banana Environment
For this project, you will __not__ need to install Unity - this is because we have already built the environment for you, and you can download it from one of the links below. You need only select the environment that matches your operating system:

- Linux: click [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
- Mac OSX: click [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
- Windows (32-bit): click [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
- Windows (64-bit): click [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

Then, place the file in the `p1_navigation/` folder in the DRLND GitHub repository, and unzip (or decompress) the file.

_(For Windows users)_ Check out this link if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

_(For AWS)_ If you'd like to train the agent on AWS (and have not enabled a virtual screen), then please use this link to obtain the "headless" version of the environment. You will not be able to watch the agent without enabling a virtual screen, but you will be able to train the agent. (To watch the agent, you should follow the instructions to enable a virtual screen, and then download the environment for the Linux operating system above.)

## 3. Explore the Environment
After you have followed the instructions above, open `Navigation.ipynb` (located in the `p1_navigation/` folder in the DRLND GitHub repository) and follow the instructions to learn how to use the Python API to control the agent.
