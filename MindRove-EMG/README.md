# MindRove-EMG #
For the EMG wristband from mindrove. See `NML-GUI` section for [Quick Start instructions](#nml-gui). For post-processing recordings in MATLAB, see the [MATLAB](#matlab) section. **Recommended Python version 3.10!**

## Installation (using `pip`) ##

This repository can be installed as a Python package using an editable install. This makes it easy to develop, modify source files, and immediately test changes without reinstalling. To simply use this package in another project, you can add it to `requirements.txt` as `nml_wtf_mindrove`. To install directly via `pip` from `CLI`, use:
```bat
pip install nml-wtf-mindrove
```

### 1. Clone the repository

Using SSH:

```bat
git clone git@github.com:Neuro-Mechatronics-Interfaces/MindRove-EMG.git
cd MindRove-EMG
```

If you do not have SSH keys configured, you can also use HTTPS:

```bat
git clone https://github.com/Neuro-Mechatronics-Interfaces/MindRove-EMG.git
cd MindRove-EMG
```

### 2. Create and activate a virtual environment

#### Windows
_Note: Python 3.12 will give you headaches with the `pyqtgraph` package, stick to 3.10 to be safe!_
```bat
py -3.10 -m venv .venv
.venv\Scripts\activate
```

#### Linux/macOS

```bat
python3 -m venv .venv
source .venv/bin/activate
```

Your terminal prompt should show:
```bat
(.venv) C:\Your\Local\Project
```

### 3. Install the repository in editable mode
From inside the repo:
```bat
pip install -e .
```

This installs the package under the name:

`nml_wtf_mindrove`

All Python modules under `nml/`, `mindrove/`, and `mindrove/` are now importable. Assets in `assets/`, `lib/`, and `configurations/` are included automatically. Note that this includes static compiled libraries for `ViGEM`, which were built on a Windows-11 64-bit OS. `ViGEM` is built to run on Windows (so any `GamepadController` etc. that uses it, won't work on other platforms). 

### 4. Run the main GUI application

```bat
python app.py
```

You can also import modules anywhere inside the same virtual environment:

```bat
from nml.application import Application
```

---

## Starter Scripts ##
If you installed via `pip` from `PyPI`, then after `pip install nml-wtf-mindrove`, you can test the install using a couple of scripts. First, make sure your band is on and connected to your computer via WiFi. Then, from the same terminal: 
```bat
nwm-gui
```
This will launch a stream GUI with a view different interfaces you can click through as well as ways to save/log data.  
```bat
nwm-demo-latch
```
This uses `ViGEM` to emulate an XBOX controller at kernel level on windows, which presses the `A` button when you flex, latching the assertion state and toggling it again when you flex again. There is a directional option from the combo box which then should only exit the asserted state if you flex in the opposite direction of the assertion. 
```bat
nwm-demo-stretch
```
This has the same interface as `nwm-demo-latch` but has continuous decode outputs meant to connect via LSL to a receiver that controls a stretch robot in the lab. 

---

## Installation (without `pip`) ##
Make sure you have your `ssh` key added to your GitHub account, and your `ssh-agent` is running in your local terminal environment.  
```bash
git clone git@github.com:Neuro-Mechatronics-Interfaces/MindRove-EMG.git && cd MindRove-EMG && .venv/Scripts/activate
```
Depending on your installed version of `venv` / Python, the `./.venv/Scripts/activate` command may change slightly (it will also depend on what your terminal environment is).  
* _Note: if you have successfully activated the virtual environment, then your terminal should say `(.venv)` now._

## Quick Start ##
Use the main `app.py` as an entry point:  
```bash
python app.py
```

### Compiling Libraries ### 
If you need to compile the `ViGEm` libraries used by the Python API, you'll first need to download and install `ViGEmClient` from GitHub. Once you've compiled the libraries, you can compile from a `MSVC Developer Command Terminal` (I used MSVC 2022 but theoretically other versions should work). Use a comand like the following (note that this assumes you cloned `ViGEmClient` to `C:\MyRepos\Libraries\ViGEmClient`, that you built using `CMake` in `C:\MyRepos\Libraries\ViGEmClient\build`, and that you compiled the resulting Visual Studio Solution `ALL_BUILD` solution in `Release` mode):
```batch
cl /LD /I"C:\MyRepos\Libraries\ViGEmClient\include" /MD "src\vigem_gamepad.c" /Fo"lib\vigem_gamepad.obj" /Fe"lib\vigem_gamepad.dll" /link /LIBPATH:"C:\MyRepos\Libraries\ViGEmClient\build\Release" ViGEmClient.lib Setupapi.lib User32.lib
```

#### ViGEm Button Mapping Quick Reference ####
The hex code integers in the right column are bit masks for the corresponding button-press in the left column. The encoding/decoding of these masks can be handled by `nml.vigem.ViGEmEnum` static `encode` and `decode` methods, respectively.  
| **Button**         | **Code (Hex)** |
|---------------------|----------------|
| A                  | 0x1000         |
| B                  | 0x2000         |
| X                  | 0x4000         |
| Y                  | 0x8000         |
| LB                 | 0x0100         |
| RB                 | 0x0200         |
| Back               | 0x0020         |
| Start              | 0x0010         |
| Left Thumb         | 0x0040         |
| Right Thumb        | 0x0080         |
| D-Pad Up           | 0x0001         |
| D-Pad Down         | 0x0002         |
| D-Pad Left         | 0x0004         |
| D-Pad Right        | 0x0008         |
| ALL                | 0xFFFF         |


## Use ##  
After you install the MindRove SDK (the --recurse-submodules step in first line of installer instructions above), and activate your virtual environment (the third line in the installer instructions above)/install pip dependencies, you can use these preliminary tools.  

### Realtime Stream/Record ###
This will make squiggles for EMG channels play on the screen, assuming you've correctly connected to the device via Wi-Fi after turning it on.  
```bash
python realtime_plot.py --file my_data.bin
```

### TSV Log Structure ###
Based on the saved data structure, the channels probably are mapped something like this:  
| Column Index (0-indexed) | Data Type            |
|--------------------------|----------------------|
| 0-7                      | EEG/EMG Channels     |
| 8-17                     | Resistance Channels  |
| 18                       | Battery              |
| 19                       | Beep (1), Boop (2)   |
| 20-22                    | Accelerometer        |
| 23-25                    | Gyroscope            |
| 26                       | Counter              |
| 27                       | Timestamp            |
| 28                       | Markers              |

## NML GUI ##
To run the main NML GUI, use:  
```bash
python app.py
```
If you plan to re-open a bunch of times and want to set the default subject stub in the filename, you can specify the `file` option, e.g.  
```bash
python app.py --file=max --suffix=4
```
This example would default the filename textbox to `data/max_2025_02_19` (as of writing this on 2025-02-19). The suffix (appended at end) would default to 4. 

You can set the desired filename, and whether to save streamed data using the checkbox on the main interface.  
To open the stream windows and see signals, click the button on the left middle part of the main GUI. Once you click this button, the game button for "Arrows" will be enabled. You can click this button to populate a prompt game with 4 directional arrows, which will start when you click the "Start" button and end when you click the "Stop" button. Event codes related to this game are on Markers (0-indexed) column 28 as described in the next section.

### Synthetic Mode ###
To run synthetic data streams without any device, the default settings will change the board to sample at 250 samples/sec (with the EMG board the default is 500 samples/sec). It also changes the channel scalings/and labels slightly, which is partially accounted for in the GUI but not completely. However it can still be useful to use the synthetic board for test/development purposes. This can be done as follows:  
```bash
python app.py --synth 1
```

Currently, the GUI is not configured to handle playback from previous data files.  

### Event Codes ###
| Event | Code |
| ----- | ---- |
| Up Arrow Shown | 2 |
| Right Arrow Shown | 3 |
| Down Arrow Shown | 4 |
| Left Arrow Shown | 5 |
| Rest | 6 |
| Up Key Press | 7 |
| Right Key Press | 8 |
| Down Key Press | 9 |
| Left Key Press | 10 |
| Direction Game Start | 11 |
| Direction Game End | 12 |
| Pac-Man Game Open | 13 |
| Pac-Man Level Start | 14 |
| Pac-Man Pellet Consumed | 15 |
| Pac-Man Died | 16 |
| Pac-Man Consumed Powerup | 17 |
| Pac-Man Consumed Ghost | 18 |
| Pac-Man Level Clear | 19 |
| Pac-Man Level Up | 20 |
| Pac-Man Level Stop | 21 |
| Real-Time Plot Marker-0 | 30 |
| Real-Time Plot Marker-1 | 31 |
| Real-Time Plot Marker-2 | 32 |
| Real-Time Plot Marker-3 | 33 |
| Real-Time Plot Marker-4 | 34 |
| Real-Time Plot Marker-5 | 35 |
| Real-Time Plot Marker-6 | 36 |
| Real-Time Plot Marker-7 | 37 |
| Real-Time Plot Marker-8 | 38 |
| Real-Time Plot Marker-9 | 39 |
| Real-Time Rest Calibration Click | 40 |
| Real-Time Rest Calibration Start | 41 |
| Real-Time Rest Calibration End | 42 |
| Real-Time Threshold Calibration Click | 43 |
| Real-Time Threshold Calibration Start | 44 |
| Real-Time Threshold Calibration End | 45 |
| Fitts-Task Start | 50 |
| Fitts-Task Trial Start | 51 | 
| Fitts-Task Move | 52 |
| Fitts-Task Rest | 53 |
| Fitts-Task Trial Complete | 54 |
| Fitts-Task Stop | 55 |
| Covariance-State Changed to 0 | 60 | 
| Covariance-State Changed to 1 | 61 | 
| Covariance-State Changed to 2 | 62 | 
| Covariance-State Changed to 3 | 63 | 
| Covariance-State Changed to 4 | 64 | 
| Covariance-State Changed to 5 | 65 | 
| Covariance-State Changed to 6 | 66 | 
| Covariance-State Changed to 7 | 67 | 
| Covariance-State Changed to 8 | 68 |
| Covariance-State Changed to 9 | 69 |  
| WebSocket Message | 1024 + [message_value: in range 0-256] |
| UDP State Message With Invalid "Value" | 2047 |
| UDP State Message | 2048 + [message_value: in range 0-512: Default is the Counter value `rem(sampleCounter,512)` where sampleCounter is the latest sample counter value from SAGA-A] |

* _About WebSocket Messages_: This is set up to make it easy to send game-keyed values from web-hosted Javascript tasks (like Max's https://reaction-task.nml.wtf "games"), via WebSocket interface if the MindRove band is running the main `nml.application` interface on the same device hosting the web task. Briefly:
  + **Basic Reaction-Task**: Codes are task states, which range from values of -3 to +4. The values are shifted over by adding 1027 to them before the transmission, so a received value of `1024` would correspond to Basic Reaction-Task state value of `-3`. A value of `1032` is sent at the beginning of this task and a value of `1033` is sent at the end of the taks. States are keyed as follows:
    * **TIMEOUT** (-3) = 1024
    * **IDLE** (-2) = 1025
    * **READY** (-1) = 1026
    * **WAIT_ASSERTION_CUE** (0) = 1027
    * **WAIT_ASSERTION_REACTION** (1) = 1028
    * **WAIT_DEASSERTION_REACTION** (2) = 1029
    * **HOLD_ASSERTION** (3) = 1030
    * **DEBOUNCE** (4) = 1031

  + _About UDP State Messages_: These are meant to synchronize acquisition loops in MATLAB (e.g. using Max's TMSi dual interface for poly5 acquisition) with the MindRove records. 
  
## MATLAB ## 
In the MATLAB folder, there is an enumeration class (`MindRoveCode`) and a few utility functions for reading and plotting saved recordings. The main thing you might want to change in these utilities is `options.InputRoot` in the `plot_mindrove_features.m` file--it should default to the local relative path for data files but if you have saved your data elsewhere then you should update this option to make reading it easier.  
