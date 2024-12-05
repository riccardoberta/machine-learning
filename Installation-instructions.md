## Environment installation instructions

Here’s a step-by-step guide on how to set up a machine learning virtual environment using Conda on Windows (similar instructions works also for MacOS or Linux) with Visual Studio Code (VS Code) as IDE. The environment will be named "machine-learning" and will include Python, Jupyter, Pandas, NumPy, SciKit-Learn, and TensorFlow.

### Step 1: Install Conda

**Anaconda** is an open-source distribution of Python and R, widely used for data science, machine learning, and scientific computing. It simplifies package management and environment creation and includes several libraries.

1. Download and install Anaconda or Miniconda from the [official website](https://www.anaconda.com/products/individual). Miniconda is a lightweight version of Anaconda, recommended if you don’t need all the extra libraries.
2. Follow the installation instructions and make sure to check the option to Add Anaconda/Miniconda to the PATH during installation.

After installation, verify by opening a terminal and typing:

```bash
conda --version
```

You should see the version of Conda installed.

### Step 2: Install Visual Studio Code (VS Code)

1. Download and install **Visual Studio Code** from the [official website](https://code.visualstudio.com/)
2. After installation, open VS Code and install the Python extension by Microsoft. You can find it in the **Extensions Marketplace** (click on the square icon in the sidebar or press `Ctrl+Shift+X`).

### Step 3: Create a Virtual Environment

A virtual environment is **an isolated workspace** that allows you to install and manage specific versions of Python and its libraries without affecting other projects or the system's global Python setup. This isolation helps ensure that each project has the exact dependencies it needs, avoiding conflicts between different projects. Now, let's create the virtual environment using Conda with the required packages.

1. Open the command prompt.
2. Type the following command to create a new environment called `machine-learning`. You can specify the Python version you need (e.g., `python=3.12`). Adjust this version if needed.

   ```bash
   conda create --name machine-learning python=3.12
   ```

3. Once the environment is created, activate it:

     ```bash
     conda activate machine-learning
     ```

4. Now, install the necessary packages (**Pandas**, **NumPy**, **Scikit-Learn**, and **TensorFlow**):

   ```bash
   conda install pandas numpy scikit-learn
   ```

5. For TensorFlow, it’s better to install via pip (obviously, you need to stay in the `machine-learning` environment):

   ```bash
   pip install tensorflow
   ```

### Step 4: Set Up Visual Studio Code

1. Open Visual Studio Code
2. Go to **View > Command Palette** (`Ctrl+Shift+P`).
3. Type and select **Python: Select Interpreter**.
4. In the list of interpreters, select the one associated with your `machine-learning` Conda environment

### Step 5: Verify Installation

1. Open the terminal in VS Code
2. Create a new file (e.g., `test_ml.py`) and write a simple Python script using the installed libraries, like:

   ```python
   import numpy as np
   import pandas as pd
   import sklearn
   import tensorflow as tf

   print(np.__version__)
   print(pd.__version__)
   print(sklearn.__version__)
   print(tf.__version__)
   ```

### Step 6: Managing the Environment

To install additional packages later, you can always activate the environment on the command prompt and use either `conda install <package>` or `pip install <package>`.

### Step 7: Install Jupyter

**Jupyter** is an open-source tool that provides interactive notebooks for writing and running code, visualizing data, and documenting workflows. It supports multiple languages, with Python being the most popular. To integrate Jupyter in VS Code, you need to install Juopyter, the Jupyter extension and configure it to work with your Conda environment. Here’s how you can set up Jupyter in VS Code:

1. Open a command prompt
2. Activate your `machine-learning` Conda environment:

   ```bash
   conda activate machine-learning
   ```

3. Install Jupyter in the environment:
   
   ```bash
   pip install jupyter
   ```

4. Open Visual Studio Code
5. Go to the Extensions Marketplace by clicking on the square icon on the left sidebar or pressing `Ctrl+Shift+X`
6. Search for the Jupyter extension by Microsoft and click Install

### Step 8: Create a Jupyter Notebook

1. In Visual Studio Code, create a new file and save it with the `.ipynb` extension, for example, `notebook.ipynb`
2. VS Code will automatically detect the Jupyter extension and open the notebook interface
3. In the top-right corner, you will see a kernel picker. Make sure your `machine-learning` environment is selected as the kernel. If it’s not selected:
   - Click the kernel picker and choose Select Kernel.
   - Choose the interpreter associated with the `machine-learning` Conda environment

You can now start writing Jupyter code cells in VS Code. Each cell is executed individually, and you can mix code, text, and visualizations in the notebook:

1. Write some code in a cell, for example:

   ```python
   import numpy as np
   import pandas as pd
   
   df = pd.DataFrame({
       'A': np.random.rand(100),
       'B': np.random.rand(100)
   })
   
   df.head()
   ```

2. To run the cell, press the Run Cell button (small play icon next to the cell) or use `Shift+Enter` to execute the cell.

By integrating Jupyter with VS Code, you get the flexibility of a notebook interface while taking advantage of VS Code's powerful features for code editing and debugging! Now you’re ready to develop machine learning models with VS Code, Conda, Jupyter and your `machine-learning` virtual environment!
