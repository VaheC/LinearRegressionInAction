# LinearRegressionInAction
This project uses linear regression to predict air temperature using maintenance dataset from UCI.

# Files' description
model.py contains model building code which actually 
generates all the pickle file in the directory.\
app.py contains web app code.\
templates folder contains web app templates.\
LinearRegressionProject.ipynb contains exploratory data analysis and feature selection procedure in detail.

## Deployment
At this moment the project is not deployed.

## Installation

If you have all packages from requirements.txt 
installed on your pc , then you can run all the codes
present in this project. 
    
## Run Locally
Install anaconda on your pc.\
Create a virtual environment in anaconda prompt. 

```bash
  conda -n env_name python=version

  example: conda -n sg_credit python=3.7

```
Python version can be taken from the requirements.txt file.

Activate the virtual environment in anaconda prompt

```bash
  activate env_name 

  example: activate sg_credit 
```

Clone the project in command prompt

```bash
  git clone https://link-to-project
```

Go to the cloned project directory on your pc 
in anaconda prompt

```bash
  cd directory-of-project

  example: cd C:\Users\Desktop\SouthGermanCredit
```

Install dependencies in anaconda prompt

```bash
  pip install -r /path/to/requirements.txt
```

Run the model building code in anaconda prompt

```bash
  python model.py
```

Start the web app on your pc from anaconda prompt

```bash
  python app.py
```
Use http shown in anaconda prompt (the last line of 
output after running the code above) to open the web
app in a web browser.

```bash
  example: http://127.0.0.1:5000/
```
## Acknowledgements

- [How to write a Good readme](https://readme.so)

