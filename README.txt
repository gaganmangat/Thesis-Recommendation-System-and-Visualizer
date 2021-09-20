This Readme file is for Windows

REQUIRED PACKAGES
requirements.txt
virtualenv

REQUIRED SOFTWARE:
postgresql - Also add postgresql to the path
Microsoft Visual C++ 14.0 or above

SETUP TO RUN:
1. Download zip file to your local machine
2. Extract the zip file
3. Open cmd prompt
4. Goto that Path

Example-
	cd ~/Desktop/g11-code

STEPS TO RUN:
1. Create a new virtual environment in that directory
	virtualenv env

2. Activate the virtual environment using the following command
	env\Scripts\activate

3. Install all dependencies using the following command
	pip install -r requirements.txt
	
4. Then navigate to /sample/sample/
	cd sample/sample/

5. For creating database in postgres run the following commands
	i)  psql -U postgres
	ii) Enter the password for the database (This password should match the password present in sample/
		settings.py file)
	iii) CREATE DATABASE "Articles_IITK";
	iv) exit

6. For loading data into database run the following commands
	python manage.py makemigrations home
	python manage.py migrate home
	python manage.py runscript load

7. For running the server
	python manage.py runserver



****** Also the project folder contains Jupyter_Notebooks Folder, This is provided only for reference ***********

