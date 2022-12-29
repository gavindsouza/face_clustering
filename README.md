### Steps to run
1. Clone this repository
```
$ git clone https://github.com/gavindsouza/face_clustering
```

2. Change to working directory & Install dependencies
```
$ cd face_clustering
```

Ideally create a virtual environment by
```
$ python3 -m venv venv
$ source venv/bin/activate
```

Install dependencies by
```
$ pip3 install -r requirements.txt
```

Note: Currently all under `requirements.txt` are dev-requirements

3. Acquire test dataset (Optional)

Link: `https://www.kaggle.com/gasgallo/faces-data`

Make sure test folder has no sub dirs or other files, only images.

4. Run `test_pipeline.py`
```
$ python3 test_pipeline.py
```
Results saved in 'temp_files/results.csv'