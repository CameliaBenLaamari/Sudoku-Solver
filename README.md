# Sudoku Solver
Sudoku Solver is an AI application for solving Sudoku grids using Backtracking algorithm and OCR.


## How to use
1. Clone the repo on your local machine using `git clone https://github.com/CameliaBenLaamari/Sudoku-Solver.git`.
2. Install the following packages:
  - numpy
  - opencv-python
  - tensorflow
  - pillow
  - django-mathfilters
``` 
  pip install numpy
  pip install opencv-python
  pip install tensorflow
  pip install pillow
```
3. Run `python manage.py makemigrations` to check if there are any committed changes.
4. Run `python manage.py migrate` to push these changes (if any).
6. Finally run `python manage.py runserver` to run the app on `http://localhost:8000/`.
7. Upload an image of a sudoku grid. The better the quality, the better the result you get.

![screenshot](https://user-images.githubusercontent.com/76062686/147242753-68e09b18-6264-4b66-993a-5056ddd0368f.png)

Training dataset source: https://www.kaggle.com/karnikakapoor/digits

## Authors
- Ahmed Kallel
- Camelia Ben Laamari

<br/><br/>

© 2021 SUP'COM.
