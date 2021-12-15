from django.http.response import HttpResponse
from django.shortcuts import render

from backend.forms import UploadForm
from backend.main import SolveSudoku
from backend.models import SudokuGrid

import time

import logging

# Create your views here.


def index(request):
    if request.method == 'POST':
        form = UploadForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            sudoku = form.instance
            solution = SolveSudoku('https://www.101computing.net/wp/wp-content/uploads/sudoku-grid.png')
            context = {
                'form': form,
                'sudoku': sudoku,
                'solution': solution
            }
            return render(request, 'index.html', context)
    else:
        form = UploadForm()
    return render(request, 'index.html', {'form': form})


def result(request):
    if request.method == 'GET':
        grids = SudokuGrid.objects.all()
        return render(request, 'result.html', {'grids': grids})
    return HttpResponse('Successfully uploaded!')
