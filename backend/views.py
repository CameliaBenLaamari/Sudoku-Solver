from django.http.response import HttpResponse
from django.shortcuts import render

from backend.forms import UploadForm
from backend.main import ReadGrid, SolveSudoku
from backend.models import SudokuGrid

# Create your views here.


def index(request):
    if request.method == 'POST':
        form = UploadForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            sudoku = form.instance
            grid = ReadGrid(sudoku.grid.url)
            solution = SolveSudoku(sudoku.grid.url)
            solution = compare(grid, solution)
            context = {
                'form': form,
                'sudoku': sudoku,
                'grid': grid,
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


def compare(grid, solution):
    for i in range(9):
        for j in range(9):
            if (grid[i][j] > 0):
                solution[i][j] = solution[i][j] + 10
    return solution
