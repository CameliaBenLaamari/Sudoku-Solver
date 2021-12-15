from django.db import models

# Create your models here.


class SudokuGrid(models.Model):
    grid = models.ImageField(upload_to='images/')
