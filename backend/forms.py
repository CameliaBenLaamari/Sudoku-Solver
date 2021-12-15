from django import forms
from .models import *


class UploadForm(forms.ModelForm):

    class Meta:
        model = SudokuGrid
        fields = ('grid',)
        labels = {
            'grid': '',
        }
