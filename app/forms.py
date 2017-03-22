# -*- coding: utf-8 -*-

from django import forms


class DocumentForm(forms.Form):
    csvfile = forms.FileField(
        label='Select a file'
    )
