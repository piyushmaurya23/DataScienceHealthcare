from django.db import models


# Create your models here.

class Document(models.Model):
    csvfile = models.FileField(upload_to='csv/%Y/%m/%d')
