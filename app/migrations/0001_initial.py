# -*- coding: utf-8 -*-
# Generated by Django 1.10.6 on 2017-03-22 08:41
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Document',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('csvfile', models.FileField(upload_to='csv/%Y/%m/%d')),
            ],
        ),
    ]
