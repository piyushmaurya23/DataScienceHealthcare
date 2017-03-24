import pandas as pd
import seaborn as sns
from io import BytesIO
import base64
import matplotlib.pyplot as plt
from django.shortcuts import render
from django.http import HttpResponseRedirect
from django.core.urlresolvers import reverse

from .models import Document
from .forms import DocumentForm


def list(request):
    # Handle file upload
    if request.method == 'POST':
        form = DocumentForm(request.POST, request.FILES)
        if form.is_valid():
            newcsv = Document(csvfile=request.FILES['csvfile'])
            newcsv.save()

            # Redirect to the document list after POST
            return HttpResponseRedirect(reverse('app:list'))
    else:
        form = DocumentForm()  # A empty, unbound form

    # Load documents for the list page
    documents = Document.objects.all()

    # Render list page with the documents and the form
    return render(
        request,
        'list.html',
        {'documents': documents, 'form': form}
    )


def selected(request):
    # csvfile = request.GET.get('csvfile')
    if request.method == "POST":
        csvfile = request.POST.get('csvfile')
    home = '/home/piyush/Devel/finalProject'
    csvpath = home + csvfile

    context = {
        'csvfile': csvfile,
        'csvpath': csvpath,
    }

    return render(request, 'selected.html', context)


def computation(request):
    if request.method == "POST":
        csvfile = request.POST.get('csvfile')

    df = pd.read_csv(csvfile)
    df = df.dropna()
    numerics = ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    numeric_df = df.select_dtypes(include=numerics)

    num_list = numeric_df.columns.values

    # if num_list:
    num_mean = []
    num_median = []
    num_std = []
    for num in num_list:
        num_mean.append(numeric_df[num].mean())
        num_median.append(numeric_df[num].median())
        num_std.append(numeric_df[num].std())

    numList = zip(num_list, num_mean, num_median, num_std)

    context = {
        'numlist': numList,

    }

    return render(request, 'computation.html', context)


def visualisation(request):
    if request.method == "POST":
        csvfile = request.POST.get('csvfile')

    df = pd.read_csv(csvfile)
    df = df.dropna()
    numerics = ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    non_numeric_df = df.select_dtypes(exclude=numerics)
    non_numeric_list = non_numeric_df.columns.values
    # image = BytesIO()
    # sns.set_style("dark")
    # sns.countplot(non_numeric_df['Product'])
    # y = [1, 2, 3, 4]
    # x = [3, 2, 4, 5]
    # plt.plot(x, y)
    # plt.savefig(image, format='png')
    # image.seek(0)
    # plot_url = base64.b64encode(image.getvalue())
    context = {
        'csvfile': csvfile,
        'non_numeric_list': non_numeric_list,
    }

    return render(request, 'visualisation.html', context)


def graph(request):
    if request.method == "POST":
        csvfile = request.POST.get('csvfile')
        attribute = request.POST.get('attribute')

    df = pd.read_csv(csvfile)
    df = df.dropna()

    image = BytesIO()
    sns.set_style("dark")
    sns.countplot(df[attribute])
    plt.savefig(image, format='png')
    image.seek(0)
    plot_url = base64.b64encode(image.getvalue())

    return render(request, 'graph.html', {'plot_url': plot_url})


def learning(request):
    if request.method == "POST":
        csvfile = request.POST.get('csvfile')

    context = {
        
    }

    return render(request, 'learning.html', context)
