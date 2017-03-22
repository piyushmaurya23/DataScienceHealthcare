import pandas as pd

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
    numerics = ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    numeric_df = df.select_dtypes(include=numerics)
    num_mean = numeric_df.mean()

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

# def visualisation(request):
#     if request.method == "POST":
#         csvfile = request.POST.get('csvfile')
#
#     df = pd.read_csv(csvfile)

# Create your views here.

# def upload(request):
#     if request.POST and request.FILES:
#         csvfile = request.FILES['csv_file']
#         dialect = csv.Sniffer().sniff(codecs.EncodedFile(csvfile, "utf-8").read(1024))
#         csvfile.open()
#         reader = csv.reader(codecs.EncodedFile(csvfile, "utf-8"), delimiter=',', dialect=dialect)
#
#     return render(request, "index.html", locals())
