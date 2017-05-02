import os, base64
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
from django.shortcuts import render
from django.http import HttpResponseRedirect
from django.core.urlresolvers import reverse
from .models import Document
from .forms import DocumentForm
from sklearn import preprocessing, tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier


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
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # home = '/home/piyush/Devel/finalProject'
    csvpath = BASE_DIR + csvfile

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
    plt.xticks(rotation=90)
    plt.savefig(image, format='png')
    image.seek(0)
    plot_url = base64.b64encode(image.getvalue())

    return render(request, 'graph.html', {'plot_url': plot_url})


def learning(request):
    if request.method == "POST":
        csvfile = request.POST.get('csvfile')

        df = pd.read_csv(csvfile)
        df = df.dropna()
        le = preprocessing.LabelEncoder()
        features = df.columns.values
    for feature in features:
        le.fit(df[feature])
        df[feature] = le.transform(df[feature])
    train, test = train_test_split(df, test_size=0.05, random_state=42)
    train_features = train[['Age', 'Sex', 'BodyPart', 'Symptom_1', 'Symptom_2']].values
    target = train['Disease_1'].values
    X_features = ['Age', 'Sex', 'BodyPart', 'Symptom_1', 'Symptom_2']
    X_test = test[X_features]
    y_test = test['Disease_1']
    y_test_array = y_test.as_matrix()
    X_test_array = X_test.as_matrix()

    my_tree = tree.DecisionTreeClassifier()
    my_tree = my_tree.fit(train_features, target)
    prediction_dt = my_tree.predict(X_test_array)
    accuracy_dt = accuracy_score(y_test_array, prediction_dt) * 100

    neigh = KNeighborsClassifier()
    neigh.fit(train_features, target)
    prediction_neigh = neigh.predict(X_test_array)
    accuracy_neigh = accuracy_score(y_test_array, prediction_neigh) * 100

    ada = AdaBoostClassifier()
    ada.fit(train_features, target)
    prediction_ada = ada.predict(X_test_array)
    accuracy_ada = accuracy_score(y_test_array, prediction_ada) * 100

    rf = RandomForestClassifier()
    rf.fit(train_features, target)
    prediction_rf = rf.predict(X_test_array)
    accuracy_rf = accuracy_score(y_test_array, prediction_rf) * 100

    # mlp = MLPClassifier()
    # mlp.fit(train_features, target)
    # prediction_mlp = mlp.predict(X_test_array)
    # accuracy_mlp = accuracy_score(y_test_array, prediction_mlp) * 100

    context = {
        'accuracy_dt': accuracy_dt,
        'accuracy_neigh': accuracy_neigh,
        'accuracy_ada': accuracy_ada,
        'accuracy_rf': accuracy_rf,
    }

    return render(request, 'learning.html', context)
