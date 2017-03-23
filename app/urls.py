from django.conf.urls import url
from .views import list, selected, computation, visualisation, graph

urlpatterns = [
    url(r'^list/$', list, name='list'),
    url(r'^selected/$', selected, name='selected'),
    url(r'^computation/$', computation, name='computation'),
    url(r'^visualisation/$', visualisation, name='visualisation'),
    url(r'^graph/$', graph, name='graph'),
]
