from django.conf.urls import url
from .views import list, selected, computation

urlpatterns = [
    url(r'^list/$', list, name='list'),
    url(r'^selected/$', selected, name='selected'),
    url(r'^computation/$', computation, name='computation'),
]
