from django.db import models

# Create your models here.
class Destination(models.Model):
    id = models.IntegerField(primary_key=True)
    Title = models.CharField(max_length=3000)
    Author = models.CharField(max_length=1000)
    Supervisor = models.CharField(max_length=1000)
    Degree = models.CharField(max_length=20)
    Department = models.CharField(max_length=30)
    Abstract = models.CharField(max_length=300000)
    URL = models.CharField(max_length=500, null=True)
    Date1 = models.CharField(max_length=20,null=True)

    def __str__(self):
        return self.Title
