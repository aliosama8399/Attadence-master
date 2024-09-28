from django.db import models

# Create your models here.


class Hall(models.Model):
    id = models.BigAutoField(primary_key=True)
    name = models.CharField(max_length=255,unique=True)


class Subject(models.Model):
    id = models.BigAutoField(primary_key=True)
    name = models.CharField(max_length=255,unique=True)