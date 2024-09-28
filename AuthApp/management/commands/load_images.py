# AuthApp/management/commands/load_images.py

from django.core.management.base import BaseCommand
from AuthApp.views import load_training_images_in_background
import threading

class Command(BaseCommand):
    help = 'Loads training images in the background when the server starts'

    def handle(self, *args, **kwargs):
        # Start a background thread to load images
        thread = threading.Thread(target=load_training_images_in_background)
        thread.setDaemon(True)  # Daemon thread will stop when main thread stops
        thread.start()

        self.stdout.write(self.style.SUCCESS('Loading training images in the background...'))
