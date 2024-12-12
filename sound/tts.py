from transformers import pipeline

#import torch

synthesizer = pipeline("text-to-speech", "fishaudio/fish-speech-1.5")

synthesizer("Look I am generating speech in three lines of code!")
